"""V2X-Seq leader evaluation from saved event tables (scripts/v2xseq_extract.py output), no tuning.

Models: rules A / B, LR(d,v,a), LR(d,v), MLP(d,v,a), LSTM / Transformer on (x,y,v,a) and on (d,v,a).
  external : fitted once on the FLUID FI leaders (--fluid-event-dir), applied unchanged to the V2X-Seq leaders;
             subsets all / dilemma-zone band (d/v 2.5-5.5 s)
  loio     : leave-one-intersection-out inside V2X-Seq; sites with < --min-site events are never tested (training
             only); per-site and pooled metrics, subsets all / dz / v0 < 10 / v0 >= 10 m/s
  curve    : LOIO learning curve: per fold keep 25/50/75/100 % of the training yellow cycles (--curve-reps draws),
             LR(d,v,a), LR(d,v), LSTM(d,v,a), Transformer(d,v,a) with one seed; pooled macro-F1 per draw.
             Draw batches can be run separately (--rep-start 5 --curve-reps 5 --fracs 0.25,0.5,0.75); curve.csv is
             rebuilt from every curve_draws*.csv in --out. Spawn pool (fork deadlocks after torch ran in the parent).
All CIs: cycle bootstrap (ev['cycle']), paired differences vs rule A. --exclude drops listed (segment, track_id).

  .venv/bin/python scripts/v2xseq_eval.py --event-dir data/raw/v2xseq/results/leaders_v4 --out results/v2xseq/eval/v4 \
      --tasks external,loio
"""
import argparse
import json
import os
import sys
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import bench_eval
from dzp.bench_models import LogRegDV, LogRegDVA, LSTMSeq, MLPDVA, RuleA, RuleB, TransformerSeq
from run_benchmark import at
from seq_dva import LSTMDVA, TransformerDVA

DZ_LO, DZ_HI, V_SPLIT = 2.5, 5.5, 10.0
_D = {}


def model_list(n_seeds):
    return [(RuleA, {}), (RuleB, {"decel": 3.0}), (LogRegDVA, {}), (LogRegDV, {}), (MLPDVA, {}),
            (LSTMSeq, {"n_seeds": n_seeds}), (TransformerSeq, {"n_seeds": n_seeds}),
            (LSTMDVA, {"n_seeds": n_seeds}), (TransformerDVA, {"n_seeds": n_seeds})]


def load(event_dir):
    ev = pd.read_csv(f"{event_dir}/events_through.csv")
    feat = pd.read_csv(f"{event_dir}/features_through.csv")
    seq = {0.0: np.load(f"{event_dir}/sequences_through.npz")["lead+0.0"]}
    F, S = at(feat, seq, 0.0, np.arange(len(ev)))
    ok = F["t_r"].notna().values           # yellow still on at the segment end -> no recorded red -> t_r unknown
    if (~ok).any():
        print(f"dropping {int((~ok).sum())} events without a recorded red onset (t_r NaN)", flush=True)
    return ev[ok].reset_index(drop=True), F[ok].reset_index(drop=True), S[ok]


def load_tables(event_dir, exclude=""):
    ev, F, S = load(event_dir)
    if "segment" not in ev.columns:          # FLUID / SinD tables: recording = group
        ev["segment"] = ev["group"]
    if exclude:
        ex = pd.read_csv(exclude)
        key = set(zip(ex["segment"], ex["track_id"].astype(int)))
        keep = np.array([(s, int(t)) not in key for s, t in zip(ev["segment"], ev["track_id"])])
        print(f"excluding {int((~keep).sum())} events", flush=True)
        ev, F, S = ev[keep].reset_index(drop=True), F[keep].reset_index(drop=True), S[keep]
    return ev, F, S


def subsets(ev, F):
    tt, v0 = F["tt"].values, F["v"].values
    return {"all": np.ones(len(ev), bool), "dz": (tt >= DZ_LO) & (tt <= DZ_HI), "v<10": v0 < V_SPLIT, "v>=10": v0 >= V_SPLIT}


def external(args, ev, F, S, out):
    fe = pd.read_csv(f"{args.fluid_event_dir}/events_through.csv")
    ff = pd.read_csv(f"{args.fluid_event_dir}/features_through.csv")
    Ftr, Str = at(ff, {0.0: np.load(f"{args.fluid_event_dir}/sequences_through.npz")["lead+0.0"]}, 0.0, np.arange(len(fe)))
    ytr = fe["label"].values.astype(int)
    assert len(fe) == 88, "expected the 88 FLUID FI leaders"
    P = {}
    for M, kw in model_list(args.n_seeds):
        m = M(**kw).fit(Ftr, Str, ytr)
        P[M.name] = (m.proba(F, S), m.score(F, S))
        print(f"[external] {M.name} fitted", flush=True)
    rows = []
    for name, mask in subsets(ev, F).items():
        if name.startswith("v"):
            continue
        b = bench_eval.bootstrap(ev, P, mask, args.n_boot, ref=RuleA.name)
        rows.append(b.assign(subset=name))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(f"{out}/external.csv", index=False)
    pd.DataFrame({k: v[0] for k, v in P.items()}).assign(segment=ev["segment"], track_id=ev["track_id"]).to_csv(f"{out}/preds_external.csv", index=False)
    print(df[df.metric == "macro_f1"].round(3).to_string(), flush=True)


def loio_folds(ev, min_site):
    sites = ev["group"].value_counts()
    test_sites = sorted(sites.index[sites >= min_site])
    return [(s, np.where(ev["group"].values != s)[0], np.where(ev["group"].values == s)[0]) for s in test_sites]


def loio(args, ev, F, S, out):
    y = ev["label"].values.astype(int)
    folds = loio_folds(ev, args.min_site)
    tested = np.zeros(len(ev), bool)
    for _, _, te in folds:
        tested[te] = True
    P = {}
    for M, kw in model_list(args.n_seeds):
        p, sc = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for site, tr, te in folds:
            m = M(**kw).fit(F.iloc[tr], S[tr], y[tr])
            p[te], sc[te] = m.proba(F.iloc[te], S[te]), m.score(F.iloc[te], S[te])
        P[M.name] = (p, sc)
        print(f"[loio] {M.name} done", flush=True)
    pd.DataFrame({k: v[0] for k, v in P.items()}).assign(segment=ev["segment"], track_id=ev["track_id"], site=ev["group"]).to_csv(f"{out}/preds_loio.csv", index=False)
    rows = []
    for name, mask in subsets(ev, F).items():
        b = bench_eval.bootstrap(ev, P, mask & tested, args.n_boot, ref=RuleA.name)
        rows.append(b.assign(scope="pooled", subset=name))
    for site, _, te in folds:
        m = np.zeros(len(ev), bool)
        m[te] = True
        b = bench_eval.bootstrap(ev, P, m, args.n_boot, ref=RuleA.name)
        rows.append(b.assign(scope=site, subset="all"))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(f"{out}/loio.csv", index=False)
    json.dump(dict(test_sites=[s for s, _, _ in folds], n_tested=int(tested.sum()), n_tested_go=int(y[tested].sum()),
                   n_total=len(ev), per_site=ev.groupby("group")["label"].agg(n="size", go="sum").to_dict("index")),
              open(f"{out}/loio_counts.json", "w"), indent=1, default=int)
    q = df[(df.metric == "macro_f1") & (df.scope == "pooled")]
    print(q.round(3).to_string(), flush=True)


def _init(event_dir, exclude, min_site, threads):
    # spawn context: a forked worker deadlocks once torch has run in the parent (OpenMP), so reload the tables here
    import torch
    torch.set_num_threads(threads)
    os.nice(19)
    ev, F, S = load_tables(event_dir, exclude)
    _D.update(ev=ev, F=F, S=S, folds=loio_folds(ev, min_site))


def _draw(task):
    frac, rep = task
    ev, F, S, folds = _D["ev"], _D["F"], _D["S"], _D["folds"]
    y, cyc = ev["label"].values.astype(int), bench_eval.cycles(ev)
    models = [(LogRegDVA, {}), (LogRegDV, {}), (LSTMDVA, {"n_seeds": 1}), (TransformerDVA, {"n_seeds": 1})]
    P = {M.name: np.full(len(y), np.nan) for M, _ in models}
    for k, (site, tr, te) in enumerate(folds):
        if frac < 1:
            uc = np.unique(cyc[tr])
            rng = np.random.default_rng([k, int(frac * 100), rep])
            keep = rng.choice(uc, max(1, int(round(frac * len(uc)))), replace=False)
            tr = tr[np.isin(cyc[tr], keep)]
        if len(np.unique(y[tr])) < 2:
            continue
        for M, kw in models:
            P[M.name][te] = M(**kw).fit(F.iloc[tr], S[tr], y[tr]).proba(F.iloc[te], S[te])
    rows = []
    for name, p in P.items():
        ok = np.isfinite(p)
        rows.append(dict(frac=frac, rep=rep, model=name, macro_f1=bench_eval._macro_f1(y[ok], (p[ok] >= 0.5).astype(int)), n_scored=int(ok.sum())))
    print(f"[curve] frac={frac:.2f} rep={rep} done", flush=True)
    return rows


def curve(args, ev, F, S, out):
    fracs = [float(x) for x in args.fracs.split(",")]
    r0, r1 = args.rep_start, args.rep_start + args.curve_reps
    tasks = [(f, r) for f in fracs for r in (range(r0, r1) if f < 1 else [0])]
    threads = max(1, os.cpu_count() // args.workers)
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.event_dir, args.exclude, args.min_site, threads)) as pool:
        rows = sum(pool.map(_draw, tasks, chunksize=1), [])
    pd.DataFrame(rows).to_csv(f"{out}/curve_draws.csv" if r0 == 0 else f"{out}/curve_draws_rep{r0}-{r1 - 1}.csv", index=False)
    curve_merge(out)


def curve_merge(out):
    """curve.csv from every curve_draws*.csv in `out` (draw batches run separately, same seeds as model_compare.py)."""
    import glob
    raw = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{out}/curve_draws*.csv"))], ignore_index=True)
    raw = raw.drop_duplicates(["frac", "rep", "model"])
    lc = raw.groupby(["model", "frac"])["macro_f1"].agg(mean="mean", lo=lambda x: np.percentile(x, 2.5), hi=lambda x: np.percentile(x, 97.5), draws="size").reset_index()
    lc.to_csv(f"{out}/curve.csv", index=False)
    print(lc.round(3).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-dir", required=True)
    ap.add_argument("--fluid-event-dir", default="data/raw/fluid/results/benchmark_fi_leader")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default="external,loio")
    ap.add_argument("--exclude", default="", help="CSV with segment,track_id rows to drop")
    ap.add_argument("--min-site", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--curve-reps", type=int, default=5)
    ap.add_argument("--rep-start", type=int, default=0, help="first draw index (seeds [fold, frac, rep] as model_compare.py)")
    ap.add_argument("--fracs", default="0.25,0.5,0.75,1")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ev, F, S = load_tables(args.event_dir, args.exclude)
    print(f"leaders {len(ev)} go {int(ev.label.sum())} sites {ev.group.nunique()}", flush=True)
    json.dump(dict(n=len(ev), n_go=int(ev.label.sum()), subsets={k: int(v.sum()) for k, v in subsets(ev, F).items()},
                   t_r=ev["yellow_s"].round(1).value_counts().to_dict()), open(f"{args.out}/counts.json", "w"), indent=1, default=int)
    for task in args.tasks.split(","):
        {"external": external, "loio": loio, "curve": curve}[task](args, ev, F, S, args.out)


if __name__ == "__main__":
    main()
