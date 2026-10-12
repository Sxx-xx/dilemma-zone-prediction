"""Model-vs-model comparison and learning curve on one leader/follower split (leave-one-group-out, no tuning).

1. Paired cycle-bootstrap differences of every model vs LR(d, v, a) (same procedure as the rule-A comparison).
2. Learning curve: in every fold, keep a random fraction of the *training* yellow cycles (25/50/75/100 %, `--reps`
   draws each), train LR(d,v,a), LR(d,v), LSTM, Transformer, predict the untouched test fold; macro-F1 of the pooled
   out-of-fold predictions per draw -> mean and 2.5/97.5 percentiles over draws. 100 % is a single draw (no sampling).
   `--curve-only` skips step 1 (vs_logreg_dva.csv is left as it is).

  .venv/bin/python scripts/model_compare.py --dataset fluid --root data/raw/fluid/FI/derived_data --site FI \
      --approaches W_approach,E_approach --light-map W_approach=light_E_s,E_approach=light_E_s --split leader \
      --out results/fluid_fi/leader_follower/leader/model_compare
"""
import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dzp import bench_eval
from dzp.bench_models import LogRegDV, LogRegDVA, LSTMSeq, TransformerSeq, model_list
from run_benchmark import at, extract, through

CURVE_MODELS = [(LogRegDVA, {}), (LogRegDV, {}), (LSTMSeq, {"n_seeds": 3}), (TransformerSeq, {"n_seeds": 3})]
_D = {}


def _draw(task):
    """One (fraction, draw): pooled out-of-fold P(go) per curve model."""
    frac, rep = task
    ev, F, S = _D["ev"], _D["F"], _D["S"]
    y, cyc = ev["label"].values.astype(int), bench_eval.cycles(ev)
    P = {M.name: np.full(len(y), np.nan) for M, _ in CURVE_MODELS}
    for k, (tr, te) in enumerate(bench_eval.splits(ev, "group")):
        if frac < 1:
            uc = np.unique(cyc[tr])
            rng = np.random.default_rng([k, int(frac * 100), rep])
            keep = rng.choice(uc, max(1, int(round(frac * len(uc)))), replace=False)
            tr = tr[np.isin(cyc[tr], keep)]
        if len(np.unique(y[tr])) < 2:
            continue
        for M, kw in CURVE_MODELS:
            m = M(**kw).fit(F.iloc[tr], S[tr], y[tr])
            P[M.name][te] = m.proba(F.iloc[te], S[te])
    rows = []
    for name, p in P.items():
        ok = np.isfinite(p)
        rows.append(dict(frac=frac, rep=rep, model=name, macro_f1=bench_eval._macro_f1(y[ok], (p[ok] >= 0.5).astype(int)),
                         n_scored=int(ok.sum())))
    print(f"[curve] frac={frac:.2f} rep={rep} done", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    for a in ("--dataset", "--root", "--out"):
        ap.add_argument(a, required=True)
    for a in ("--site", "--exclude", "--approaches", "--light-map"):
        ap.add_argument(a, default="")
    ap.add_argument("--group", default="recording", choices=["recording", "intersection"])
    ap.add_argument("--split", default="leader", choices=["all", "leader", "follower"])
    ap.add_argument("--decel", type=float, default=3.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--fracs", default="0.25,0.5,0.75,1")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--curve-only", action="store_true", help="skip the model comparison, run the learning curve only")
    args = ap.parse_args()
    args.leads = [0.0]
    os.makedirs(args.out, exist_ok=True)

    ev_all, feat_all, seq_all = extract(args, 0.0)
    _, ev, feat, seqs = through(ev_all, feat_all, seq_all, args.split)
    F, S = at(feat, seqs, 0.0, np.arange(len(ev)))
    print(f"{args.split}: n={len(ev)} go={int(ev['label'].sum())}", flush=True)

    # ---- 1. every model vs LR(d, v, a)
    if not args.curve_only:
        P = bench_eval.oof(model_list(args.decel, True, args.n_seeds), F, S, ev, "group")
        b = bench_eval.bootstrap(ev, P, None, args.n_boot, ref=LogRegDVA.name)
        b.to_csv(f"{args.out}/vs_logreg_dva.csv", index=False)
        print(b[b.metric != "brier"].round(3).to_string(), flush=True)

    # ---- 2. learning curve
    _D.update(ev=ev, F=F, S=S)
    fracs = [float(x) for x in args.fracs.split(",")]
    tasks = [(f, r) for f in fracs for r in (range(args.reps) if f < 1 else [0])]
    with Pool(args.workers) as pool:
        rows = sum(pool.map(_draw, tasks), [])
    raw = pd.DataFrame(rows)
    raw.to_csv(f"{args.out}/learning_curve_draws.csv", index=False)
    lc = raw.groupby(["model", "frac"])["macro_f1"].agg(
        mean="mean", lo=lambda x: np.percentile(x, 2.5), hi=lambda x: np.percentile(x, 97.5), draws="size").reset_index()
    lc.to_csv(f"{args.out}/learning_curve.csv", index=False)
    print(lc.round(3).to_string(), flush=True)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, r in lc.groupby("model", sort=False):
        ax.errorbar(r["frac"] * 100, r["mean"], yerr=[r["mean"] - r["lo"], r["hi"] - r["mean"]], marker="o", capsize=3,
                    label=name)
    ax.set_xlabel("training yellow cycles kept per fold (%)")
    ax.set_ylabel("macro-F1 (leave-one-group-out)")
    ax.set_title(f"learning curve, {args.split} (n={len(ev)}); bars: 2.5-97.5 % over {args.reps} draws", fontsize=9)
    ax.set_xticks([f * 100 for f in fracs])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_learning_curve.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
