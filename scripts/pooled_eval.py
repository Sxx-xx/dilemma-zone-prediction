"""P. Pooled validation over V2X-Seq + FLUID + SinD + WOMD leader events (existing extraction / features / models).

Tasks (--tasks, comma list, in this order):
  sind-leads : SinD v2.0 Changchun selected leaders (scripts/sind_full.py, lead 0 only) re-built at leads 0..2 s with the
               same code path (sind.load_full_recording -> events.extract_events -> features.build); rows matched to
               the existing leader table on (recording, track_id, sid, yellow frame); lead-0 features checked equal.
  table      : unified leader table under --table (gitignored, run_benchmark event-dir format):
               events_through.csv (dataset, site_id, group = fold unit, cycle = bootstrap unit, event_id, label, ...),
               features_through.csv (lead_s 0, 0.5, 1, 1.5, 2), sequences_through.npz; P1 composition +
               harmonisation tables under --out.
  boundary   : stage 1 (WOMD excluded): pooled t_50 (d/v logistic, d/v <= 12 s) with cycle bootstrap CI, per dataset;
               LR tests with dataset dummies (intercept, intercept + slope); pooled LOIO of the calibrated rule A at
               onset (all / band, pooled and per dataset); t_50 with WOMD added (one line).
  models     : stage 2 (all four): (a) pooled LOIO over site_id (WOMD: its group 10-fold), (b) leave-one-dataset-out;
               per lead: rule A (recorded t_r), calibrated rule A (t_50(L) from the training folds), LR(d,v,a),
               band-only LR(d,v,a), MLP(d,v,a), LSTM(d,v,a); out-of-fold predictions under --table/preds_*.
  report     : P2 / P3 tables (macro-F1 all and band, AUROC, Brier; cycle-bootstrap CIs; pooled + per dataset), MLP vs
               calibrated rule A per lead with an event bootstrap, comparison with the per-dataset results (|diff| >=
               0.02 flagged), figure (band macro-F1 vs lead, pooled + per dataset).

Band = d/v at onset in [2.5, 5.5] s. At lead L an event is usable if its vehicle has not crossed the stop line (d > 0)
and its features exist (same rule as scripts/leadtime_dz.py).

  .venv/bin/python scripts/pooled_eval.py --tasks sind-leads,table,boundary,models,report --workers 6
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

LEADS = [0.0, 0.5, 1.0, 1.5, 2.0]
DZ_LO, DZ_HI, TT_MAX = 2.5, 5.5, 12.0
DATASETS = ["v2xseq", "fluid", "sind", "womd"]
SRC = {"v2xseq": "data/raw/v2xseq/results/leaders_v4_leads", "fluid": "data/raw/fluid/results/leadtime_fi_leader",
       "sind": "data/raw/pooled/sind_leads", "womd": "data/raw/womd/w1_leads/tables_all_gkf"}
SIND_ROOT, SIND_EV = "data/raw/sind/full/SinD_v2.0_CSV_Dataset", "data/raw/sind/results/full"
FEAT_COLS = ["ev", "lead_s", "d", "v", "a", "t_r", "tt", "ldr_id", "clamped"]
MODELS = ["ruleA", "ruleA_calibrated", "logreg_dva", "band_logreg_dva", "mlp_dva", "lstm_dva"]
_G = {}


# ------------------------------------------------------------------------------------------------ sind-leads
def _sind_one(rec_name):
    from dzp import events, features, sind
    from run_benchmark import movement
    city, rec = rec_name.split("/")
    r = sind.load_full_recording(SIND_ROOT, city, rec, sind.full_offsets(SIND_ROOT), {})
    aps, _ = events.approach_crossings(r)
    ev, _, _ = events.extract_events(r)
    ev["group"], ev["movement"] = r.name, movement(r, ev)
    f, s = features.build(r, ev, aps, LEADS, dt=sind.DT)
    return rec_name, ev, f, s


def sind_leads(args):
    old = pd.read_csv(f"{SIND_EV}/events_through.csv")
    fo = pd.read_csv(f"{SIND_EV}/features_through.csv")
    recs = sorted(old["group"].unique())
    with get_context("spawn").Pool(min(len(recs), args.workers)) as pool:
        res = pool.map(_sind_one, recs)
    E, F, S = [], [], {L: [] for L in LEADS}
    off = 0
    for rec_name, ev, f, s in res:
        o = old[old["group"] == rec_name]
        key = lambda df: list(zip(df["track_id"], df["sid"], df["f_yellow"].round().astype(int)))
        pos = {k: i for i, k in enumerate(key(ev))}
        idx = np.array([pos[k] for k in key(o)])
        f = f[f["ev"].isin(idx)].copy()
        f["ev"] = f["ev"].map({j: off + i for i, j in enumerate(idx)})
        E.append(o)
        F.append(f)
        for L in LEADS:
            S[L].append(s[L][idx])
        off += len(o)
    ev = pd.concat(E, ignore_index=True)
    feat = pd.concat(F, ignore_index=True).sort_values(["ev", "lead_s"]).reset_index(drop=True)
    # lead-0 features must reproduce the existing table
    a = feat[feat.lead_s == 0].set_index("ev")[["d", "v", "a", "t_r", "tt"]].values
    ref = []
    for rec_name in recs:
        o = old.index[old["group"] == rec_name]
        ref.append(fo[fo.lead_s == 0].set_index("ev").loc[o, ["d", "v", "a", "t_r", "tt"]].values)
    ref = np.concatenate(ref)
    dev = float(np.nanmax(np.abs(a - ref)))
    os.makedirs(SRC["sind"], exist_ok=True)
    ev.to_csv(f"{SRC['sind']}/events_through.csv", index=False)
    feat.to_csv(f"{SRC['sind']}/features_through.csv", index=False)
    np.savez_compressed(f"{SRC['sind']}/sequences_through.npz", **{f"lead{L:+.1f}": np.concatenate(S[L]).astype(np.float32) for L in LEADS})
    print(f"sind leads: {len(ev)} leaders from {len(recs)} recordings; max |lead-0 feature diff| vs existing table {dev:.2e}", flush=True)


# ------------------------------------------------------------------------------------------------ table
HARMON = [
    dict(dataset="v2xseq", sensor="roadside lidar/camera fusion (V2X-Seq TFD)", hz=10, resampled="no (native 10 Hz)",
         yellow="recorded per cycle (SPaT)", t_r="red onset - prediction time, from the segment's signal records",
         stop_line="map stop line shifted per approach by the calibrated offset table (results/v2xseq/stopline_shift.csv)",
         acceleration="per-frame derivative of speed (a_lon)", fold_unit="intersection (site)", bootstrap_unit="yellow cycle (segment:onset)"),
    dict(dataset="fluid", sensor="drone video (FLUID FI)", hz=10, resampled="no (10 FPS trajectories)",
         yellow="3.003 s fixed (video light state)", t_r="red onset - prediction time, from the light matrix",
         stop_line="stop line from the FI map", acceleration="per-frame derivative of speed (a_lon)",
         fold_unit="intersection (FI, one site)", bootstrap_unit="yellow cycle (video:onset)"),
    dict(dataset="sind", sensor="drone video (SinD v2.0, Changchun)", hz=9.99, resampled="no (100.1 ms step)",
         yellow="recorded per cycle (traffic-light CSV, offset-corrected)", t_r="red onset - prediction time, from the light CSV",
         stop_line="stop line from the lanelet map", acceleration="per-frame derivative of speed (a_lon)",
         fold_unit="intersection (cc, one site)", bootstrap_unit="yellow cycle (recording:onset)"),
    dict(dataset="womd", sensor="on-board perception (WOMD v1.3.1)", hz=10, resampled="no (native 10 Hz)",
         yellow="original dynamic map states (Michigan fill used for red onsets only)",
         t_r="red onset - prediction time; red onset from the original signals or Michigan fill (t_r_source)",
         stop_line="stop point of the controlled lane in the scenario map", acceleration="1 s moving average of speed derivative",
         fold_unit="group 10-fold over spatial regions (existing definition)", bootstrap_unit="region"),
]


def build_table(args):
    E, F, Q = [], [], {L: [] for L in LEADS}
    off = 0
    for ds in DATASETS:
        d = SRC[ds]
        ev = pd.read_csv(f"{d}/events_through.csv")
        feat = pd.read_csv(f"{d}/features_through.csv")
        z = np.load(f"{d}/sequences_through.npz")
        if "clamped" not in feat.columns:
            feat["clamped"] = 0
        f0 = feat[feat.lead_s == 0].set_index("ev").reindex(np.arange(len(ev)))
        ok = f0["t_r"].notna().values
        keep = np.where(ok)[0]
        from dzp import bench_eval
        cyc = bench_eval.cycles(ev)
        inter = ev["intersection"].astype(str).values
        site = inter if ds in ("fluid", "sind") else ev["group"].astype(str).values
        sub = ev[{"v2xseq": "segment", "fluid": "group", "sind": "group", "womd": "intersection"}[ds]].astype(str).values
        e = pd.DataFrame(dict(dataset=ds, site_id=[f"{ds}:{s}" for s in site], sub_site=[f"{ds}:{s}" for s in sub],
                              cycle=[f"{ds}:{c}" for c in cyc], event_id=np.arange(len(ev)), track_id=ev["track_id"].values,
                              label=ev["label"].values.astype(int), yellow_s=ev["yellow_s"].values))
        e = e.iloc[keep].reset_index(drop=True)
        e["group"] = e["site_id"]
        f = feat[feat["ev"].isin(keep)][FEAT_COLS].copy()
        f["ev"] = f["ev"].map({o: off + i for i, o in enumerate(keep)})
        E.append(e)
        F.append(f)
        for L in LEADS:
            Q[L].append(z[f"lead{L:+.1f}"][keep].astype(np.float32))
        off += len(keep)
        print(f"{ds}: {len(ev)} leaders, {len(ev) - len(keep)} dropped (no t_r at onset)", flush=True)
    ev = pd.concat(E, ignore_index=True)
    feat = pd.concat(F, ignore_index=True).sort_values(["ev", "lead_s"]).reset_index(drop=True)
    os.makedirs(args.table, exist_ok=True)
    ev.to_csv(f"{args.table}/events_through.csv", index=False)
    feat.to_csv(f"{args.table}/features_through.csv", index=False)
    np.savez_compressed(f"{args.table}/sequences_through.npz", **{f"lead{L:+.1f}": np.concatenate(Q[L]) for L in LEADS})
    # P1
    os.makedirs(args.out, exist_ok=True)
    D = load(args.table)
    rows = []
    for ds in DATASETS + ["pooled"]:
        m = np.ones(len(ev), bool) if ds == "pooled" else (ev["dataset"].values == ds)
        y, b = D["y"], D["band0"]
        r = dict(dataset=ds, events=int(m.sum()), stop=int((y[m] == 0).sum()), stop_share=float((y[m] == 0).mean()),
                 band=int((m & b).sum()), band_stop=int(((y == 0) & m & b).sum()), band_stop_share=float((y[m & b] == 0).mean()),
                 sites=ev.loc[m, "site_id"].nunique(), sub_sites=ev.loc[m, "sub_site"].nunique(), cycles=ev.loc[m, "cycle"].nunique(),
                 yellow_median=float(np.median(ev.loc[m, "yellow_s"])), yellow_min=float(ev.loc[m, "yellow_s"].min()), yellow_max=float(ev.loc[m, "yellow_s"].max()),
                 v0_median=float(np.median(D["F"][0.0]["v"].values[m])), d0_median=float(np.median(D["F"][0.0]["d"].values[m])))
        for L in LEADS:
            r[f"usable_{L:.1f}s"] = int((m & D["usable"][L]).sum())
            r[f"band_usable_{L:.1f}s"] = int((m & b & D["usable"][L]).sum())
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{args.out}/P1_composition.csv", index=False)
    pd.DataFrame(HARMON).to_csv(f"{args.out}/P1_harmonisation.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)


def load(table):
    from run_benchmark import at
    ev = pd.read_csv(f"{table}/events_through.csv")
    feat = pd.read_csv(f"{table}/features_through.csv")
    z = np.load(f"{table}/sequences_through.npz")
    seqs = {L: z[f"lead{L:+.1f}"] for L in LEADS}
    idx = np.arange(len(ev))
    F, S = {}, {}
    for L in LEADS:
        F[L], S[L] = at(feat, seqs, L, idx)
    y = ev["label"].values.astype(int)
    band0 = ((F[0.0]["tt"] >= DZ_LO) & (F[0.0]["tt"] <= DZ_HI)).values
    usable = {}
    for L in LEADS:
        d = F[L]["d"].values
        usable[L] = np.isfinite(d) & F[L]["v"].notna().values & (d > 0)
    return dict(ev=ev, F=F, S=S, y=y, band0=band0, usable=usable)


# ------------------------------------------------------------------------------------------------ helpers
def t50_fit(x, y):
    if len(np.unique(y)) < 2:
        return np.nan
    lr = LogisticRegression(max_iter=1000).fit(x[:, None], y)
    return -lr.intercept_[0] / lr.coef_[0, 0] if lr.coef_[0, 0] < 0 else np.nan


def _loglik(X, y):
    lr = LogisticRegression(C=np.inf, max_iter=5000).fit(X, y)
    p = np.clip(lr.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), lr


def cyc_draws(cyc, mask, n_boot, seed=0):
    idx = np.where(mask)[0]
    c = cyc[idx]
    uc, inv = np.unique(c, return_inverse=True)
    mem = [idx[inv == k] for k in range(len(uc))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([mem[k] for k in rng.integers(0, len(uc), len(uc))]) for _ in range(n_boot)]


def ev_draws(mask, n_boot, seed=0):
    idx = np.where(mask)[0]
    rng = np.random.default_rng(seed)
    return [idx[rng.integers(0, len(idx), len(idx))] for _ in range(n_boot)]


def metrics(y, p, s):
    from dzp import bench_eval
    return bench_eval.metrics(y, p, s)


# ------------------------------------------------------------------------------------------------ boundary
def boundary(args):
    D = load(args.table)
    ev, y, F0 = D["ev"], D["y"], D["F"][0.0]
    tt, tr = F0["tt"].values, F0["t_r"].values
    ds = ev["dataset"].values
    cyc = ev["cycle"].values
    three = np.isin(ds, ["v2xseq", "fluid", "sind"])
    keep = tt <= TT_MAX
    rows = []
    for name, m in [("pooled_3", three)] + [(d, ds == d) for d in ["v2xseq", "fluid", "sind"]] + [("pooled_4 (with WOMD)", np.ones(len(y), bool)), ("womd", ds == "womd")]:
        mm = m & keep
        idx = np.where(mm)[0]
        t = t50_fit(tt[idx], y[idx])
        bs = np.array([t50_fit(tt[ix], y[ix]) for ix in cyc_draws(cyc, mm, args.n_boot)])
        rows.append(dict(set=name, n=len(idx), n_go=int(y[idx].sum()), t50=t, ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5),
                         boot_undefined=int(np.isnan(bs).sum()), yellow_median=float(np.median(ev["yellow_s"].values[idx]))))
    bd = pd.DataFrame(rows)
    bd.to_csv(f"{args.out}/P_boundary_t50.csv", index=False)
    print(bd.round(3).to_string(), flush=True)
    # LR tests with dataset dummies (V2X-Seq reference)
    idx = np.where(three & keep)[0]
    x, yy, dd = tt[idx], y[idx], ds[idx]
    Dm = np.column_stack([(dd == "fluid"), (dd == "sind")]).astype(float)
    ll0, m0 = _loglik(x[:, None], yy)
    ll1, m1 = _loglik(np.column_stack([x, Dm]), yy)
    ll2, m2 = _loglik(np.column_stack([x, Dm, x[:, None] * Dm]), yy)
    b0, b = m1.intercept_[0], m1.coef_[0]
    t50_common = {"v2xseq": -b0 / b[0], "fluid": -(b0 + b[1]) / b[0], "sind": -(b0 + b[2]) / b[0]}
    tests = [dict(test="dataset intercepts vs none (common slope)", df=2, lr_stat=2 * (ll1 - ll0), p=chi2.sf(2 * (ll1 - ll0), 2)),
             dict(test="dataset slopes given intercepts", df=2, lr_stat=2 * (ll2 - ll1), p=chi2.sf(2 * (ll2 - ll1), 2)),
             dict(test="dataset intercepts + slopes vs none", df=4, lr_stat=2 * (ll2 - ll0), p=chi2.sf(2 * (ll2 - ll0), 4))]
    lt = pd.DataFrame(tests)
    lt["n"] = len(idx)
    for k, v in t50_common.items():
        lt[f"t50_common_slope_{k}"] = v
    lt.to_csv(f"{args.out}/P_boundary_lrtest.csv", index=False)
    print(lt.round(4).to_string(), flush=True)
    # pooled LOIO of the calibrated rule A at onset (three datasets)
    sites = ev["site_id"].values
    P = np.full(len(y), np.nan)
    S = np.full(len(y), np.nan)
    fp = []
    for s in np.unique(sites[three]):
        te = np.where(three & (sites == s))[0]
        trn = np.where(three & (sites != s) & keep)[0]
        t = t50_fit(tt[trn], y[trn])
        P[te], S[te] = (tt[te] < t).astype(float), t - tt[te]
        fp.append(dict(test_site=s, n_test=len(te), t50_train=t))
    pd.DataFrame(fp).to_csv(f"{args.out}/P_boundary_loio_folds.csv", index=False)
    rows = []
    for sub, sm in [("all", np.ones(len(y), bool)), ("band", D["band0"])]:
        for name, m in [("pooled_3", three)] + [(d, ds == d) for d in ["v2xseq", "fluid", "sind"]]:
            mm = m & sm
            pt = metrics(y[mm], P[mm], S[mm])
            bs = pd.DataFrame([metrics(y[ix], P[ix], S[ix]) for ix in cyc_draws(cyc, mm, args.n_boot)])
            rows.append(dict(subset=sub, set=name, n=int(mm.sum()), n_go=int(y[mm].sum()), macro_f1=pt["macro_f1"],
                             ci_lo=np.nanpercentile(bs.macro_f1, 2.5), ci_hi=np.nanpercentile(bs.macro_f1, 97.5)))
    lo = pd.DataFrame(rows)
    lo.to_csv(f"{args.out}/P_boundary_loio_calibrated_ruleA.csv", index=False)
    fpd = pd.DataFrame(fp)
    print(lo.round(3).to_string(), flush=True)
    print(f"fold t50: {fpd.t50_train.min():.3f}-{fpd.t50_train.max():.3f}", flush=True)


# ------------------------------------------------------------------------------------------------ models
def _init(table, threads):
    import torch
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    _G.update(load(table))


def _fold(task):
    """task = (scheme, fold name, lead, test mask kind); returns preds for the test events."""
    from leadtime_dz import _BandLR
    from dzp.bench_models import LogRegDVA, MLPDVA
    from seq_dva import LSTMDVA
    scheme, name, L, n_seeds = task
    ev, y, band0, usable = _G["ev"], _G["y"], _G["band0"], _G["usable"][L]
    F, S = _G["F"][L], _G["S"][L]
    col = "site_id" if scheme == "loio" else "dataset"
    g = ev[col].values
    tr = np.where((g != name) & usable)[0]
    te = np.where((g == name) & usable)[0]
    out = dict(ev=te)
    if not len(te) or len(np.unique(y[tr])) < 2:
        return scheme, name, L, None
    tt, trr = F["tt"].values, F["t_r"].values
    k = tr[tt[tr] <= TT_MAX]
    t = t50_fit(tt[k], y[k])
    out["ruleA"], out["ruleA_score"] = (tt[te] < trr[te]).astype(float), trr[te] - tt[te]
    out["ruleA_calibrated"], out["ruleA_calibrated_score"] = (tt[te] < t).astype(float), t - tt[te]
    out["t50_train"] = np.full(len(te), t)
    out["logreg_dva"] = LogRegDVA().fit(F.iloc[tr], S[tr], y[tr]).proba(F.iloc[te], S[te])
    m = _BandLR(band_mask=band0[tr]).fit(F.iloc[tr], S[tr], y[tr])
    out["band_logreg_dva"] = m.proba(F.iloc[te], S[te]) if m is not None else np.full(len(te), np.nan)
    out["mlp_dva"] = MLPDVA().fit(F.iloc[tr], S[tr], y[tr]).proba(F.iloc[te], S[te])
    out["lstm_dva"] = LSTMDVA(n_seeds=n_seeds).fit(F.iloc[tr], S[tr], y[tr]).proba(F.iloc[te], S[te])
    return scheme, name, L, pd.DataFrame(out)


def models(args):
    D = load(args.table)
    ev = D["ev"]
    tasks = []
    for L in LEADS:
        for s in sorted(ev["site_id"].unique()):
            tasks.append(("loio", s, L, args.n_seeds))
        for d in DATASETS:
            tasks.append(("lodo", d, L, args.n_seeds))
    tasks.sort(key=lambda t: -(ev["site_id"] == t[1]).sum() if t[0] == "loio" else -1e9)  # big folds first
    done = {}
    t0 = time.time()
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.table, args.threads)) as pool:
        for i, (scheme, name, L, df) in enumerate(pool.imap_unordered(_fold, tasks)):
            if df is not None:
                done.setdefault((scheme, L), []).append(df.assign(fold=name))
            if i % 10 == 0 or i == len(tasks) - 1:
                print(f"{i + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    for (scheme, L), parts in done.items():
        pd.concat(parts, ignore_index=True).sort_values("ev").to_csv(f"{args.table}/preds_{scheme}_lead{L:+.1f}.csv", index=False)


# ------------------------------------------------------------------------------------------------ report
def _preds(table, scheme, L, n):
    p = pd.read_csv(f"{table}/preds_{scheme}_lead{L:+.1f}.csv")
    P = {}
    for m in MODELS:
        a = np.full(n, np.nan)
        a[p.ev.values] = p[m].values
        s = np.full(n, np.nan)
        s[p.ev.values] = p[f"{m}_score"].values if f"{m}_score" in p.columns else p[m].values
        P[m] = (a, s)
    fold = np.full(n, None, object)
    fold[p.ev.values] = p.fold.values
    t50 = np.full(n, np.nan)
    t50[p.ev.values] = p.t50_train.values
    return P, fold, t50


def report(args):
    D = load(args.table)
    ev, y, band0 = D["ev"], D["y"], D["band0"]
    n = len(y)
    ds = ev["dataset"].values
    cyc = ev["cycle"].values
    sets = [("pooled", np.ones(n, bool))] + [(d, ds == d) for d in DATASETS]
    tabs = {}
    sig = []
    for scheme in ("loio", "lodo"):
        rows = []
        for L in LEADS:
            P, fold, t50 = _preds(args.table, scheme, L, n)
            tested = np.isfinite(P["mlp_dva"][0])
            for sub, sm in (("all", np.ones(n, bool)), ("band", band0)):
                for name, m in sets:
                    if scheme == "lodo" and name == "pooled":
                        continue
                    mm = m & sm & tested
                    if mm.sum() < 5:
                        continue
                    draws = cyc_draws(cyc, mm, args.n_boot)
                    for mod in MODELS:
                        if mod == "band_logreg_dva" and sub == "all":
                            continue
                        p, s = P[mod]
                        mk = mm & np.isfinite(p)
                        pt = metrics(y[mk], p[mk], s[mk])
                        bs = pd.DataFrame([metrics(y[ix], p[ix], s[ix]) for ix in draws])
                        r = dict(scheme=scheme, lead_s=L, subset=sub, set=name, model=mod, n=int(mk.sum()), n_go=int(y[mk].sum()))
                        for k in ("macro_f1", "auroc", "brier"):
                            r[k] = pt[k]
                            r[f"{k}_lo"], r[f"{k}_hi"] = np.nanpercentile(bs[k], [2.5, 97.5]) if len(bs) else (np.nan, np.nan)
                        rows.append(r)
                    # MLP vs calibrated rule A, event bootstrap
                    pm, pc = P["mlp_dva"][0], P["ruleA_calibrated"][0]
                    from dzp.bench_eval import _macro_f1
                    f = lambda ix: _macro_f1(y[ix], (pm[ix] >= 0.5).astype(int)) - _macro_f1(y[ix], (pc[ix] >= 0.5).astype(int))
                    idx = np.where(mm)[0]
                    bs = np.array([f(ix) for ix in ev_draws(mm, args.n_boot)])
                    sig.append(dict(scheme=scheme, lead_s=L, subset=sub, set=name, n=len(idx), diff_mlp_minus_calrule=f(idx),
                                    ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
            if scheme == "loio":
                tf = pd.DataFrame(dict(fold=fold, t50=t50)).dropna().drop_duplicates("fold")
                tabs.setdefault("t50", []).append(dict(lead_s=L, t50_min=tf.t50.min(), t50_max=tf.t50.max(), t50_median=tf.t50.median(), folds=len(tf)))
        tabs[scheme] = pd.DataFrame(rows)
        print(f"{scheme} done", flush=True)
    tabs["loio"].to_csv(f"{args.out}/P2_pooled_loio.csv", index=False)
    tabs["lodo"].to_csv(f"{args.out}/P3_leave_one_dataset_out.csv", index=False)
    pd.DataFrame(sig).to_csv(f"{args.out}/P_mlp_vs_calibrated_ruleA.csv", index=False)
    pd.DataFrame(tabs["t50"]).to_csv(f"{args.out}/P2_fold_t50_by_lead.csv", index=False)
    compare(args, tabs["loio"])
    figure(args, tabs["loio"])


def _ref_rows():
    """Per-dataset results from the earlier stages: (dataset, subset, lead, model, macro_f1, n, source)."""
    R = []
    map_l = {"band_logreg_dva": "band_logreg_dva", "mlp_dva": "mlp_dva", "lstm_dva": "lstm_dva", "ruleA_calibrated": "ruleA_calibrated"}
    for ds, path in (("v2xseq", "results/v2xseq/leadtime_dz/leadtime.csv"), ("womd", "results/womd/leadtime_dz/leadtime.csv"),
                     ("fluid", "results/fluid_fi/leadtime_dz/leadtime.csv")):
        q = pd.read_csv(path)
        q = q[(q.metric == "macro_f1")].drop_duplicates(["model", "lead_s"])
        for r in q.itertuples():
            if r.model in map_l:
                R.append(dict(dataset=ds, subset="band", lead_s=r.lead_s, model=map_l[r.model], ref_macro_f1=r.value, ref_n=r.n, source=path))
    map0 = {"ruleA_d/v<t_r": "ruleA", "logreg_dva": "logreg_dva", "mlp_dva": "mlp_dva", "lstm_dva": "lstm_dva", "ruleA_calibrated": "ruleA_calibrated"}
    for ds, path in (("v2xseq", "results/v2xseq/eval/v4/calibrated_rule/metrics.csv"), ("womd", "results/womd/w1/final/gkf/calibrated_rule/metrics.csv")):
        q = pd.read_csv(path)
        q = q[(q.metric == "macro_f1") & q.subset.isin(["all", "dz"])].drop_duplicates(["model", "subset"])
        for r in q.itertuples():
            if r.model in map0:
                R.append(dict(dataset=ds, subset="all" if r.subset == "all" else "band", lead_s=0.0, model=map0[r.model], ref_macro_f1=r.value, ref_n=r.n, source=path))
    q = pd.read_csv("results/fluid_fi/leader_follower/leader/metrics.csv")
    q = q[(q.metric == "macro_f1") & (q.scheme == "group") & (q.subset == "all") & (q.offset_s == 0) & (q.lead_s == 0)].drop_duplicates("model")
    for r in q.itertuples():
        if r.model in map0:
            R.append(dict(dataset="fluid", subset="all", lead_s=0.0, model=map0[r.model], ref_macro_f1=r.value, ref_n=r.n, source="results/fluid_fi/leader_follower/leader/metrics.csv"))
    q = pd.read_csv("results/sind/full/loio_metrics.csv")
    q = q[(q.metric == "macro_f1") & q.subset.isin(["all", "dz"])].drop_duplicates(["model", "subset"])
    for r in q.itertuples():
        m = {"ruleA_d/v<t_r": "ruleA", "ruleA_calibrated": "ruleA_calibrated", "logreg_dva": "logreg_dva", "mlp_dva": "mlp_dva"}.get(r.model)
        if m:
            R.append(dict(dataset="sind", subset="all" if r.subset == "all" else "band", lead_s=0.0, model=m, ref_macro_f1=r.value, ref_n=r.n,
                          source="results/sind/full/loio_metrics.csv"))
    return pd.DataFrame(R)


def compare(args, lo):
    ref = _ref_rows()
    cur = lo[lo.set.isin(DATASETS)].rename(columns={"set": "dataset"})
    c = ref.merge(cur[["dataset", "subset", "lead_s", "model", "macro_f1", "n"]], on=["dataset", "subset", "lead_s", "model"], how="left")
    c["diff"] = c["macro_f1"] - c["ref_macro_f1"]
    c["flag_ge_0.02"] = c["diff"].abs() >= 0.02
    c.to_csv(f"{args.out}/P_compare_with_per_dataset.csv", index=False)
    print(c.round(3).to_string(), flush=True)


def figure(args, lo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    q = lo[(lo.subset == "band")]
    mods = [("mlp_dva", "MLP (d, v, a)", "o", "-"), ("band_logreg_dva", "Band-only LR (d, v, a)", "s", "--"),
            ("lstm_dva", "LSTM (d, v, a)", "^", "-."), ("ruleA_calibrated", "Calibrated rule A", "D", ":")]
    lw = {"v2xseq": ("V2X-Seq", "0.2", "o", "-"), "fluid": ("FLUID", "0.45", "s", "--"), "sind": ("SinD", "0.45", "^", ":"), "womd": ("WOMD", "0.65", "x", "-.")}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
    for ax, (m, lab, mk, ls) in zip(axes, mods):
        for d, (dl, col, dmk, dls) in lw.items():
            r = q[(q.model == m) & (q.set == d)].sort_values("lead_s")
            ax.plot(r.lead_s, r.macro_f1, color=col, lw=0.8, ls=dls, marker=dmk, ms=4, mfc="white", label=f"{dl} (n = {int(r.n.iloc[0])})")
        r = q[(q.model == m) & (q.set == "pooled")].sort_values("lead_s")
        ax.fill_between(r.lead_s, r.macro_f1_lo, r.macro_f1_hi, color="0.85", lw=0)
        ax.plot(r.lead_s, r.macro_f1, color="black", lw=2.2, ls=ls, marker=mk, ms=6, label="Pooled, 95% CI")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("Prediction time after yellow onset (s)")
        ax.set_xticks(LEADS)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("Macro-F1, dilemma-zone band")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs("results/figures/data", exist_ok=True)
    fig.savefig(f"{args.out}/fig_p_leadtime_band.png", dpi=300)
    fig.savefig("results/figures/fig_p_pooled_leadtime_band.png", dpi=300)
    q.to_csv("results/figures/data/fig_p_pooled_leadtime_band.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="table,boundary,models,report")
    ap.add_argument("--table", default="data/raw/pooled/table")
    ap.add_argument("--out", default="results/pooled")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for t in args.tasks.split(","):
        print(f"== {t}", flush=True)
        {"sind-leads": sind_leads, "table": build_table, "boundary": boundary, "models": models, "report": report}[t](args)


if __name__ == "__main__":
    main()
