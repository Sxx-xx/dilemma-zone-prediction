"""N. Staged (wait-or-commit) decision simulation in the dilemma zone.

Uses the L analysis' out-of-fold P(go) per lead (scripts/leadtime_dz.py -> preds_lead*.csv) for MLP(d,v,a) and
band-only LR(d,v,a) on the band fixed at onset (d/v in [2.5, 5.5] s). Rule with threshold theta: an event is
committed at the first lead in 0, 0.5, 1, 1.5, 2 s where P <= theta (stop) or P >= 1 - theta (go); events still
open at 2.0 s are committed there at P >= 0.5. An event whose vehicle has crossed the stop line at a lead (d <= 0)
is committed as go at that lead (observable); an event whose track has ended keeps its last available P.
Baselines: commit all at 0 s / 1.0 s / 2.0 s with P >= 0.5 (same crossing / track-end handling).

Reported per theta and baseline: macro-F1 with a cluster bootstrap CI over ev['cycle'], risky error rate (actual
stops committed as go / actual stops) and conservative error rate (actual go committed as stop / actual go), both
with CIs, mean / median commit time, cumulative share committed by each lead.

Theta chosen without test data: for every outer fold, inner out-of-fold P(go) on the training folds' band events
(leave-one-group-out over the training groups, same models retrained per inner fold and lead) -> largest theta in the
grid whose risky error on the training folds is <= 10 % (none -> commit all at 2.0 s) -> applied to the test fold.

  .venv/bin/python scripts/staged_dz.py --event-dir data/raw/v2xseq/results/leaders_v4_leads \\
      --preds results/v2xseq/leadtime_dz --out results/v2xseq/staged --workers 8
"""
import argparse
import os
import sys
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import bench_eval

LEADS = [0.0, 0.5, 1.0, 1.5, 2.0]
THETAS = [0.05, 0.1, 0.15, 0.2, 0.3]
MODELS = ["mlp_dva", "band_logreg_dva"]
DZ_LO, DZ_HI, RISK_MAX = 2.5, 5.5, 0.10
_G = {}


def load(event_dir):
    from run_benchmark import at
    ev = pd.read_csv(f"{event_dir}/events_through.csv")
    feat = pd.read_csv(f"{event_dir}/features_through.csv")
    z = np.load(f"{event_dir}/sequences_through.npz")
    seqs = {L: z[f"lead{L:+.1f}"] for L in LEADS}
    idx = np.arange(len(ev))
    FS = {L: at(feat, seqs, L, idx) for L in LEADS}
    F0 = FS[0.0][0]
    ok0 = F0["t_r"].notna().values
    band0 = ((F0["tt"] >= DZ_LO) & (F0["tt"] <= DZ_HI)).values & ok0
    crossed, usable = {}, {}
    for L in LEADS:
        F = FS[L][0]
        avail = ok0 & np.isfinite(F["d"].values) & F["v"].notna().values
        crossed[L] = avail & (F["d"].values <= 0)
        usable[L] = avail & ~crossed[L]
    return ev, FS, band0, crossed, usable


def decide(P, X, theta):
    """P: (n, leads) P(go), NaN where unusable; X: (n, leads) crossed. theta=None -> no early commit.
    Returns commit decision (0/1) and commit time (s)."""
    n, leads = len(P), LEADS[: P.shape[1]]
    yhat, tc = np.full(n, -1), np.full(n, np.nan)
    last = np.full(n, np.nan)
    for j, L in enumerate(leads):
        open_ = yhat < 0
        last = np.where(np.isfinite(P[:, j]), P[:, j], last)
        cx = open_ & X[:, j]
        yhat[cx], tc[cx] = 1, L
        open_ &= ~cx
        if theta is not None:
            p = P[:, j]
            hit = open_ & np.isfinite(p) & ((p <= theta) | (p >= 1 - theta))
            yhat[hit], tc[hit] = (p[hit] >= 0.5).astype(int), L
    open_ = yhat < 0
    yhat[open_], tc[open_] = (np.nan_to_num(last[open_], nan=0.5) >= 0.5).astype(int), leads[-1]
    return yhat, tc


def at_lead(P, X, j):
    """Commit everything at lead index j with P >= 0.5 (crossed earlier -> go at that time)."""
    return decide(P[:, : j + 1], X[:, : j + 1], None)


def stats(y, yhat, tc):
    st = y == 0
    return dict(macro_f1=bench_eval._macro_f1(y, yhat), risky=float(np.mean(yhat[st] == 1)) if st.any() else np.nan,
                conservative=float(np.mean(yhat[~st] == 0)) if (~st).any() else np.nan, t_mean=float(np.mean(tc)), t_median=float(np.median(tc)))


# --------------------------------------------------------------------------------------------- inner CV
def _init(event_dir, threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    _G.update(zip(("ev", "FS", "band0", "crossed", "usable"), load(event_dir)))


def _inner(task):
    from leadtime_dz import _BandLR
    from dzp.bench_models import MLPDVA
    s, L = task
    ev, band0, usable, y = _G["ev"], _G["band0"], _G["usable"][L], _G["ev"]["label"].values.astype(int)
    F, S = _G["FS"][L]
    g = ev["group"].values
    tr_all = np.where(g != s)[0]
    out = []
    for s2 in np.unique(g[tr_all]):
        tr = tr_all[(g[tr_all] != s2) & usable[tr_all]]
        te = tr_all[(g[tr_all] == s2) & usable[tr_all] & band0[tr_all]]
        if not len(te) or len(np.unique(y[tr])) < 2:
            continue
        m = _BandLR(band_mask=band0[tr]).fit(F.iloc[tr], S[tr], y[tr])
        pl = m.proba(F.iloc[te], S[te]) if m is not None else np.full(len(te), np.nan)
        pm = MLPDVA().fit(F.iloc[tr], S[tr], y[tr]).proba(F.iloc[te], S[te])
        out.append(pd.DataFrame(dict(outer=s, lead_s=L, ev=te, band_logreg_dva=pl, mlp_dva=pm)))
    return pd.concat(out) if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-dir", required=True)
    ap.add_argument("--preds", required=True, help="leadtime_dz output dir with preds_lead*.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ev, FS, band0, crossed, usable = load(args.event_dir)
    y = ev["label"].values.astype(int)
    g = ev["group"].values
    P = {m: np.column_stack([pd.read_csv(f"{args.preds}/preds_lead{L:+.1f}.csv")[m].values for L in LEADS]) for m in MODELS}
    X = np.column_stack([crossed[L] for L in LEADS])
    ib = np.where(band0)[0]
    # test-fold coverage: every band event must have an onset prediction
    assert all(np.isfinite(P[m][ib, 0]).all() for m in MODELS)
    print(f"band {len(ib)} (go {int(y[ib].sum())}), crossed by 2 s {int(X[ib].any(1).sum())}, "
          f"track ended by 2 s {int((~usable[2.0][ib] & ~X[ib, -1]).sum())}", flush=True)

    # ---- inner CV for the fold-wise theta choice
    cache = f"{args.out}/inner_preds.csv.gz"
    if os.path.exists(cache):
        inner = pd.read_csv(cache)
    else:
        outer = sorted(np.unique(g[ib]))
        tasks = [(s, L) for s in outer for L in LEADS]
        with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.event_dir, args.threads)) as pool:
            res = []
            for i, r in enumerate(pool.imap_unordered(_inner, tasks)):
                res.append(r)
                if i % 10 == 0:
                    print(f"inner {i + 1}/{len(tasks)}", flush=True)
        inner = pd.concat([r for r in res if r is not None], ignore_index=True)
        inner.to_csv(cache, index=False)

    sel_rows = []
    chosen = {m: np.full(len(y), np.nan) for m in MODELS}  # theta applied to each test event (0 = all at 2.0 s)
    for s in sorted(np.unique(g[ib])):
        te = ib[g[ib] == s]
        tr = ib[g[ib] != s]
        sub = inner[inner.outer == s]
        for m in MODELS:
            Pi = np.full((len(y), len(LEADS)), np.nan)
            for j, L in enumerate(LEADS):
                q = sub[sub.lead_s == L]
                Pi[q.ev.values, j] = q[m].values
            ok = tr[np.isfinite(Pi[tr, 0])]
            best, risks = 0.0, {}
            for th in THETAS:
                yh, _ = decide(Pi[ok], X[ok], th)
                risks[th] = float(np.mean(yh[y[ok] == 0] == 1)) if (y[ok] == 0).any() else np.nan
                if risks[th] <= RISK_MAX:
                    best = th
            chosen[m][te] = best
            sel_rows.append(dict(fold=s, model=m, n_train_band=len(ok), theta=best, **{f"risk_train_{th}": r for th, r in risks.items()}))
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(f"{args.out}/theta_selection.csv", index=False)

    # ---- policies on the outer out-of-fold predictions
    pol = {}
    for m in MODELS:
        Pm, Xm = P[m][ib], X[ib]
        for th in THETAS:
            pol[(m, f"theta={th}")] = decide(Pm, Xm, th)
        for j, L in enumerate(LEADS):
            if L in (0.0, 1.0, 2.0):
                pol[(m, f"all@{L:.1f}s")] = at_lead(Pm, Xm, j)
        yh, tc = np.zeros(len(ib), int), np.zeros(len(ib))
        for th in np.unique(chosen[m][ib]):
            k = chosen[m][ib] == th
            a, b = decide(Pm[k], Xm[k], th if th > 0 else None)
            yh[k], tc[k] = a, b
        pol[(m, "theta=train-selected")] = (yh, tc)

    cyc = bench_eval.cycles(ev)[ib]
    uc = np.unique(cyc)
    mem = {c: np.where(cyc == c)[0] for c in uc}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([mem[c] for c in rng.choice(uc, len(uc))]) for _ in range(args.n_boot)]
    yb = y[ib]
    rows, cum, boot = [], [], {}
    for (m, p), (yh, tc) in pol.items():
        boot[(m, p)] = pd.DataFrame([stats(yb[d], yh[d], tc[d]) for d in draws])
    for (m, p), (yh, tc) in pol.items():
        pt = stats(yb, yh, tc)
        bs = boot[(m, p)]
        r = dict(model=m, policy=p, n=len(ib), n_go=int(yb.sum()))
        for k in ("macro_f1", "risky", "conservative", "t_mean"):
            lo, hi = np.nanpercentile(bs[k], [2.5, 97.5])
            r.update({k: pt[k], f"{k}_lo": lo, f"{k}_hi": hi})
        r["t_median"] = pt["t_median"]
        ref = boot[(m, "all@2.0s")]
        for k in ("macro_f1", "risky"):  # paired difference to committing all at 2.0 s
            r[f"d_{k}_vs_2s"] = pt[k] - stats(yb, *pol[(m, "all@2.0s")])[k]
            r[f"d_{k}_vs_2s_lo"], r[f"d_{k}_vs_2s_hi"] = np.nanpercentile(bs[k] - ref[k], [2.5, 97.5])
        rows.append(r)
        cum.append(dict(model=m, policy=p, **{f"committed_by_{L:.1f}s": float(np.mean(tc <= L)) for L in LEADS}))
    res = pd.DataFrame(rows).merge(pd.DataFrame(cum), on=["model", "policy"])
    res.to_csv(f"{args.out}/staged.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(res[["model", "policy", "macro_f1", "macro_f1_lo", "macro_f1_hi", "d_macro_f1_vs_2s", "d_macro_f1_vs_2s_lo", "d_macro_f1_vs_2s_hi",
                   "risky", "risky_lo", "risky_hi", "conservative", "t_mean", "t_median"]
                  + [c for c in res.columns if c.startswith("committed_by")]].round(3).to_string(index=False))
        print(sel.groupby("model").theta.value_counts().to_string())


if __name__ == "__main__":
    main()
