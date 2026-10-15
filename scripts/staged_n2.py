"""N2. Staged decision, revised theta choice + matched-wait comparison + fixed theta = 0.05 (writes new files only).

Same staged rule as scripts/staged_dz.py (decide / at_lead). Per dataset and model (MLP(d,v,a), band-only LR(d,v,a)):
  1. theta per outer fold = largest theta in {0.05, 0.1, 0.15, 0.2, 0.3} whose training-fold risky error <= 10 % AND
     conservative error <= 20 % (inner out-of-fold P on the training folds' band events); none -> 0.05. Reported per
     fold (chosen theta, training errors, test macro-F1 / delta vs commit-all-at-2.0 s / mean commit time / errors)
     and overall (cycle bootstrap CIs, as in N).
  2. matched wait: staged theta = 0.1 vs committing all at the lead closest to its mean commit time; macro-F1
     difference with an event bootstrap (1,000).
  3. pre-fixed theta = 0.05, one row.
Outer P: --preds-kind leadtime -> <preds>/preds_lead*.csv (scripts/leadtime_dz.py, row = event);
         --preds-kind pooled   -> <preds>/preds_loio_lead*.csv (scripts/pooled_eval.py, column ev).
Inner P: <out>/inner_preds.csv.gz if present (V2X-Seq / WOMD: the N run's leave-one-group-out file is passed with
         --inner), else computed with GroupKFold(--inner-folds) over the training groups and cached.

  .venv/bin/python scripts/staged_n2.py --event-dir data/raw/v2xseq/results/leaders_v4_leads --preds results/v2xseq/leadtime_dz \\
      --inner results/v2xseq/staged/inner_preds.csv.gz --out results/v2xseq/staged
"""
import argparse
import os
import sys
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

import staged_dz
from staged_dz import LEADS, MODELS, THETAS, at_lead, decide, stats
from dzp import bench_eval

RISK_MAX, CONS_MAX, FALLBACK = 0.10, 0.20, 0.05
_G = {}


def _init(event_dir, threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    _G.update(zip(("ev", "FS", "band0", "crossed", "usable"), staged_dz.load(event_dir)))


def _inner_gkf(task):
    from sklearn.model_selection import GroupKFold
    from leadtime_dz import _BandLR
    from dzp.bench_models import MLPDVA
    s, L, k = task
    ev, band0, usable, y = _G["ev"], _G["band0"], _G["usable"][L], _G["ev"]["label"].values.astype(int)
    F, S = _G["FS"][L]
    g = ev["group"].values
    tr_all = np.where((g != s) & usable)[0]
    out = []
    for a, b in GroupKFold(n_splits=min(k, len(np.unique(g[tr_all])))).split(tr_all, groups=g[tr_all]):
        tr, te = tr_all[a], tr_all[b]
        te = te[band0[te]]
        if not len(te) or len(np.unique(y[tr])) < 2:
            continue
        m = _BandLR(band_mask=band0[tr]).fit(F.iloc[tr], S[tr], y[tr])
        pl = m.proba(F.iloc[te], S[te]) if m is not None else np.full(len(te), np.nan)
        pm = MLPDVA().fit(F.iloc[tr], S[tr], y[tr]).proba(F.iloc[te], S[te])
        out.append(pd.DataFrame(dict(outer=s, lead_s=L, ev=te, band_logreg_dva=pl, mlp_dva=pm)))
    return pd.concat(out) if out else None


def errors(y, yh):
    st = y == 0
    return (float(np.mean(yh[st] == 1)) if st.any() else np.nan, float(np.mean(yh[~st] == 0)) if (~st).any() else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-dir", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--preds-kind", default="leadtime", choices=["leadtime", "pooled"])
    ap.add_argument("--inner", default="", help="existing inner prediction file (else computed)")
    ap.add_argument("--inner-folds", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ev, FS, band0, crossed, usable = staged_dz.load(args.event_dir)
    y = ev["label"].values.astype(int)
    g = ev["group"].values
    n = len(y)
    P = {}
    for m in MODELS:
        cols = []
        for L in LEADS:
            if args.preds_kind == "leadtime":
                cols.append(pd.read_csv(f"{args.preds}/preds_lead{L:+.1f}.csv")[m].values)
            else:
                q = pd.read_csv(f"{args.preds}/preds_loio_lead{L:+.1f}.csv")
                a = np.full(n, np.nan)
                a[q.ev.values] = q[m].values
                cols.append(a)
        P[m] = np.column_stack(cols)
    P = {m: np.where(band0[:, None], v, np.nan) for m, v in P.items()}
    X = np.column_stack([crossed[L] for L in LEADS])
    ib = np.where(band0 & np.isfinite(P["mlp_dva"][:, 0]))[0]
    print(f"band {len(ib)} (go {int(y[ib].sum())})", flush=True)

    cache = args.inner or f"{args.out}/inner_preds.csv.gz"
    if os.path.exists(cache):
        inner = pd.read_csv(cache)
    else:
        tasks = [(s, L, args.inner_folds) for s in sorted(np.unique(g[ib])) for L in LEADS]
        with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.event_dir, 1)) as pool:
            res = []
            for i, r in enumerate(pool.imap_unordered(_inner_gkf, tasks)):
                res.append(r)
                if i % 20 == 0:
                    print(f"inner {i + 1}/{len(tasks)}", flush=True)
        inner = pd.concat([r for r in res if r is not None], ignore_index=True)
        inner.to_csv(cache, index=False)

    # ---- 1. fold-wise theta (risky <= 10 % and conservative <= 20 %, else 0.05)
    chosen = {m: np.full(n, np.nan) for m in MODELS}
    sel = []
    for s in sorted(np.unique(g[ib])):
        te, tr = ib[g[ib] == s], ib[g[ib] != s]
        sub = inner[inner.outer == s]
        for m in MODELS:
            Pi = np.full((n, len(LEADS)), np.nan)
            for j, L in enumerate(LEADS):
                q = sub[sub.lead_s == L]
                Pi[q.ev.values, j] = q[m].values
            ok = tr[np.isfinite(Pi[tr, 0])]
            best, row = None, dict(fold=s, model=m, n_train_band=len(ok))
            for th in THETAS:
                yh, _ = decide(Pi[ok], X[ok], th)
                r, c = errors(y[ok], yh)
                row[f"risky_train_{th}"], row[f"cons_train_{th}"] = r, c
                if r <= RISK_MAX and c <= CONS_MAX:
                    best = th
            th = best if best is not None else FALLBACK
            chosen[m][te] = th
            yh, tc = decide(P[m][te], X[te], th)
            y2, _ = at_lead(P[m][te], X[te], len(LEADS) - 1)
            r, c = errors(y[te], yh)
            row.update(theta=th, fallback=best is None, n_test=len(te), n_test_go=int(y[te].sum()),
                       test_macro_f1=bench_eval._macro_f1(y[te], yh), test_d_vs_2s=bench_eval._macro_f1(y[te], yh) - bench_eval._macro_f1(y[te], y2),
                       test_t_mean=float(tc.mean()), test_risky=r, test_conservative=c)
            sel.append(row)
    sel = pd.DataFrame(sel)
    sel.to_csv(f"{args.out}/n2_theta_selection.csv", index=False)

    # ---- policies on the outer predictions (band events ib)
    pol = {}
    for m in MODELS:
        Pm, Xm = P[m][ib], X[ib]
        pol[(m, "theta=0.05 (fixed)")] = decide(Pm, Xm, 0.05)
        pol[(m, "theta=0.1")] = decide(Pm, Xm, 0.1)
        yh, tc = np.zeros(len(ib), int), np.zeros(len(ib))
        for th in np.unique(chosen[m][ib]):
            k = chosen[m][ib] == th
            yh[k], tc[k] = decide(Pm[k], Xm[k], th)
        pol[(m, "theta=train-selected (risky<=0.10 & cons<=0.20)")] = (yh, tc)
        for j, L in enumerate(LEADS):
            pol[(m, f"all@{L:.1f}s")] = at_lead(Pm, Xm, j)
    yb = y[ib]
    cyc = bench_eval.cycles(ev)[ib]
    uc = np.unique(cyc)
    mem = {c: np.where(cyc == c)[0] for c in uc}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([mem[c] for c in rng.choice(uc, len(uc))]) for _ in range(args.n_boot)]
    boot = {k: pd.DataFrame([stats(yb[d], v[0][d], v[1][d]) for d in draws]) for k, v in pol.items()}
    rows = []
    for (m, p), (yh, tc) in pol.items():
        if p.startswith("all@") and p != "all@2.0s":
            continue
        pt, bs, ref = stats(yb, yh, tc), boot[(m, p)], boot[(m, "all@2.0s")]
        r = dict(model=m, policy=p, n=len(ib), n_go=int(yb.sum()))
        for k in ("macro_f1", "risky", "conservative", "t_mean"):
            r[k] = pt[k]
            r[f"{k}_lo"], r[f"{k}_hi"] = np.nanpercentile(bs[k], [2.5, 97.5])
        r["t_median"] = pt["t_median"]
        r["d_macro_f1_vs_2s"] = pt["macro_f1"] - stats(yb, *pol[(m, "all@2.0s")])["macro_f1"]
        r["d_macro_f1_vs_2s_lo"], r["d_macro_f1_vs_2s_hi"] = np.nanpercentile(bs["macro_f1"] - ref["macro_f1"], [2.5, 97.5])
        for L in LEADS:
            r[f"committed_at_{L:.1f}s"] = float(np.mean(np.isclose(tc, L)))  # share committed exactly at this lead
        for L in LEADS:
            r[f"committed_by_{L:.1f}s"] = float(np.mean(tc <= L))
        if "train-selected" in p:
            r["theta_counts"] = "; ".join(f"{t:g}x{c}" for t, c in sel[sel.model == m].theta.value_counts().sort_index().items())
            r["fallback_folds"] = int(sel[sel.model == m].fallback.sum())
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(f"{args.out}/n2_staged.csv", index=False)

    # ---- 2. matched wait: theta = 0.1 vs commit-all at the closest lead (event bootstrap)
    rng = np.random.default_rng(1)
    ed = [rng.integers(0, len(ib), len(ib)) for _ in range(args.n_boot)]
    mw = []
    for m in MODELS:
        yh, tc = pol[(m, "theta=0.1")]
        tm = float(tc.mean())
        j = int(np.argmin([abs(L - tm) + 1e-9 * (L < tm) for L in LEADS]))
        ya, _ = pol[(m, f"all@{LEADS[j]:.1f}s")]
        f = lambda ix: bench_eval._macro_f1(yb[ix], yh[ix]) - bench_eval._macro_f1(yb[ix], ya[ix])
        bs = np.array([f(ix) for ix in ed])
        mw.append(dict(model=m, staged="theta=0.1", staged_t_mean=tm, matched_lead_s=LEADS[j], staged_macro_f1=bench_eval._macro_f1(yb, yh),
                       matched_macro_f1=bench_eval._macro_f1(yb, ya), diff=f(np.arange(len(ib))), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), n=len(ib)))
    mw = pd.DataFrame(mw)
    mw.to_csv(f"{args.out}/n2_matched_wait.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(res[["model", "policy", "macro_f1", "macro_f1_lo", "macro_f1_hi", "d_macro_f1_vs_2s", "d_macro_f1_vs_2s_lo", "d_macro_f1_vs_2s_hi",
                   "risky", "conservative", "t_mean", "t_median"]].round(3).to_string(index=False))
        print(res.loc[res.policy.str.contains("train"), ["model", "theta_counts", "fallback_folds"]].to_string(index=False))
        print(mw.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
