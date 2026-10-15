"""L. Dilemma-zone lead time and probability analysis.

Band membership fixed at yellow onset (d/v in [2.5, 5.5] s at lead 0). For each lead L in 0, 0.5, 1, 1.5, 2 s the
inputs are (d, v, a) at onset + L and the 2 s window ending there; events whose vehicle has already crossed the stop
line at that time (d <= 0), or whose track has ended, are excluded (counts reported). Models (same folds as the
project's evaluation, fold unit = ev['group']): band-only LR(d,v,a) (trained on the training folds' band events at
lead L), MLP(d,v,a) and LSTM(d,v,a) (trained on all training events at lead L), calibrated rule A (go iff d/v at
lead L < t50(L), t50 fitted on the training folds' leaders at lead L). CIs: bootstrap over ev['cycle'].
Probability analysis per lead for band LR and MLP: reliability curve (10 bins) + Brier, share of events with
0.2 < P < 0.8, macro-F1 and n of the confident events only.

Outputs under --out: leadtime.csv, exclusions.csv, calibration.csv, confidence.csv, t50_by_lead.csv, preds_*.csv

  .venv/bin/python scripts/leadtime_dz.py --event-dir data/raw/fluid/results/leadtime_fi_leader --out results/fluid_fi/leadtime_dz --min-site 1
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dzp import bench_eval
from dzp.bench_models import LogRegDVA, MLPDVA
from run_benchmark import at
from seq_dva import LSTMDVA

DZ_LO, DZ_HI, TT_MAX = 2.5, 5.5, 12.0
LEADS = [0.0, 0.5, 1.0, 1.5, 2.0]


class _BandLR(LogRegDVA):
    name = "band_logreg_dva"

    def __init__(self, band_mask=None, **_):
        super().__init__()
        self.band_mask = band_mask

    def fit(self, F, S, y):
        m = self.band_mask
        return super().fit(F[m], S[m], y[m]) if m is not None and len(np.unique(y[m])) == 2 else None


def t50(x, y):
    if len(np.unique(y)) < 2:
        return np.nan
    lr = LogisticRegression(max_iter=1000).fit(x[:, None], y)
    return -lr.intercept_[0] / lr.coef_[0, 0] if lr.coef_[0, 0] < 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-site", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    ev = pd.read_csv(f"{args.event_dir}/events_through.csv")
    feat = pd.read_csv(f"{args.event_dir}/features_through.csv")
    z = np.load(f"{args.event_dir}/sequences_through.npz")
    seqs = {L: z[f"lead{L:+.1f}"] for L in LEADS}
    if "segment" not in ev.columns:
        ev["segment"] = ev["group"]
    F0, _ = at(feat, seqs, 0.0, np.arange(len(ev)))
    ok0 = F0["t_r"].notna().values
    band0 = ((F0["tt"] >= DZ_LO) & (F0["tt"] <= DZ_HI)).values & ok0
    y = ev["label"].values.astype(int)
    sites = ev["group"].value_counts()
    test_sites = sorted(sites.index[sites >= args.min_site])
    folds = [(s, np.where(ev["group"].values != s)[0], np.where(ev["group"].values == s)[0]) for s in test_sites]
    print(f"events {len(ev)} band {int(band0.sum())} folds {len(folds)}", flush=True)
    rows, excl, cal, conf, t50rows = [], [], [], [], []
    for L in LEADS:
        F, S = at(feat, seqs, L, np.arange(len(ev)))
        d = F["d"].values
        avail = ok0 & np.isfinite(d) & (F["v"].notna().values)
        crossed = avail & (d <= 0)
        usable = avail & ~crossed
        excl.append(dict(lead_s=L, band=int(band0.sum()), band_track_ended=int((band0 & ~avail).sum()), band_crossed=int((band0 & crossed).sum()),
                         band_usable=int((band0 & usable).sum()), band_usable_go=int(y[band0 & usable].sum()), all_usable=int(usable.sum())))
        P = {}
        for name in ("band_logreg_dva", "mlp_dva", "lstm_dva", "ruleA_calibrated"):
            P[name] = np.full(len(y), np.nan)
        t50s = []
        for site, tr, te in folds:
            tr_u, te_u = tr[usable[tr]], te[usable[te] & band0[te]]
            if len(te_u) == 0 or len(np.unique(y[tr_u])) < 2:
                continue
            tr_b = tr_u[band0[tr_u]]
            m = _BandLR(band_mask=band0[tr_u]).fit(F.iloc[tr_u], S[tr_u], y[tr_u])
            if m is not None:
                P["band_logreg_dva"][te_u] = m.proba(F.iloc[te_u], S[te_u])
            P["mlp_dva"][te_u] = MLPDVA().fit(F.iloc[tr_u], S[tr_u], y[tr_u]).proba(F.iloc[te_u], S[te_u])
            P["lstm_dva"][te_u] = LSTMDVA(n_seeds=args.n_seeds).fit(F.iloc[tr_u], S[tr_u], y[tr_u]).proba(F.iloc[te_u], S[te_u])
            k = tr_u[F["tt"].values[tr_u] <= TT_MAX]
            t = t50(F["tt"].values[k], y[k])
            t50s.append(t)
            P["ruleA_calibrated"][te_u] = (F["tt"].values[te_u] < t).astype(float)
        t50rows.append(dict(lead_s=L, t50_mean=float(np.nanmean(t50s)), t50_min=float(np.nanmin(t50s)), t50_max=float(np.nanmax(t50s)), folds=len(t50s)))
        mask = band0 & usable & np.isfinite(P["mlp_dva"])
        Pd = {k: (v, v) for k, v in P.items()}
        b = bench_eval.bootstrap(ev, Pd, mask, args.n_boot, ref="ruleA_calibrated")
        b = b[b.metric.isin(["macro_f1", "brier"])].assign(lead_s=L)
        rows.append(b)
        pd.DataFrame(P).assign(band=band0, usable=usable, segment=ev["segment"], track_id=ev["track_id"]).to_csv(f"{args.out}/preds_lead{L:+.1f}.csv", index=False)
        # probability analysis
        for name in ("band_logreg_dva", "mlp_dva"):
            p = P[name][mask]
            yy = y[mask]
            bins = np.clip((p * 10).astype(int), 0, 9)
            for bi in range(10):
                mm = bins == bi
                cal.append(dict(lead_s=L, model=name, bin=bi, p_lo=bi / 10, p_hi=(bi + 1) / 10, n=int(mm.sum()), p_mean=float(p[mm].mean()) if mm.any() else np.nan,
                                go_rate=float(yy[mm].mean()) if mm.any() else np.nan))
            unc = (p > 0.2) & (p < 0.8)
            cf = ~unc
            conf.append(dict(lead_s=L, model=name, n=int(mask.sum()), brier=float(np.mean((p - yy) ** 2)), uncertain_share=float(unc.mean()),
                             confident_n=int(cf.sum()), confident_go=int(yy[cf].sum()),
                             confident_macro_f1=bench_eval._macro_f1(yy[cf], (p[cf] >= 0.5).astype(int)) if cf.any() else np.nan,
                             uncertain_macro_f1=bench_eval._macro_f1(yy[unc], (p[unc] >= 0.5).astype(int)) if unc.any() else np.nan))
        print(f"[lead {L:+.1f}] usable band {int(mask.sum())}: " + ", ".join(f"{r.model} {r.value:.3f}" for r in b[b.metric == 'macro_f1'].itertuples()), flush=True)
    lt = pd.concat(rows, ignore_index=True)
    lt.to_csv(f"{args.out}/leadtime.csv", index=False)
    pd.DataFrame(excl).to_csv(f"{args.out}/exclusions.csv", index=False)
    pd.DataFrame(cal).to_csv(f"{args.out}/calibration.csv", index=False)
    pd.DataFrame(conf).to_csv(f"{args.out}/confidence.csv", index=False)
    pd.DataFrame(t50rows).to_csv(f"{args.out}/t50_by_lead.csv", index=False)
    q = lt[lt.metric == "macro_f1"].pivot(index="model", columns="lead_s", values="value").round(3)
    print(q.to_string(), flush=True)
    first = {m: next((L for L in LEADS if q.loc[m, L] >= 0.8), None) for m in q.index}
    print("first lead reaching 0.8:", first, flush=True)
    pd.DataFrame([dict(model=m, first_lead_ge_0_8=v) for m, v in first.items()]).to_csv(f"{args.out}/first_lead_0_8.csv", index=False)


if __name__ == "__main__":
    main()
