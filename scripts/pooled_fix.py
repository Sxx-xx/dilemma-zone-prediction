"""P-fix. Pooled validation follow-up (new *_fix files under results/pooled/fix; existing P files untouched).

1. Pooled LOIO calibrated rule A re-done: t_50(L) per fold from the training folds' non-WOMD events (V2X-Seq + FLUID
   + SinD, usable at lead L, d/v <= 12 s), leads 0..2 s; fold t_50 ranges; MLP / LSTM / band-only LR minus this rule
   (all and band, pooled and per dataset, event bootstrap 1,000).
2. Leave-one-dataset-out per lead, band macro-F1 with cycle-bootstrap CIs, for MLP, LSTM, band-only LR, calibrated
   rule A (t_50 from the training datasets without WOMD), rule A; next to the test dataset's own LOIO
   (L section: V2X-Seq 23-fold, WOMD group 10-fold, FLUID video 8-fold; SinD recording 5-fold run here with
   scripts/leadtime_dz.py; rule A needs no training and is the same number in both columns).
3. Pooled LOIO without WOMD (V2X-Seq + FLUID + SinD, preds from scripts/pooled_eval.py --table table_no_womd):
   band macro-F1 per lead (pooled and per dataset). The N2 theta = 0.05 row comes from scripts/staged_n2.py.

  .venv/bin/python scripts/pooled_fix.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from pooled_eval import LEADS, TT_MAX, _preds, cyc_draws, ev_draws, load, metrics, t50_fit
from dzp.bench_eval import _macro_f1

DSN = ["v2xseq", "fluid", "sind", "womd"]
SELF = {"v2xseq": "results/v2xseq/leadtime_dz/leadtime.csv", "womd": "results/womd/leadtime_dz/leadtime.csv",
        "fluid": "results/fluid_fi/leadtime_dz/leadtime.csv", "sind": "results/pooled/fix/sind_self_leadtime_dz/leadtime.csv"}


def cal_fix(D, L, fold_of, test_mask_fn):
    """Calibrated rule A with t_50 from the training events outside WOMD. fold_of: array of fold names (None = untested)."""
    ev, y = D["ev"], D["y"]
    F = D["F"][L]
    tt = F["tt"].values
    usable = D["usable"][L]
    ds = ev["dataset"].values
    P, S = np.full(len(y), np.nan), np.full(len(y), np.nan)
    rows = []
    for f in pd.unique(fold_of[fold_of != None]):  # noqa: E711
        te = np.where(fold_of == f)[0]
        trn = np.where(test_mask_fn(f) & usable & (ds != "womd") & (tt <= TT_MAX))[0]
        t = t50_fit(tt[trn], y[trn])
        P[te], S[te] = (tt[te] < t).astype(float), t - tt[te]
        rows.append(dict(lead_s=L, fold=f, n_train_non_womd=len(trn), t50=t))
    return P, S, pd.DataFrame(rows)


def score(y, p, s, mask, cyc, n_boot):
    mk = mask & np.isfinite(p)
    pt = metrics(y[mk], p[mk], s[mk])
    bs = pd.DataFrame([metrics(y[ix], p[ix], s[ix]) for ix in cyc_draws(cyc, mk, n_boot)])
    return dict(n=int(mk.sum()), n_go=int(y[mk].sum()), macro_f1=pt["macro_f1"], macro_f1_lo=np.nanpercentile(bs.macro_f1, 2.5),
                macro_f1_hi=np.nanpercentile(bs.macro_f1, 97.5), auroc=pt["auroc"], brier=pt["brier"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/raw/pooled/table")
    ap.add_argument("--table-no-womd", default="data/raw/pooled/table_no_womd")
    ap.add_argument("--out", default="results/pooled/fix")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    D = load(args.table)
    ev, y, band0 = D["ev"], D["y"], D["band0"]
    n, ds, cyc, site = len(y), ev["dataset"].values, ev["cycle"].values, ev["site_id"].values
    sets = [("pooled", np.ones(n, bool))] + [(d, ds == d) for d in DSN]

    # ---- 1. pooled LOIO, calibrated rule A with non-WOMD t_50
    t50s, diffs, calrows = [], [], []
    for L in LEADS:
        P, fold, _ = _preds(args.table, "loio", L, n)
        pc, sc, tf = cal_fix(D, L, fold, lambda f: site != f)
        t50s.append(tf)
        tested = np.isfinite(P["mlp_dva"][0])
        for sub, sm in (("all", np.ones(n, bool)), ("band", band0)):
            for name, m in sets:
                mm = m & sm & tested
                calrows.append(dict(lead_s=L, subset=sub, set=name, model="ruleA_calibrated_fix", **score(y, pc, sc, mm, cyc, args.n_boot)))
                draws = ev_draws(mm, args.n_boot)
                idx = np.where(mm)[0]
                for mod in ("mlp_dva", "lstm_dva", "band_logreg_dva"):
                    if mod == "band_logreg_dva" and sub == "all":
                        continue
                    pm = P[mod][0]
                    f = lambda ix: _macro_f1(y[ix], (pm[ix] >= 0.5).astype(int)) - _macro_f1(y[ix], (pc[ix] >= 0.5).astype(int))
                    bs = np.array([f(ix) for ix in draws])
                    diffs.append(dict(lead_s=L, subset=sub, set=name, model=mod, n=len(idx), model_macro_f1=_macro_f1(y[idx], (pm[idx] >= 0.5).astype(int)),
                                      cal_fix_macro_f1=_macro_f1(y[idx], (pc[idx] >= 0.5).astype(int)), diff=f(idx),
                                      ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    t50s = pd.concat(t50s, ignore_index=True)
    t50s.to_csv(f"{args.out}/P2_fold_t50_non_womd_fix.csv", index=False)
    rng = t50s.groupby("lead_s").t50.agg(["min", "max", "median", "size"])
    rng.to_csv(f"{args.out}/P2_fold_t50_range_fix.csv")
    pd.DataFrame(calrows).to_csv(f"{args.out}/P2_calibrated_ruleA_fix.csv", index=False)
    pd.DataFrame(diffs).to_csv(f"{args.out}/P2_models_vs_calibrated_ruleA_fix.csv", index=False)
    print(rng.round(3).to_string(), flush=True)

    # ---- 2. leave-one-dataset-out per lead, band, next to the self LOIO
    rows = []
    for L in LEADS:
        P, fold, _ = _preds(args.table, "lodo", L, n)
        pc, sc, tf = cal_fix(D, L, fold, lambda f: ds != f)
        tested = np.isfinite(P["mlp_dva"][0])
        for d in DSN:
            mm = (ds == d) & band0 & tested
            t50 = tf.loc[tf.fold == d, "t50"].iloc[0]
            for mod, (p, s) in [("mlp_dva", P["mlp_dva"]), ("lstm_dva", P["lstm_dva"]), ("band_logreg_dva", P["band_logreg_dva"]),
                                ("ruleA_calibrated_fix", (pc, sc)), ("ruleA", P["ruleA"])]:
                r = dict(test=d, train="+".join(x for x in DSN if x != d), lead_s=L, model=mod, **score(y, p, s, mm, cyc, args.n_boot))
                if mod == "ruleA_calibrated_fix":
                    r["t50_train_non_womd"] = t50
                rows.append(r)
    lodo = pd.DataFrame(rows)
    # self LOIO (per-dataset runs); rule A computed directly (no training)
    selfrows = []
    for d, path in SELF.items():
        q = pd.read_csv(path)
        q = q[q.metric == "macro_f1"].drop_duplicates(["model", "lead_s"])
        for r in q.itertuples():
            m = {"ruleA_calibrated": "ruleA_calibrated_fix"}.get(r.model, r.model)
            selfrows.append(dict(test=d, lead_s=r.lead_s, model=m, self_macro_f1=r.value, self_lo=r.ci_lo, self_hi=r.ci_hi, self_n=r.n, self_source=path))
    for L in LEADS:
        P, _, _ = _preds(args.table, "lodo", L, n)
        for d in DSN:
            r = lodo[(lodo.test == d) & (lodo.lead_s == L) & (lodo.model == "ruleA")].iloc[0]
            selfrows.append(dict(test=d, lead_s=L, model="ruleA", self_macro_f1=r.macro_f1, self_lo=r.macro_f1_lo, self_hi=r.macro_f1_hi, self_n=r.n,
                                 self_source="no training: same as LODO"))
    lodo = lodo.merge(pd.DataFrame(selfrows), on=["test", "lead_s", "model"], how="left")
    lodo.to_csv(f"{args.out}/P3_lodo_band_by_lead_fix.csv", index=False)
    with pd.option_context("display.width", 250):
        print(lodo[["test", "lead_s", "model", "n", "macro_f1", "macro_f1_lo", "macro_f1_hi", "self_macro_f1", "self_lo", "self_hi", "self_n"]].round(3).to_string(index=False))

    # ---- 3. pooled LOIO without WOMD, band per lead
    D3 = load(args.table_no_womd)
    ev3, y3, b3 = D3["ev"], D3["y"], D3["band0"]
    n3, ds3, cyc3 = len(y3), ev3["dataset"].values, ev3["cycle"].values
    rows = []
    for L in LEADS:
        P, _, _ = _preds(args.table_no_womd, "loio", L, n3)
        tested = np.isfinite(P["mlp_dva"][0])
        for name, m in [("pooled_3", np.ones(n3, bool))] + [(d, ds3 == d) for d in DSN[:3]]:
            for mod in ("mlp_dva", "lstm_dva", "band_logreg_dva", "logreg_dva", "ruleA_calibrated", "ruleA"):
                p, s = P[mod]
                rows.append(dict(lead_s=L, set=name, model=mod, **score(y3, p, s, m & b3 & tested, cyc3, args.n_boot)))
    nw = pd.DataFrame(rows)
    nw.to_csv(f"{args.out}/P2_no_womd_loio_band_fix.csv", index=False)
    print(nw.pivot_table(index=["set", "model"], columns="lead_s", values="macro_f1").round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
