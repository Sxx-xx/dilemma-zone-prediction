"""Changchun t_r check for the SinD external test: per-cycle recorded yellow vs a fixed 3.0 s yellow.

Reads the per-leader rows written by scripts/external_sind_sample.py (--event-dir/preds_leaders.csv: d, v, a, t_r at
onset from the signal record, FLUID-trained LR predictions) and rescoring:
  recorded : t_r = time from the onset frame to the recorded red (as in the external test)
  fixed    : t_r = 3.0 s - (onset frame - recorded yellow onset), i.e. the same onset with a 3.0 s yellow
Only rule A uses t_r. Rule B uses (d, v); LR(d,v,a) / LR(d,v) and the sequence models take no t_r, so they are
unchanged by construction. Labels stay those of the recorded signal. Output: --out/t_r_check.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import bench_eval, sind
from dzp.bench_models import RuleA, RuleB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-dir", default="data/raw/sind/results/external_sample")
    ap.add_argument("--out", required=True)
    ap.add_argument("--yellow", type=float, default=3.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    p = pd.read_csv(f"{args.event_dir}/preds_leaders.csv")
    k0 = np.ceil(p["f_yellow"].values)
    rows = []
    for variant, t_r in (("recorded", p["t_r"].values), ("fixed", args.yellow - (k0 - p["f_yellow"].values) * sind.DT)):
        F = pd.DataFrame({"d": p["d"], "v": p["v"], "a": p["a"], "t_r": t_r})
        F["tt"] = F["d"] / np.maximum(F["v"], 0.1)
        F["dreq"] = np.where(F["d"] > 0, F["v"] ** 2 / (2 * np.maximum(F["d"], 0.1)), np.inf)
        P = {m.name: (m.proba(F, None), m.score(F, None)) for m in (RuleA(), RuleB(decel=3.0))}
        P["logreg_dva"] = (p["logreg_dva"].values, p["logreg_dva"].values)
        b = bench_eval.bootstrap(p, P, None, args.n_boot, ref=RuleA.name)
        b["ruleA_go_pred"] = int(P[RuleA.name][0].sum())
        rows.append(b.assign(t_r=variant))
    df = pd.concat(rows, ignore_index=True)
    rec, fix = (pd.Series(RuleA().proba(pd.DataFrame({"d": p.d, "tt": p.d / np.maximum(p.v, 0.1), "t_r": t}), None))
                for t in (p["t_r"].values, args.yellow - (k0 - p["f_yellow"].values) * sind.DT))
    df["ruleA_flips"] = int((rec != fix).sum())
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(f"{args.out}/t_r_check.csv", index=False)
    print(df[df.metric != "brier"].round(3).to_string())
    print(p.assign(t_r_fixed=args.yellow - (k0 - p["f_yellow"].values) * sind.DT, ruleA_rec=rec, ruleA_fix=fix)
          .loc[rec != fix, ["f_yellow", "label", "d", "v", "t_r", "t_r_fixed", "ruleA_rec", "ruleA_fix"]].round(2).to_string())


if __name__ == "__main__":
    main()
