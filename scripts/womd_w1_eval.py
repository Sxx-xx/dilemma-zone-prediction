"""W1 step 7: evaluation on the WOMD leader tables (scripts/womd_w1_aggregate.py output, project schema).

Runs the existing V2X-Seq machinery on --event-dir:
  loio   : scripts/v2xseq_eval.py --tasks loio (rules A/B, LR dv/dva, MLP, LSTM/Transformer (x,y,v,a) and (d,v,a);
           intersections with >= --min-site events tested; all / dz / v<10 / v>=10; cycle bootstrap)
  D      : scripts/v2xseq_followup.py --tasks C,D (rule A t_r sweep, driver boundary per site, calibrated rule A under
           full LOIO with band-only LR, pairwise CIs; V2X-Seq 2.67 s zero-shot added here)
  curve  : scripts/v2xseq_eval.py --tasks curve_dz (band learning curve, 10 draws)
  yellow : this script -- boundary t_50 per yellow-length bin and "calibration gain vs (yellow - boundary)" table,
           rule A with the V2X-Seq boundary 2.67 s (zero-shot) and with the recorded yellow, cycle bootstrap.

  .venv/bin/python scripts/womd_w1_eval.py --event-dir data/raw/womd/w1/tables_validation --out results/womd/w1/validation/eval --steps loio,D,yellow,curve
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

PY = sys.executable


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def yellow_table(args):
    from dzp import bench_eval
    from dzp.bench_models import RuleA
    from v2xseq_eval import load_tables
    from v2xseq_followup import _t50
    ev, F, S = load_tables(args.event_dir)
    y, tt, tr = ev["label"].values.astype(int), F["tt"].values, F["t_r"].values
    cyc = bench_eval.cycles(ev)
    out = f"{args.out}/yellow"
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)

    def boot_t50(idx):
        uc = np.unique(cyc[idx])
        mem = {c: idx[cyc[idx] == c] for c in uc}
        vals = np.array([_t50(*(lambda ix: (tt[ix], y[ix]))(np.concatenate([mem[c] for c in rng.choice(uc, len(uc))]))) for _ in range(args.n_boot)], float)
        return np.nanpercentile(vals, [2.5, 97.5])

    rows = []
    bins = [(2.5, 3.25), (3.25, 3.75), (3.75, 4.25), (4.25, 6.0)]
    groups = [("all", np.ones(len(y), bool))] + [(f"yellow {lo:.2f}-{hi:.2f}", (tr >= lo) & (tr < hi)) for lo, hi in bins]
    sites = ev["group"].value_counts()
    groups += [(f"site {s}", ev["group"].values == s) for s in sites.index if sites[s] >= args.min_site]
    for name, m in groups:
        idx = np.where(m & (tt <= 12))[0]
        if len(idx) < 20 or len(np.unique(y[idx])) < 2:
            continue
        t50 = _t50(tt[idx], y[idx])
        lo, hi = boot_t50(idx)
        P = {"ruleA_recorded": ((tr[idx] - tt[idx] > 0).astype(float), tr[idx] - tt[idx]),
             "ruleA_t50_insample": ((t50 - tt[idx] > 0).astype(float), t50 - tt[idx]),
             "ruleA_v2x_2.67": ((2.67 - tt[idx] > 0).astype(float), 2.67 - tt[idx])}
        b = bench_eval.bootstrap(ev.iloc[idx].reset_index(drop=True), P, None, args.n_boot, ref="ruleA_recorded")
        b = b[b.metric == "macro_f1"].set_index("model")
        rows.append(dict(group=name, n=len(idx), n_go=int(y[idx].sum()), yellow_median=float(np.median(tr[idx])), t50=t50, t50_lo=lo, t50_hi=hi,
                         gap_yellow_minus_t50=float(np.median(tr[idx]) - t50),
                         f1_recorded=b.loc["ruleA_recorded", "value"], f1_t50=b.loc["ruleA_t50_insample", "value"],
                         gain_t50=b.loc["ruleA_t50_insample", "diff_vs_ref"], gain_lo=b.loc["ruleA_t50_insample", "diff_lo"], gain_hi=b.loc["ruleA_t50_insample", "diff_hi"],
                         f1_v2x267=b.loc["ruleA_v2x_2.67", "value"], gain_v2x267=b.loc["ruleA_v2x_2.67", "diff_vs_ref"], gain_v2x_lo=b.loc["ruleA_v2x_2.67", "diff_lo"],
                         gain_v2x_hi=b.loc["ruleA_v2x_2.67", "diff_hi"]))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/boundary_by_yellow.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="loio,D,yellow,curve")
    ap.add_argument("--min-site", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    steps = args.steps.split(",")
    if "loio" in steps:
        run([PY, "scripts/v2xseq_eval.py", "--event-dir", args.event_dir, "--out", args.out, "--tasks", "loio", "--min-site", str(args.min_site), "--n-boot", str(args.n_boot)])
    if "D" in steps:
        run([PY, "scripts/v2xseq_followup.py", "--event-dir", args.event_dir, "--out", f"{args.out}/followup", "--tasks", "C,D", "--min-site", str(args.min_site),
             "--n-boot", str(args.n_boot), "--threads", "4"])
    if "yellow" in steps:
        yellow_table(args)
    if "curve" in steps:
        run([PY, "scripts/v2xseq_eval.py", "--event-dir", args.event_dir, "--out", args.out, "--tasks", "curve_dz", "--min-site", str(args.min_site),
             "--curve-reps", "10", "--workers", str(args.workers)])


if __name__ == "__main__":
    main()
