"""Sensor-noise robustness on the FLUID FI leaders (leave-one-video-out, 8 folds, no tuning).

Gaussian noise is added to the onset features of the *test* fold: d (sd 0.5 m, fixed), v (sd SD_V), a (sd SD_A),
full SD_V x SD_A grid plus a noise-free reference row. Rule A sees the same noisy d, v (tt = d / v recomputed).
  test-only    : LR(d,v,a), LR(d,v) fitted on clean training folds (once), applied to noisy test inputs
  noise-aware  : refitted in every repetition on training inputs with the same noise (fresh draws)
Each condition is repeated `--reps` times. Metric = macro-F1 averaged over repetitions; CI = cycle bootstrap of that
average (each bootstrap draw re-scores all repetitions), difference vs rule A paired on the same draws and the same
noise (rule A under the same noise as the model).

  .venv/bin/python scripts/noise_robustness.py --out results/fluid_fi/leader_follower/leader/noise
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import bench_eval
from dzp.bench_models import LogRegDV, LogRegDVA, RuleA
from run_benchmark import at

SD_D = 0.5
SD_V = (0.0, 0.1, 0.3, 0.5)
SD_A = (0.0, 0.25, 0.5, 1.0)


def noisy(F, sd, rng):
    G = F.copy()
    for c, s in zip(("d", "v", "a"), sd):
        if s > 0:
            G[c] = G[c].values + rng.normal(0, s, len(G))
    G["tt"] = G["d"].values / np.maximum(G["v"].values, 0.1)
    return G


def f1_reps(y, yhat):
    """macro-F1 per repetition; yhat (R, n)."""
    f = []
    for c in (0, 1):
        tp = ((yhat == c) & (y == c)).sum(1)
        den = (yhat == c).sum(1) + (y == c).sum()
        f.append(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0))
    return np.mean(f, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fluid-event-dir", default="data/raw/fluid/results/benchmark_fi_leader")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ev = pd.read_csv(f"{args.fluid_event_dir}/events_through.csv")
    feat = pd.read_csv(f"{args.fluid_event_dir}/features_through.csv")
    F, _ = at(feat, {0.0: np.load(f"{args.fluid_event_dir}/sequences_through.npz")["lead+0.0"]}, 0.0, np.arange(len(ev)))
    y = ev["label"].values.astype(int)
    assert len(ev) == 88, "expected the 88 FLUID FI leaders"
    folds = list(bench_eval.splits(ev, "group"))
    clean = {M.name: [M().fit(F.iloc[tr], None, y[tr]) for tr, _ in folds] for M in (LogRegDVA, LogRegDV)}

    cyc = bench_eval.cycles(ev)
    ucyc = np.unique(cyc)
    members = {c: np.where(cyc == c)[0] for c in ucyc}
    rng_b = np.random.default_rng(0)
    draws = [np.concatenate([members[c] for c in rng_b.choice(ucyc, len(ucyc))]) for _ in range(args.n_boot)]

    conds = [(0.0, 0.0, 0.0)] + [(SD_D, v, a) for v in SD_V for a in SD_A]
    rows = []
    for ci, sd in enumerate(conds):
        R = 1 if sd == (0.0, 0.0, 0.0) else args.reps
        H = {k: np.zeros((R, len(y)), int) for k in ("ruleA", "logreg_dva", "logreg_dv",
                                                      "logreg_dva+noise_train", "logreg_dv+noise_train")}
        for r in range(R):
            rng = np.random.default_rng([ci, r])
            Fn = noisy(F, sd, rng)                       # one test-noise draw shared by all models
            H["ruleA"][r] = RuleA().proba(Fn, None) >= 0.5
            for k, (tr, te) in enumerate(folds):
                for M in (LogRegDVA, LogRegDV):
                    H[M.name][r, te] = clean[M.name][k].proba(Fn.iloc[te], None) >= 0.5
                    Ftr = noisy(F.iloc[tr], sd, rng)
                    m = M().fit(Ftr, None, y[tr])
                    H[M.name + "+noise_train"][r, te] = m.proba(Fn.iloc[te], None) >= 0.5
        pt = {k: f1_reps(y, h).mean() for k, h in H.items()}
        bs = {k: np.array([f1_reps(y[ix], h[:, ix]).mean() for ix in draws]) for k, h in H.items()}
        for k in H:
            if sd == (0.0, 0.0, 0.0) and k.endswith("+noise_train"):
                continue
            diff = bs[k] - bs["ruleA"]
            rows.append(dict(sd_d=sd[0], sd_v=sd[1], sd_a=sd[2], model=k.replace("+noise_train", ""),
                             training="noise-aware" if k.endswith("+noise_train") else "clean", reps=R,
                             macro_f1=pt[k], ci_lo=np.percentile(bs[k], 2.5), ci_hi=np.percentile(bs[k], 97.5),
                             diff_vs_ruleA=pt[k] - pt["ruleA"], diff_lo=np.percentile(diff, 2.5),
                             diff_hi=np.percentile(diff, 97.5)))
        print(f"[noise] d={sd[0]} v={sd[1]} a={sd[2]} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}/noise_grid.csv", index=False)
    print(df.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
