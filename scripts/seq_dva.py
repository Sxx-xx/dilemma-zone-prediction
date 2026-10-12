"""Sequence models on (d, v, a) instead of (x, y, v, a), leaders only; nothing else changed (nets, 3 seeds, 20 steps).

The stored window is (x, y, v, a) in the approach frame with x = -d (features.build), so the (d, v, a) input is
(-x, v, a): the lateral position y along the stop line is dropped and x is sign-flipped (irrelevant after
standardisation). Both variants are run side by side:
  (a) FLUID FI leaders (88), leave-one-video-out (8 folds)
  (b) fitted on all 88 FLUID leaders, applied unchanged to the selected SinD sample leaders (Changchun, 29)
Cycle bootstrap 95 % CIs, paired differences vs rule A. Outputs: --out/{fluid_loo,sind_external}.csv

  .venv/bin/python scripts/seq_dva.py --out results/fluid_fi/leader_follower/leader/seq_dva
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import bench_eval
from dzp.bench_models import LSTMSeq, RuleA, TransformerSeq
from external_sind_sample import sind_events
from run_benchmark import at


def dva(S):
    return np.stack([-S[..., 0], S[..., 2], S[..., 3]], -1)


class _DVA:
    def fit(self, F, S, y):
        return super().fit(F, dva(S), y)

    def proba(self, F, S):
        return super().proba(F, dva(S))

    score = proba


class LSTMDVA(_DVA, LSTMSeq):
    name = "lstm_dva"


class TransformerDVA(_DVA, TransformerSeq):
    name = "transformer_dva"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sind-root", default="data/raw/sind")
    ap.add_argument("--fluid-event-dir", default="data/raw/fluid/results/benchmark_fi_leader")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    models = [(RuleA, {}), (LSTMSeq, {"n_seeds": args.n_seeds}), (LSTMDVA, {"n_seeds": args.n_seeds}),
              (TransformerSeq, {"n_seeds": args.n_seeds}), (TransformerDVA, {"n_seeds": args.n_seeds})]

    fe = pd.read_csv(f"{args.fluid_event_dir}/events_through.csv")
    ff = pd.read_csv(f"{args.fluid_event_dir}/features_through.csv")
    Ftr, Str = at(ff, {0.0: np.load(f"{args.fluid_event_dir}/sequences_through.npz")["lead+0.0"]}, 0.0, np.arange(len(fe)))
    ytr = fe["label"].values.astype(int)
    assert len(fe) == 88 and (Ftr["ldr_id"] < 0).all(), "expected the 88 FLUID FI leaders"

    # (a) FLUID leave-one-video-out
    P = bench_eval.oof(models, Ftr, Str, fe, "group")
    a = bench_eval.bootstrap(fe, P, None, args.n_boot, ref=RuleA.name)
    a.to_csv(f"{args.out}/fluid_loo.csv", index=False)
    print(a[a.metric != "brier"].round(3).to_string(), flush=True)

    # (b) all FLUID leaders -> SinD sample leaders
    ev, _, F, S, _, _ = sind_events(args.sind_root)
    sel = np.where(ev["selected"] & ev["leader"])[0]
    et, Ft, St = ev.iloc[sel].reset_index(drop=True), F.iloc[sel].reset_index(drop=True), S[sel]
    Pb = {}
    for M, kw in models:
        m = M(**kw).fit(Ftr, Str, ytr)
        Pb[M.name] = (m.proba(Ft, St), m.score(Ft, St))
    b = bench_eval.bootstrap(et, Pb, None, args.n_boot, ref=RuleA.name)
    b.to_csv(f"{args.out}/sind_external.csv", index=False)
    print(f"SinD leaders n={len(et)} go={int(et.label.sum())}")
    print(b[b.metric != "brier"].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
