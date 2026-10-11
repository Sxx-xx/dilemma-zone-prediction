"""NGSIM Lankershim int2 (Main St, RSM089) as auxiliary data: rule d/v<t_r vs LR / MLP at three signal
offsets (k = -3.6 / 0 / +5.0 s, the range left by the red-crossing check), to see whether the ranking holds.

Split: repeated StratifiedGroupKFold (5 folds, groups = yellow cycles, N_REP seeds) because there is one
intersection and only ~15 go events. Bootstrap: paired cluster bootstrap over cycles of
macro-F1(model) - macro-F1(rule), averaging the difference over the repeats inside each resample.

Outputs: results/ngsim_lankershim/eval/{metrics.csv, bootstrap.csv, ranking.csv, events_k*.csv}
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import lankershim_check as L
from dzp import evaluate, ngsim
from dzp.events import HIST_S
from dzp.models import MLP, LogReg, RuleDV

OUT = "results/ngsim_lankershim/eval"
OFFSETS = (-36, 0, 50)          # frames (signal frame = trajectory frame + k)
N_REP = 20
N_BOOT = 2000
MODELS = [RuleDV, LogReg, MLP]


def main():
    os.makedirs(OUT, exist_ok=True)
    t = L.load_tracks()
    ngsim.INTERSECTIONS = {i: f"int{i}_RSM{n:03d}" for i, n in L.MAPPING.items()}
    sig = L.build_signals(L.MAPPING, "start", "lead")
    stops = ngsim.fill_missing_stop_lines(ngsim.estimate_stop_lines(t, sig))
    sig, stops = sig[sig.int_id == 2], stops[stops.int_id == 2]
    onset = int(round(HIST_S / ngsim.DT)) - 1
    rule = RuleDV.name

    met_rows, boot_rows = [], []
    for k in OFFSETS:
        ks = k * ngsim.DT
        ev, seq, _ = ngsim.extract_events(t, sig, stops, k=k)
        ev.to_csv(f"{OUT}/events_k{ks:+.1f}.csv", index=False)
        y = ev["label"].values.astype(int)
        preds = []
        for rep in range(N_REP):
            res, pr = evaluate.run(ev, seq, onset, ngsim.DT, "strat", models=MODELS, seed=rep)
            res = res[res.split == "pooled"].assign(rep=rep)
            met_rows.append(res.assign(k_s=ks, n_events=len(ev), n_go=int(y.sum())))
            preds.append(pr.assign(rep=rep))
        preds = pd.concat(preds)

        # paired cluster bootstrap over cycles, difference averaged over repeats within each resample
        rng = np.random.default_rng(0)
        cyc = ev["cycle_id"].values
        ucyc = np.unique(cyc)
        members = {c: np.where(cyc == c)[0] for c in ucyc}
        draws = [np.concatenate([members[c] for c in rng.choice(ucyc, len(ucyc))]) for _ in range(N_BOOT)]
        for tau, g in preds.groupby("tau"):
            P = {(m, r): gg.sort_values("row")["pred"].values for (m, r), gg in g.groupby(["model", "rep"])}
            for M in MODELS[1:]:
                diffs = []
                for idx in draws:
                    yb = y[idx]
                    d = [evaluate._macro_f1_fast(yb, P[(M.name, r)][idx]) - evaluate._macro_f1_fast(yb, P[(rule, r)][idx])
                         for r in range(N_REP)]
                    diffs.append(np.mean(d))
                diffs = np.array(diffs)
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                boot_rows.append(dict(k_s=ks, tau=tau, model=M.name, mean_diff=diffs.mean(), ci_lo=lo, ci_hi=hi,
                                      p_model_better=float((diffs > 0).mean())))
        print(f"k={ks:+.1f}s: {len(ev)} events, go={int(y.sum())}, cycles={len(ucyc)}", flush=True)

    met = pd.concat(met_rows)
    summ = (met.groupby(["k_s", "tau", "model"])
            .agg(macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"), rec_stop=("rec_stop", "mean"),
                 prec_stop=("prec_stop", "mean"), n_events=("n_events", "first"), n_go=("n_go", "first"))
            .reset_index())
    summ.to_csv(f"{OUT}/metrics.csv", index=False)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(f"{OUT}/bootstrap.csv", index=False)
    rank = (summ.sort_values(["k_s", "tau", "macro_f1"], ascending=[True, True, False])
            .groupby(["k_s", "tau"])["model"].apply(lambda s: " > ".join(s)).reset_index(name="ranking"))
    rank.to_csv(f"{OUT}/ranking.csv", index=False)
    print(summ.round(3).to_string())
    print(boot.round(3).to_string())
    print(rank.to_string())


if __name__ == "__main__":
    main()
