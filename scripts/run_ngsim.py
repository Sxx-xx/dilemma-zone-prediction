"""NGSIM Peachtree: alignment checks, stop-line estimates, yellow-onset events, 4-model comparison
(within-site GroupKFold by cycle, and leave-one-intersection-out).

Outputs (results/ngsim_peachtree/): stop_lines.csv, alignment_scan.csv, events.csv, sequences.npz,
extraction_stats.json, metrics_within.csv, metrics_loio.csv
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import evaluate, ngsim
from dzp.events import HIST_S

RAW = "data/raw/ngsim/peachtree"
OUT = "results/ngsim_peachtree"
SWAP = {"1245-0100": "0400-0415", "0400-0415": "1245-0100"}


def red_fraction(cr, sig, k, swap=False, min_speed=None):
    """Share of through-vehicle stop-line crossings that happen on red."""
    n = red = 0
    for (per, i, d), g in cr.groupby(["period", "int_id", "dir"]):
        if min_speed is not None:
            g = g[g["speed"] >= min_speed]
        s = ngsim.signal_state(sig, SWAP[per] if swap else per, i, d, g["frame"].values, k)
        v = s != "?"
        n += v.sum()
        red += (s[v] == "R").sum()
    return red / max(n, 1)


def main(k: float = 0.0):
    out = OUT if k == 0 else f"{OUT}/sensitivity_k{int(k)}"
    os.makedirs(out, exist_ok=True)
    t = ngsim.load_tracks(f"{RAW}/peachtree_trajectories.csv")
    sig = ngsim.load_signals(f"{RAW}/signal_change")
    stops = ngsim.fill_missing_stop_lines(ngsim.estimate_stop_lines(t, sig))
    stops.to_csv(f"{out}/stop_lines.csv", index=False)

    # --- alignment diagnostics -------------------------------------------------------------
    cr = ngsim.crossings(t, stops)
    cr = cr[cr["movement"] == 1].copy()
    v = t.set_index(["track", "frame_id"])["v_s"].abs()
    cr["speed"] = v.reindex(list(zip(cr["track"], np.floor(cr["frame"]).astype(int)))).values
    scan = pd.DataFrame([dict(k_frames=k, red_frac_all=red_fraction(cr, sig, k),
                              red_frac_fast=red_fraction(cr, sig, k, min_speed=5.0))
                         for k in range(-100, 101, 5)])
    scan.to_csv(f"{out}/alignment_scan.csv", index=False)
    swap_all = red_fraction(cr, sig, 0, swap=True)
    swap_fast = red_fraction(cr, sig, 0, swap=True, min_speed=5.0)
    print(scan.round(3).to_string())
    print(f"period files swapped: red_frac_all={swap_all:.3f} fast={swap_fast:.3f}")

    # --- events + evaluation ------------------------------------------------------------------
    ev, seq, stats = ngsim.extract_events(t, sig, stops, k=k)
    ev.to_csv(f"{out}/events.csv", index=False)
    np.savez_compressed(f"{out}/sequences.npz", seq=seq)
    stats.update(n_events=len(ev), swap_red_frac_all=swap_all, swap_red_frac_fast=swap_fast,
                 per_site=ev.groupby("intersection")["label"].agg(["size", "sum"]).rename(
                     columns={"size": "n", "sum": "n_go"}).to_dict("index"))
    json.dump(stats, open(f"{out}/extraction_stats.json", "w"), indent=1, default=int)
    print(json.dumps(stats, indent=1, default=int))

    onset = int(round(HIST_S / ngsim.DT)) - 1
    for scheme in ("within", "loio"):
        res, preds = evaluate.run(ev, seq, onset, ngsim.DT, scheme)
        res.to_csv(f"{out}/metrics_{scheme}.csv", index=False)
        preds.to_csv(f"{out}/preds_{scheme}.csv", index=False)
        evaluate.bootstrap_vs_baseline(ev, preds).to_csv(f"{out}/bootstrap_{scheme}.csv", index=False)
        print(scheme)
        print(res.round(3).to_string())


if __name__ == "__main__":
    # optional arg: signal offset k in frames (signal frame = trajectory frame + k), default 0
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0)
