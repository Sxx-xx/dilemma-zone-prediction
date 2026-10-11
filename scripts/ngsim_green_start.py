"""NGSIM Peachtree: narrow the signal-log <-> trajectory clock offset k using queue leaders that
start moving at green onset.

Convention (same as dzp.ngsim): signal frame = trajectory frame + k.
Green onset in trajectory frames = G - k, so a queue leader with start-up reaction R (s) starts at
    dep = G - k + R   ->   delta0 := dep - G (measured at k = 0) = R - k   ->   k = R - delta0.
* Every leader has R >= 0 (nobody starts before green, except red-jumpers) -> lower bound k >= -delta0.
* A typical R gives a point estimate k ~= R_typ - median(delta0).

Queue leader: through vehicle in a through lane (lane_id 1-9, movement 1, not motorcycle), standing
still (|v| < 0.3 m/s) for the whole 2.5-5.0 s before the logged green, front 0-8 m upstream of the
estimated stop line, and the most downstream stationary vehicle in its lane.
Motion onset t0: first time (linear interpolation between frames) the raw travel position exceeds the
stationary position (median over the pre-window) by THRESH = 0.25 m and stays above it afterwards.
Raw positions, because the Savitzky-Golay speeds are non-causal and pull the onset earlier. (A first
try fitted "constant, then constant acceleration from t0"; NGSIM leaders jump to cruising speed within
1-2 frames, so the parabola extrapolated the onset 1-2 s too early. Not used.)
The threshold makes t0 slightly late: at 1.5 m/s^2 a vehicle needs ~0.6 s to move 0.25 m; with the
abrupt NGSIM starts it is mostly 0.1-0.3 s. This lag is part of the "R" below.

Outputs: results/ngsim_peachtree/green_start/leaders.csv, summary printed.
    .venv/bin/python scripts/ngsim_green_start.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import ngsim

RAW = "data/raw/ngsim/peachtree"
OUT = "results/ngsim_peachtree/green_start"
PRE = (50, 25)          # stationary check window: frames G-50 .. G-25 (5.0-2.5 s before logged green)
FIT = (-50, 70)         # onset fit window: frames G-50 .. G+70


THRESH = 0.25


def onset_time(tt: np.ndarray, s: np.ndarray, base: float):
    """First t where s - base > THRESH and stays above (linear interpolation). nan if never."""
    above = (s - base) > THRESH
    if not above[-1]:
        return np.nan
    j = len(above) - np.argmin(above[::-1])        # first index of the final run of True
    if j == 0:
        return np.nan                              # already moving at the start of the window
    y0, y1 = s[j - 1] - base, s[j] - base
    return tt[j - 1] + (THRESH - y0) / (y1 - y0) * (tt[j] - tt[j - 1])


def main():
    os.makedirs(OUT, exist_ok=True)
    t = ngsim.load_tracks(f"{RAW}/peachtree_trajectories.csv")
    sig = ngsim.load_signals(f"{RAW}/signal_change")
    stops = ngsim.fill_missing_stop_lines(ngsim.estimate_stop_lines(t, sig)).set_index(["int_id", "dir"])
    by_pf = t.groupby(["period", "frame_id"]).indices
    tracks = t.groupby("track").indices

    rows = []
    for _, c in sig.iterrows():
        G = c["next_green"]
        if not np.isfinite(G):
            continue
        G = int(G)
        i, d = c["int_id"], c["dir"]
        if (i, d) not in stops.index:
            continue
        st = stops.loc[(i, d)]
        dn = 2 if d == "NB" else 4
        idx = by_pf.get((c["period"], G - PRE[1]))
        if idx is None:
            continue
        snap = t.iloc[idx]
        snap = snap[(snap["direction"] == dn) & (snap["v_class"] != 1) & (snap["movement"] == 1)
                    & snap["lane_id"].between(1, 9)]
        dist = st["stop_s"] - ngsim.signed_s(snap, d)
        snap = snap.assign(dist=dist)
        for lane, g in snap[(snap["dist"] > -1) & (snap["dist"] < 30)].groupby("lane_id"):
            cand = []
            for _, r in g.iterrows():
                tr = t.iloc[tracks[r["track"]]]
                pre = tr[tr["frame_id"].between(G - PRE[0], G - PRE[1])]
                if len(pre) == PRE[0] - PRE[1] + 1 and (pre["v_s"].abs() < 0.3).all():
                    cand.append((r["dist"], r["track"]))
            if not cand:
                continue
            dist0, trk = min(cand)                              # most downstream stationary vehicle
            if not (0 <= dist0 <= 8):
                continue
            tr = t.iloc[tracks[trk]]
            w = tr[tr["frame_id"].between(G + FIT[0], G + FIT[1])]
            if len(w) != FIT[1] - FIT[0] + 1:
                continue
            s = ngsim.signed_s(w, d)
            tt = (w["frame_id"].values - G) * ngsim.DT          # seconds relative to logged green
            base = np.median(s[: PRE[0] - PRE[1] + 1])          # stationary position (pre-window)
            t0 = onset_time(tt, s, base)
            jitter = np.ptp(s[: PRE[0] - PRE[1] + 1])           # position noise while standing
            rows.append(dict(period=c["period"], intersection=ngsim.INTERSECTIONS[i], dir=d, lane=lane,
                             G=G, track=trk, dist0=dist0, delta0=t0, jitter_m=jitter, moved_m=s[-1] - base))
    lead = pd.DataFrame(rows)
    lead.to_csv(f"{OUT}/leaders.csv", index=False)

    ok = lead[(lead["moved_m"] > 3) & lead["delta0"].notna() & (lead["jitter_m"] < 0.15)]
    print(f"leaders found {len(lead)}, used {len(ok)} (moved > 3 m, onset found, standing jitter < 0.15 m)")
    q = lambda x: pd.Series(dict(n=len(x), min=x.min(), p05=x.quantile(0.05), p25=x.quantile(0.25),
                                 median=x.median(), p75=x.quantile(0.75), max=x.max()))
    print("\ndelta0 = motion onset - logged green (s, k = 0)")
    print(q(ok["delta0"]).round(2).to_string())
    print("\nby approach")
    print(ok.groupby(["intersection", "dir"])["delta0"].apply(q).unstack().round(2).to_string())
    print("\nby period")
    print(ok.groupby("period")["delta0"].apply(q).unstack().round(2).to_string())
    print("\nstanding jitter median %.3f m" % ok["jitter_m"].median())
    for name, g in [("all", ok)] + list(ok.groupby("period")):
        x = g["delta0"]
        print(f"\n[{name}] implied k (s) = R - median(delta0), median = {x.median():.2f}")
        print("  " + "  ".join(f"R={R:.1f}: k={R - x.median():+.2f}" for R in (0.5, 1.0, 1.5, 2.0)))
        print(f"  lower bound (R >= 0): k >= {-x.min():+.2f} (min), {-x.quantile(0.05):+.2f} (p05)")


if __name__ == "__main__":
    main()
