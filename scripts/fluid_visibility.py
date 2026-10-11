"""FLUID: how far upstream of each stop line is visible, and does the 2.5-5.5 s time-to-stop-line band fit in it?

Per site and through approach (map stop_line, vehicles whose exit is opposite to their entry):
  visible_m  = 99th percentile of the front-bumper distance upstream of the stop line of through vehicles moving
               toward it (v_n > 1 m/s, within the stop-line extent + 1.5 m)
  v_app      = mean speed at the stop-line crossing of through vehicles that never drop below 2 m/s in the visible
               upstream section ("free" approaches)
  band       = [2.5, 5.5] s x v_app (m);   pass = visible_m >= 5.5 v_app;   cover = share of the band that is visible
Writes data/raw/fluid/results/visibility.csv (gitignored).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import events, fluid
from dzp.geometry import project

BASE = "data/raw/fluid"
SITES = ("TI", "FI", "FIDRT")
OUT = f"{BASE}/results"
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def site_rows(site):
    root = f"{BASE}/{site}/derived_data"
    d_all, v_all, n_all = {}, {}, {}
    for vid in fluid.videos(root):
        rec = fluid.load_recording(root, site, vid)
        t = rec.tracks
        aps, cr = events.approach_crossings(rec)
        ent = t.groupby("track_id")[["entry_direction", "exit_direction"]].first()
        cr = cr.join(ent, on="track_id")
        thr_ids = set(ent.index[ent["exit_direction"] == ent["entry_direction"].map(OPP)])
        fx = t["x"] + 0.5 * t["length"] * np.cos(t["heading_rad"])
        fy = t["y"] + 0.5 * t["length"] * np.sin(t["heading_rad"])
        speed = np.hypot(t["vx"], t["vy"]).values
        for a in aps:
            c = cr[(cr["sid"] == a.sid) & cr["track_id"].isin(thr_ids)]
            if c.empty:
                continue
            frm = c["entry_direction"].mode().iloc[0]
            key = (frm, a.name())
            d, lat = project(a, fx.values, fy.values)
            vn = t["vx"].values * a.n[0] + t["vy"].values * a.n[1]
            on = t["track_id"].isin(c["track_id"]).values & (np.abs(lat) <= a.half_len + 1.5) & (d > 0)
            d_all.setdefault(key, []).append(d[on & (vn > 1)])
            # free approaches: min speed over the visible upstream part >= 2 m/s
            up = pd.DataFrame({"tid": t["track_id"].values[on], "v": speed[on]}).groupby("tid")["v"].min()
            free = set(up.index[up >= 2.0])
            v_all.setdefault(key, []).append(c[c["track_id"].isin(free)]["speed"].values)
            n_all[key] = n_all.get(key, 0) + len(c)
    rows = []
    for key in d_all:
        d = np.concatenate(d_all[key])
        v = np.concatenate(v_all[key])
        vis = float(np.quantile(d, 0.99)) if len(d) else np.nan
        va = float(v.mean()) if len(v) else np.nan
        lo, hi = 2.5 * va, 5.5 * va
        cover = float(np.clip((vis - lo) / (hi - lo), 0, 1)) if np.isfinite(va) and hi > lo else np.nan
        rows.append(dict(site=site, entry=key[0], approach=key[1], through_crossings=n_all[key], free_n=len(v),
                         visible_m=round(vis, 1), v_app_ms=round(va, 2), band_lo_m=round(lo, 1), band_hi_m=round(hi, 1),
                         band_cover=round(cover, 2), passes=bool(vis >= hi)))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.DataFrame([r for s in SITES for r in site_rows(s)])
    df.to_csv(f"{OUT}/visibility.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
