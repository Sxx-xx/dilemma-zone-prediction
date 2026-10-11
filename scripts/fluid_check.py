"""FLUID (Xuancheng): per-video signal check (red-crossing invariant) and yellow-onset events with the SinD
pipeline (dzp.events.extract_events, map stop lines), on the approaches that pass scripts/fluid_visibility.py.

Steps
  1. light group per approach: the vehicle light group that is least often red when the approach's through
     vehicles cross (matrix written out; `direction` in the signal CSV is ambiguous in the README).
  2. per video: red-crossing rate of through vehicles on the passing approaches at offset 0;
     videos above RED_MAX are excluded as "signal record suspect".
  3. events on the remaining videos, passing approaches only.
Outputs (gitignored): data/raw/fluid/results/{light_matrix,video_check,events}.csv
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import events, fluid, sind

BASE = "data/raw/fluid"
OUT = f"{BASE}/results"
RED_MAX = 0.08
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def passing_approaches():
    v = pd.read_csv(f"{OUT}/visibility.csv")
    v = v[v["passes"] & (v["through_crossings"] >= 30)]
    return {s: set(g["approach"]) for s, g in v.groupby("site")}


def through_crossings(rec):
    aps, cr = events.approach_crossings(rec)
    ent = rec.tracks.groupby("track_id")[["entry_direction", "exit_direction"]].first()
    cr = cr.join(ent, on="track_id")
    cr["approach"] = [aps[s].name() for s in cr["sid"]]
    cr["through"] = cr["exit_direction"] == cr["entry_direction"].map(OPP)
    return aps, cr[~cr["right_turn"]]


def state_at(rec, col, frames):
    idx = np.searchsorted(rec.lights["frame"].values, frames, side="right") - 1
    return np.where(idx < 0, -1, rec.lights[col].values[np.clip(idx, 0, None)])


def main():
    keep = passing_approaches()
    mat, vids, evs = [], [], []
    for site, aps_ok in keep.items():
        root = f"{BASE}/{site}/derived_data"
        recs = {v: fluid.load_recording(root, site, v) for v in fluid.videos(root)}
        crs = {v: through_crossings(r) for v, r in recs.items()}
        # 1) light matrix over all videos of the site (through crossings of the passing approaches)
        rows = []
        for v, (aps, cr) in crs.items():
            for ap, g in cr[cr["through"] & cr["approach"].isin(aps_ok)].groupby("approach"):
                for c in recs[v].light_cols:
                    st = state_at(recs[v], c, g["frame"].values)
                    ok = st >= 0
                    rows.append(dict(site=site, approach=ap, light=c, n=int(ok.sum()), red=int(((st == sind.RED) & ok).sum())))
        m = pd.DataFrame(rows).groupby(["site", "approach", "light"])[["n", "red"]].sum().reset_index()
        m["red_rate"] = m["red"] / m["n"].clip(lower=1)
        mat.append(m)
        # least-red light per approach; ties (groups with identical timing) broken in favour of the through '_s' light
        m["is_thru"] = m["light"].str.endswith("_s")
        best = m.sort_values(["approach", "red_rate", "is_thru"], ascending=[True, True, False]).groupby("approach").head(1)
        lmap_site = best.set_index("approach")["light"].to_dict()
        # 2) per-video red-crossing rate at offset 0
        for v, (aps, cr) in crs.items():
            g = cr[cr["through"] & cr["approach"].isin(aps_ok)]
            n = red = 0
            for ap, gg in g.groupby("approach"):
                st = state_at(recs[v], lmap_site[ap], gg["frame"].values)
                ok = st >= 0
                n += ok.sum()
                red += ((st == sind.RED) & ok).sum()
            rate = red / max(n, 1)
            vids.append(dict(site=site, video=v, through_cross=int(n), red_rate=round(rate, 3),
                             excluded=bool(rate > RED_MAX), lights=str(lmap_site)))
            if rate > RED_MAX:
                continue
            # 3) events, documented-by-data light map, passing approaches only
            doc = {a.sid: lmap_site[a.name()] for a in aps if a.name() in lmap_site}
            infer, events.infer_light_map = events.infer_light_map, (lambda _r, _c: doc)
            try:
                ev, seq, stats = events.extract_events(recs[v])
            finally:
                events.infer_light_map = infer
            if len(ev):
                ent = recs[v].tracks.groupby("track_id")[["entry_direction", "exit_direction"]].first()
                ev = ev.join(ent, on="track_id")
                ev["through"] = ev["exit_direction"] == ev["entry_direction"].map(OPP)
                evs.append(ev.assign(site=site))
            vids[-1].update({f"n_{k}": int(x) for k, x in stats.items() if k != "light_map"})
    pd.concat(mat).to_csv(f"{OUT}/light_matrix.csv", index=False)
    vc = pd.DataFrame(vids)
    vc.to_csv(f"{OUT}/video_check.csv", index=False)
    ev = pd.concat(evs, ignore_index=True)
    ev.to_csv(f"{OUT}/events.csv", index=False)
    print(pd.concat(mat).round(3).to_string())
    print(vc.to_string())
    tab = ev.groupby(["site", "recording", "approach"]).agg(n=("label", "size"), go=("label", "sum"),
                                                          through_n=("through", "sum"))
    tab["stop"] = tab["n"] - tab["go"]
    print(tab.to_string())
    print(ev.groupby("site").agg(n=("label", "size"), go=("label", "sum"), through_n=("through", "sum")).to_string())


if __name__ == "__main__":
    main()
