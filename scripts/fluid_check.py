"""FLUID (Xuancheng) TI intersection: signal-trajectory alignment (red-crossing invariant) and yellow-onset
event counts with the SinD pipeline (dzp.events.extract_events, map stop lines).

Outputs results/fluid/: light_matrix.csv, alignment_scan.csv, events_shift*.csv, summary.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import events, fluid, sind

ROOT = "data/raw/fluid/derived_data"
SITE = "TI"
OUT = "results/fluid"
SHIFTS = (0.0, -1.5)   # s added to signal times (best red-crossing offset of videos 1-2 is about -1.5 s)


def crossings(rec):
    aps, cr = events.approach_crossings(rec)
    ent = rec.tracks.groupby("track_id")[["entry_direction", "exit_direction"]].first()
    cr = cr.join(ent, on="track_id")
    cr["approach"] = [aps[s].name() for s in cr["sid"]]
    cr["through_EW"] = cr["entry_direction"].isin(["E", "W"]) & cr["exit_direction"].isin(["E", "W"]) & \
        (cr["entry_direction"] != cr["exit_direction"])
    return cr[~cr["right_turn"]]


def red_rate(rec, cr, col, shift_s):
    f = rec.lights["frame"].values + shift_s * fluid.FPS
    idx = np.searchsorted(f, cr["frame"].values, side="right") - 1
    st = np.where(idx < 0, -1, rec.lights[col].values[np.clip(idx, 0, None)])
    ok = st >= 0
    return ((st == sind.RED) & ok).sum() / max(ok.sum(), 1), int(ok.sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    vids = fluid.videos(ROOT)
    recs = {v: fluid.load_recording(ROOT, SITE, v) for v in vids}
    crs = {v: crossings(r) for v, r in recs.items()}

    # 1) which light group controls which approach / movement (red rate at shift 0)
    rows = []
    for v, cr in crs.items():
        cr = cr.assign(mv=cr["entry_direction"] + "->" + cr["exit_direction"])
        for (ap, mv), g in cr.groupby(["approach", "mv"]):
            for c in recs[v].light_cols:
                r, n = red_rate(recs[v], g, c, 0.0)
                rows.append(dict(video=v, approach=ap, mv=mv, light=c, n=n, red=r * n))
    m = pd.DataFrame(rows).groupby(["approach", "mv", "light"])[["n", "red"]].sum()
    m = (m["red"] / m["n"]).round(3).unstack("light").join(m["n"].groupby(level=[0, 1]).max())
    m[m["n"] >= 10].to_csv(f"{OUT}/light_matrix.csv")

    # 2) offset scan, E-W through traffic vs light_E_s (E_s = W_s = W_l timing)
    scan = []
    for sh in np.round(np.arange(-6, 6.01, 0.5), 1):
        for v, cr in crs.items():
            g = cr[cr["through_EW"]]
            r, n = red_rate(recs[v], g, "light_E_s", sh)
            rf, nf = red_rate(recs[v], g[g["speed"] >= 5], "light_E_s", sh)
            scan.append(dict(shift_s=sh, video=v, n=n, red=r, n_fast=nf, red_fast=rf))
    scan = pd.DataFrame(scan)
    scan.to_csv(f"{OUT}/alignment_scan.csv", index=False)

    # 3) yellow-onset events, SinD pipeline unchanged
    summary = {"light_map_inferred": {}, "events": {}}
    for sh in SHIFTS:
        evs = []
        for v in vids:
            rec = fluid.load_recording(ROOT, SITE, v, shift_s=sh)
            ev, seq, st = events.extract_events(rec)
            ent = rec.tracks.groupby("track_id")[["entry_direction", "exit_direction"]].first()
            ev = ev.join(ent, on="track_id")
            ev["movement"] = ev["entry_direction"] + "->" + ev["exit_direction"]
            evs.append(ev)
            summary["light_map_inferred"][v] = {k: str(c) for k, c in st["light_map"].items()}
            summary["events"].setdefault(f"shift{sh:+.1f}", {})[v] = {k: int(x) for k, x in st.items() if k != "light_map"}
        ev = pd.concat(evs, ignore_index=True)
        ev.to_csv(f"{OUT}/events_shift{sh:+.1f}.csv", index=False)
        tab = ev.groupby(["recording", "approach", "movement"])["label"].agg(n="size", go="sum")
        tab["stop"] = tab["n"] - tab["go"]
        summary["events"][f"shift{sh:+.1f}"]["table"] = {" | ".join(k): v for k, v in tab.to_dict("index").items()}
        summary["events"][f"shift{sh:+.1f}"]["total"] = dict(n=len(ev), stop=int((ev.label == 0).sum()), go=int(ev.label.sum()))
        print(f"shift {sh:+.1f} s\n{tab.to_string()}\ntotal {len(ev)} stop {(ev.label == 0).sum()} go {ev.label.sum()}")
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=int)
    print(m[m["n"] >= 10].to_string())
    print(scan.pivot(index="shift_s", columns="video", values="red").round(3).iloc[::2].to_string())


if __name__ == "__main__":
    main()
