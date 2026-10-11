"""Waterloo intersection dataset: stop-line check, signal-trajectory alignment (red-crossing invariant) and
yellow-onset event counts with the SinD pipeline (dzp.events.extract_events).

Lights offered to the pipeline = the four through movements (L_N_S, L_S_N, L_W_E, L_E_W). Events use the documented
approach -> light map (L_<from>_<to>); events.infer_light_map is only reported for comparison.
Outputs results/waterloo/: files.csv, alignment_scan.csv, events_shift*.csv, summary.json
"""
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import events, sind, waterloo
from dzp.geometry import project, turn_angle

ROOT = "data/raw/waterloo"
OUT = "results/waterloo"
THROUGH_COLS = list(waterloo.THROUGH_LIGHT.values())


def through_rec(path, shift_s=0.0):
    rec = waterloo.load_recording(path, shift_s)
    rec.light_cols = THROUGH_COLS
    return rec


def red_rate(rec, cr, lmap, shift_s):
    n = red = 0
    for sid, g in cr.groupby("sid"):
        f = rec.lights["frame"].values + shift_s / sind.DT
        idx = np.searchsorted(f, g["frame"].values, side="right") - 1
        st = np.where(idx < 0, -1, rec.lights[lmap[sid]].values[np.clip(idx, 0, None)])
        ok = st >= 0
        n += ok.sum()
        red += ((st == sind.RED) & ok).sum()
    return red / max(n, 1), int(n)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, scan, evs, summary = [], [], [], {"inferred_light_map": {}, "stats": {}}
    for path in waterloo.files(ROOT):
        con = sqlite3.connect(path)
        n_len = con.execute("select count(*), sum(TRACK_LENGTH is null) from TRACKS").fetchone()
        con.close()
        rec = through_rec(path)
        aps, cr = events.approach_crossings(rec)
        ta = turn_angle(rec.tracks)
        cr["turn_deg"] = cr["track_id"].map(ta)
        thr = cr[cr["turn_deg"].abs() < 30]          # through movements only
        lmap = events.infer_light_map(rec, cr)
        doc = {a.sid: waterloo.THROUGH_LIGHT[a.name()[0]] for a in aps}
        summary["inferred_light_map"][rec.name] = {aps[s].name(): c for s, c in lmap.items()}
        # stop line check: stationary fronts relative to the stop line
        t = rec.tracks
        st = t[np.hypot(t["vx"], t["vy"]) < 0.3]
        q = []
        for a in aps:
            fx = st["x"] + 0.5 * st["length"] * np.cos(st["heading_rad"])
            fy = st["y"] + 0.5 * st["length"] * np.sin(st["heading_rad"])
            d, lat = project(a, fx.values, fy.values)
            m = (np.abs(lat) < a.half_len + 1) & (d > -5) & (d < 40)
            if m.sum():
                q.append(np.quantile(d[m], 0.02))
        r0, n0 = red_rate(rec, thr, doc, 0.0)
        rows.append(dict(file=rec.name, tracks=t["track_id"].nunique(), length_null=f"{n_len[1]}/{n_len[0]}",
                         through_cross=n0, red_rate_doc=round(r0, 3), inferred_eq_doc=lmap == doc,
                         stopped_front_q02_m=round(float(np.median(q)), 2) if q else None,
                         duration_s=round(t["frame_id"].max() * sind.DT, 1)))
        for sh in np.round(np.arange(-3, 3.01, 0.1), 1):
            r, n = red_rate(rec, thr, doc, sh)
            scan.append(dict(file=rec.name, shift_s=sh, red=r, n=n))
        # extract with the DOCUMENTED approach -> through-light map (inference is unreliable with ~100 crossings
        # per file: N/S and W/E lights are identical 94% of the time)
        infer, events.infer_light_map = events.infer_light_map, (lambda _rec, _cr: doc)
        try:
            ev, seq, stats = events.extract_events(rec)
        finally:
            events.infer_light_map = infer
        if len(ev):
            ev["turn_deg"] = ev["track_id"].map(ta)
            ev["through"] = ev["turn_deg"].abs() < 30
            ev["light_doc"] = ev["sid"].map(doc)
            evs.append(ev)
        summary["stats"][rec.name] = {k: int(v) for k, v in stats.items() if k != "light_map"}
        print(rows[-1], flush=True)
    files = pd.DataFrame(rows)
    files.to_csv(f"{OUT}/files.csv", index=False)
    scan = pd.DataFrame(scan)
    tot = scan.assign(red_n=scan.red * scan.n).groupby("shift_s")[["red_n", "n"]].sum()
    tot["red"] = tot.red_n / tot.n
    tot.reset_index().to_csv(f"{OUT}/alignment_scan.csv", index=False)
    ev = pd.concat(evs, ignore_index=True)
    ev.to_csv(f"{OUT}/events_shift+0.0.csv", index=False)
    tab = ev.groupby(["approach", "through"])["label"].agg(n="size", go="sum")
    tab["stop"] = tab["n"] - tab["go"]
    summary["events"] = {f"{a} | {'through' if th else 'turn'}": v for (a, th), v in tab.to_dict("index").items()}
    summary["events_total"] = dict(n=len(ev), stop=int((ev.label == 0).sum()), go=int(ev.label.sum()),
                                   through_n=int(ev.through.sum()), through_go=int(ev[ev.through].label.sum()))
    b = tot["red"].idxmin()
    flat = tot[tot["red"] <= tot["red"].min() + 0.01].index
    summary["alignment"] = dict(red_at_0=round(tot.loc[0.0, "red"], 4), best_shift_s=float(b), red_best=round(tot.loc[b, "red"], 4),
                                within_1pp=[float(flat.min()), float(flat.max())])
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
    print(files.to_string())
    print(tab.to_string())
    print(json.dumps({k: summary[k] for k in ("events_total", "alignment")}, indent=1))


if __name__ == "__main__":
    main()
