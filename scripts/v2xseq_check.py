"""V2X-Seq TFD single-infrastructure: screening and yellow-onset event counts per intersection (no evaluation).

Per segment (dzp.v2xseq, overlapping scenes stitched): approaches from the through connector lanes, crossings,
red-crossing rate with the recorded light, visibility (as scripts/fluid_visibility.py: 99th pct of the upstream
distance of through vehicles moving toward the line; v_app = mean crossing speed of 'free' through vehicles),
yellow-onset events (dzp.events.extract_events with the data-given light map), onset features incl. the same-lane
leader (dzp.features.build, lead 0 -> leader/follower), movement by turn angle (run_benchmark.movement).

Outputs (--out, gitignored under data/raw/v2xseq/results/check):
  segments.csv, approaches.csv (per segment x approach), events_through.csv (per event, onset features),
  onsets.csv (through-approach yellow onsets), stop_positions.csv (halt distance sample), per_intersection.csv

  nice -n 19 .venv/bin/python scripts/v2xseq_check.py --root data/raw/v2xseq/full/extracted/V2X-Seq-TFD \
      --out data/raw/v2xseq/results/check --workers 4 [--shift-table results/v2xseq/stopline_shift.csv]
"""
import argparse
import json
import os
import sys
import traceback
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import events, features, sind, v2xseq
from dzp.geometry import build_approaches, project, turn_angle
from run_benchmark import movement

RED_MAX = 0.08
MIN_CROSS = 30
DZ_LO, DZ_HI = 2.5, 5.5
_G = {}


def _init(root, cache, nice, shift_path):
    os.nice(nice)
    _G["root"], _G["lanes"], _G["shift"] = root, v2xseq.lane_index(root, cache), v2xseq.load_shift_table(shift_path)


def one(seg):
    try:
        return _one(seg)
    except Exception:
        return dict(error=f"{seg.name}: {traceback.format_exc()}")


def _one(seg):
    root, lanes = _G["root"], _G["lanes"]
    rec = v2xseq.load_recording(root, seg, lanes, shift_table=_G["shift"])
    t = rec.tracks
    out = dict(segment=dict(name=seg.name, inter=seg.inter, n_scenes=len(seg.scenes), dur_s=seg.t1 - seg.t0,
                            n_tracks=int(t["track_id"].nunique()), n_rows=len(t), n_approaches=len(rec.light_cols),
                            notes="; ".join(rec.notes)), approaches=[], events=None, onsets=[], stops=[], vis=[])
    if not rec.light_cols or t.empty:
        return out
    aps, cr = events.approach_crossings(rec)
    ta = turn_angle(t)
    thr_ids = set(ta.index[np.abs(ta) < 30])
    cr = cr.join(ta.rename("ta"), on="track_id")
    fx = t["x"].values + 0.5 * t["length"].values * np.cos(t["heading_rad"].values)
    fy = t["y"].values + 0.5 * t["length"].values * np.sin(t["heading_rad"].values)
    speed = np.hypot(t["vx"].values, t["vy"].values)
    rng = np.random.default_rng(abs(hash(seg.name)) % 2**32)
    for a in aps:
        col = rec.light_map[a.sid]
        c = cr[(cr["sid"] == a.sid) & ~cr["right_turn"]]
        cth = c[c["track_id"].isin(thr_ids)]
        st = rec.light_state_at(col, cth["frame"].values)
        ok = st >= 0
        d, lat = project(a, fx, fy)
        vn = t["vx"].values * a.n[0] + t["vy"].values * a.n[1]
        on = t["track_id"].isin(cth["track_id"]).values & (np.abs(lat) <= a.half_len + 1.5) & (d > 0)
        dd = d[on & (vn > 1)]
        up = pd.DataFrame({"tid": t["track_id"].values[on], "v": speed[on]}).groupby("tid")["v"].min()
        free = cth[cth["track_id"].isin(set(up.index[up >= 2.0]))]["speed"].values
        # halt positions around the line, -15..30 m (stop-line placement sanity: leading halts should sit at d ~ 1-3 m)
        halt = t["track_id"].isin(cth["track_id"]).values & (np.abs(lat) <= a.half_len + 1.5) & (speed < 1.0) & (d < 30) & (d > -15)
        hp = pd.DataFrame({"tid": t["track_id"].values[halt], "d": d[halt]}).groupby("tid")["d"].min().values
        out["approaches"].append(dict(segment=seg.name, inter=seg.inter, sid=a.sid, name=a.name(), light=col,
                                      lanes=len(rec.approach_lanes[a.sid]["lanes"]), half_len=a.half_len, mid_x=a.mid[0], mid_y=a.mid[1],
                                      heading_deg=a.heading_deg, cross_all=len(c), cross_through=len(cth),
                                      cross_red=int(((st == sind.RED) & ok).sum()), cross_known=int(ok.sum()),
                                      cross_red_fast=int(((st == sind.RED) & ok & (cth["speed"].values >= 3.0)).sum()),
                                      vis_max=float(dd.max()) if len(dd) else np.nan,
                                      n_free=len(free), v_free_sum=float(free.sum()), n_halt=len(hp)))
        out["vis"].append(dict(segment=seg.name, inter=seg.inter, sid=a.sid,
                               d=rng.choice(dd, min(len(dd), 200), replace=False).tolist() if len(dd) else [],
                               v=rng.choice(free, min(len(free), 50), replace=False).tolist() if len(free) else []))
        out["stops"].append(dict(segment=seg.name, inter=seg.inter, sid=a.sid, d=np.round(hp, 2).tolist()))
        y = events.yellow_intervals(rec, col)
        for r in y.itertuples(index=False):
            out["onsets"].append(dict(segment=seg.name, inter=seg.inter, sid=a.sid, t_abs=seg.t0 + r.f_yellow * v2xseq.DT,
                                      yellow_s=r.yellow_s, t_rel=r.f_yellow * v2xseq.DT, t_left=seg.t1 - seg.t0 - r.f_yellow * v2xseq.DT))
    infer, events.infer_light_map = events.infer_light_map, (lambda _r, _c: rec.light_map)
    try:
        ev, _, stats = events.extract_events(rec)
    finally:
        events.infer_light_map = infer
    out["segment"].update({f"n_{k}": int(v) for k, v in stats.items() if k != "light_map"})
    if len(ev):
        ev["group"], ev["movement"] = seg.name, movement(rec, ev)
        f, _ = features.build(rec, ev, aps, [0.0], dt=v2xseq.DT)
        f = f.drop(columns=["ev", "lead_s"])
        ev = pd.concat([ev.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
        ev["turn_deg"] = ev["track_id"].map(ta)
        out["events"] = ev
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="data/raw/v2xseq/results/cache")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nice", type=int, default=19)
    ap.add_argument("--inter", default="", help="only this intersection")
    ap.add_argument("--max-segments", type=int, default=0)
    ap.add_argument("--shift-table", default="", help="stop-line shift CSV from scripts/v2xseq_calibrate.py (empty = none)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lanes = v2xseq.lane_index(args.root, args.cache)
    segs = v2xseq.segments(args.root, args.cache)
    if args.inter:
        segs = [s for s in segs if s.inter == args.inter]
    if args.max_segments:
        segs = segs[: args.max_segments]
    print(f"segments {len(segs)}, lanes {len(lanes)}", flush=True)
    S, A, E, O, H, V, errs = [], [], [], [], [], [], []
    with Pool(args.workers, initializer=_init, initargs=(args.root, args.cache, args.nice, args.shift_table)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, segs, chunksize=4)):
            if "error" in r:
                errs.append(r["error"])
                continue
            S.append(r["segment"])
            A += r["approaches"]
            O += r["onsets"]
            H += r["stops"]
            V += r["vis"]
            if r["events"] is not None:
                E.append(r["events"])
            if i % 500 == 0:
                print(f"{i}/{len(segs)} events so far {sum(len(e) for e in E)}", flush=True)
    pd.DataFrame(S).to_csv(f"{args.out}/segments.csv", index=False)
    pd.DataFrame(A).to_csv(f"{args.out}/approaches.csv", index=False)
    pd.DataFrame(O).to_csv(f"{args.out}/onsets.csv", index=False)
    pd.DataFrame(H).to_json(f"{args.out}/stop_positions.json", orient="records")
    pd.DataFrame(V).to_json(f"{args.out}/visibility_samples.json", orient="records")
    ev = pd.concat(E, ignore_index=True) if E else pd.DataFrame()
    ev.to_csv(f"{args.out}/events_all.csv", index=False)
    open(f"{args.out}/errors.txt", "w").write("\n".join(errs))
    print(f"done: {len(S)} segments, {len(ev)} events, {len(errs)} errors", flush=True)


if __name__ == "__main__":
    main()
