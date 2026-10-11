"""Yellow-onset events: which vehicles are approaching a stop line when their light turns yellow,
and did they stop or go?

Definitions (see docs/notes/01_sind_loader_labeling.md for the reasoning):
  * Candidate at yellow onset t0 for approach A controlled by light L:
      motor vehicle (car/truck/bus), front bumper upstream of A's stop line with
      0 < d_front <= D_MAX, laterally within the stop-line extent (+margin), moving toward the
      intersection at >= V_MIN, not a right-turner (right turns on red are legal in China),
      and observed for at least HIST_S seconds before t0 (so every model sees the same samples).
  * Label (go = 1, stop = 0), shared with NGSIM via label_event():
      go   : front crosses the stop line after t0 without having come to a halt
             (speed < V_STOP) upstream first, at the latest GO_GRACE_S after red onset.
      stop : comes to a halt upstream before crossing / before next green.
      otherwise (track ends unresolved, or a late crossing without halting) -> dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import sind
from .geometry import Approach, build_approaches, find_crossings, project, turn_angle

D_MAX = 80.0  # m
V_MIN = 2.0  # m/s, below this at onset the vehicle is not "deciding"
V_STOP = 1.0  # m/s, "came to a halt"
GO_GRACE_S = 3.0  # s after red onset by which a "go" vehicle must have crossed
LAT_MARGIN = 1.5  # m
RIGHT_TURN_DEG = -45.0
HIST_S = 2.0  # s of history required before onset
POST_S = 1.0  # s after onset kept in the sequence (evaluation times 0 / 0.5 / 1 s)
SEQ_CHANNELS = ("d", "v", "a", "t_red")  # front distance, speed, long. accel, time to red (SPaT)


def approach_crossings(rec: sind.Recording):
    t = rec.tracks[rec.tracks["agent_type"].isin(sind.MOTOR_VEHICLES)]
    aps = build_approaches(rec.map.stop_lines())
    ta = turn_angle(t)
    cr = pd.concat([find_crossings(t, a, LAT_MARGIN) for a in aps], ignore_index=True)
    cr["turn_deg"] = cr["track_id"].map(ta)
    cr["right_turn"] = cr["turn_deg"] < RIGHT_TURN_DEG
    return aps, cr


def infer_light_map(rec: sind.Recording, cr: pd.DataFrame) -> dict[int, str]:
    """Approach -> controlling light: the light that is least often red when vehicles cross."""
    out = {}
    cr = cr[~cr["right_turn"]]
    for sid, g in cr.groupby("sid"):
        scores = []
        for c in rec.light_cols:
            st = rec.light_state_at(c, g["frame"].values)
            v = st >= 0
            scores.append(((st == sind.RED) & v).sum() / max(v.sum(), 1))
        out[sid] = rec.light_cols[int(np.argmin(scores))]
    return out


def yellow_intervals(rec: sind.Recording, col: str) -> pd.DataFrame:
    """Yellow phases of one light: onset, red onset, next green onset (track-frame units)."""
    ev = rec.lights[["frame", col]].rename(columns={col: "s"})
    ev = ev[ev["s"] != ev["s"].shift()]  # keep only real changes
    fr, s = ev["frame"].values, ev["s"].values
    rows = []
    for i in range(1, len(s)):
        if s[i] == sind.YELLOW and s[i - 1] == sind.GREEN:
            red = fr[i + 1] if i + 1 < len(s) else np.nan
            nxt = [fr[j] for j in range(i + 1, len(s)) if s[j] == sind.GREEN]
            rows.append(dict(f_yellow=fr[i], f_red=red, f_green=nxt[0] if nxt else np.nan))
    df = pd.DataFrame(rows)
    df["yellow_s"] = (df["f_red"] - df["f_yellow"]) * sind.DT
    return df


def _track_arrays(tracks: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    cols = ["frame_id", "x", "y", "vx", "vy", "a_lon", "heading_rad", "length"]
    return {tid: {c: g[c].values for c in cols} for tid, g in tracks.groupby("track_id")}


def label_event(frames, d, speed, i0, t_c, f_red, f_green, dt):
    """Shared stop/go labeling (SinD and NGSIM). Frames are in track-frame units.

    go (1)  : front crosses the stop line (t_c) without first halting upstream, and no later
              than GO_GRACE_S after red onset (later "crossings" without a halt are creeping /
              misassigned vehicles, not yellow-onset decisions).
    stop (0): halts (speed < V_STOP) upstream after onset, before crossing / next green.
    None    : unresolved (track ends, or crosses late without halting) -> dropped.
    """
    horizon = f_green if np.isfinite(f_green) else frames[-1]
    upto = (t_c if np.isfinite(t_c) else min(horizon, frames[-1])) - frames[0]
    seg = slice(i0, int(np.floor(upto)) + 1)
    halted = bool(np.any((speed[seg] < V_STOP) & (d[seg] > 0)))
    if halted:
        return 0, "halt"
    if np.isfinite(t_c) and t_c <= f_red + GO_GRACE_S / dt:
        return 1, ("yellow" if t_c < f_red else "red")
    return None


def _window(arr: np.ndarray, i0: int, hist: int, post: int) -> np.ndarray:
    """arr[i0-hist+1 : i0+post+1], edge-padded at the end if the track stops early."""
    w = arr[i0 - hist + 1 : i0 + post + 1]
    return np.pad(w, (0, hist + post - len(w)), mode="edge")


def extract_events(rec: sind.Recording) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """One row per (yellow onset, approaching vehicle) with label and bookkeeping, plus
    sequences of shape (N, hist + post, len(SEQ_CHANNELS)); onset is at index hist - 1."""
    aps, cr = approach_crossings(rec)
    lmap = infer_light_map(rec, cr)
    mv = rec.tracks[rec.tracks["agent_type"].isin(sind.MOTOR_VEHICLES)]
    right = set(cr.loc[cr["right_turn"], "track_id"])
    ta = turn_angle(mv)
    arrs = _track_arrays(mv)
    first_cross = cr[~cr["right_turn"]].groupby(["track_id", "sid"])["frame"].apply(np.sort).to_dict()
    hist = int(round(HIST_S / sind.DT))
    post = int(round(POST_S / sind.DT))
    seqs = []
    rows, stats = [], {"candidates": 0, "unresolved": 0, "right_turn": 0, "short_history": 0}
    for sid, col in lmap.items():
        ap = aps[sid]
        for _, y in yellow_intervals(rec, col).iterrows():
            k0 = int(np.ceil(y["f_yellow"]))  # first track frame at/after onset
            for tid, a in arrs.items():
                fr = a["frame_id"]
                if not (fr[0] <= k0 <= fr[-1]):
                    continue
                i0 = k0 - fr[0]
                fx = a["x"] + 0.5 * a["length"] * np.cos(a["heading_rad"])
                fy = a["y"] + 0.5 * a["length"] * np.sin(a["heading_rad"])
                d, lat = project(ap, fx, fy)
                vn = a["vx"] * ap.n[0] + a["vy"] * ap.n[1]
                if not (0 < d[i0] <= D_MAX and abs(lat[i0]) <= ap.half_len + LAT_MARGIN and vn[i0] >= V_MIN):
                    continue
                stats["candidates"] += 1
                if tid in right or ta.get(tid, 0) < RIGHT_TURN_DEG:
                    stats["right_turn"] += 1
                    continue
                if i0 < hist - 1:
                    stats["short_history"] += 1
                    continue
                speed = np.hypot(a["vx"], a["vy"])
                cross = [c for c in first_cross.get((tid, sid), []) if c >= y["f_yellow"] - 1]
                t_c = cross[0] if cross else np.nan
                res = label_event(fr, d, speed, i0, t_c, y["f_red"], y["f_green"], sind.DT)
                if res is None:
                    stats["unresolved"] += 1
                    continue
                label, phase = res
                t_red = (y["f_red"] - fr) * sind.DT
                seqs.append(np.stack([_window(x, i0, hist, post) for x in (d, speed, a["a_lon"], t_red)], -1))
                rows.append(dict(
                    intersection=rec.intersection, recording=rec.name, sid=sid, approach=ap.name(),
                    light=col, track_id=tid, f_yellow=y["f_yellow"], f_red=y["f_red"],
                    yellow_s=y["yellow_s"], k0=k0, post_frames_observed=min(post, len(fr) - 1 - i0), d0=d[i0], v0=speed[i0], a0=a["a_lon"][i0],
                    label=label, cross_phase=phase,
                    t_cross_s=(t_c - y["f_yellow"]) * sind.DT if np.isfinite(t_c) else np.nan,
                ))
    seq = np.array(seqs).reshape(-1, hist + post, len(SEQ_CHANNELS))
    return pd.DataFrame(rows), seq, {"light_map": lmap, **stats}
