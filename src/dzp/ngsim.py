"""NGSIM Peachtree St. (Atlanta) loader + yellow-onset event extraction for the Peachtree NB/SB
through approaches at the 4 signalized intersections.

Verified in the files (see docs/FINDINGS.md):
  * Trajectories (data.transportation.gov 8ect-6jqj, location='peachtree'): 10 Hz, feet / ft/s.
    local_y = longitudinal position of the FRONT centre of the vehicle along Peachtree
    (increasing northbound). direction: 2 = NB, 4 = SB. movement: 1 through, 2 left, 3 right.
    int_id 1,2,3,5 = 10th, 11th, 12th, 14th St (signals 51, 52, 53, 73); 4 = stop-controlled 13th.
  * The two 15-min periods share one file; vehicle_id repeats. period key = global_time - 100*frame_id.
  * Signal-change logs: one row per cycle, frame numbers of begin green / yellow / red per approach.
Assumed / estimated (checked only indirectly):
  * signal-log frame == trajectory frame_id (+ small offset) -- checked by red-crossing rate.
  * period-key -> file pairing (1163018600 -> 12:45-13:00, 1163030300 -> 16:00-16:15) -- same check.
  * stop-line position: not in the data; estimated from where stopped queue leaders' fronts are.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

FT = 0.3048
DT = 0.1
INTERSECTIONS = {1: "10th", 2: "11th", 3: "12th", 5: "14th"}
DIRS = {2: "NB", 4: "SB"}
PERIOD_FILES = {1163018600: "1245-0100", 1163030300: "0400-0415"}


def load_tracks(path: str) -> pd.DataFrame:
    cols = ["vehicle_id", "frame_id", "global_time", "local_x", "local_y", "v_length", "v_class",
            "v_vel", "lane_id", "int_id", "section_id", "direction", "movement", "preceding"]
    t = pd.read_csv(path, usecols=cols)
    t["period"] = (t["global_time"] - 100 * t["frame_id"]).map(PERIOD_FILES)
    if t["period"].isna().any():
        raise ValueError("unknown period key in NGSIM file")
    t = t.sort_values(["period", "vehicle_id", "frame_id"]).reset_index(drop=True)
    t["track"] = t.groupby(["period", "vehicle_id"]).ngroup()
    t["y_m"] = t["local_y"] * FT
    t["len_m"] = t["v_length"] * FT
    # NGSIM speeds/accels are noisy (accel clipped at +-12 ft/s^2 in the file): recompute from
    # Savitzky-Golay-smoothed positions (window 1.1 s, cubic).
    t["v_s"] = np.nan
    t["a_s"] = np.nan
    for _, idx in t.groupby("track").indices.items():
        y = t["y_m"].values[idx]
        if len(y) >= 11:
            t.loc[t.index[idx], "v_s"] = savgol_filter(y, 11, 3, deriv=1, delta=DT)
            t.loc[t.index[idx], "a_s"] = savgol_filter(y, 11, 3, deriv=2, delta=DT)
        else:
            t.loc[t.index[idx], "v_s"] = np.gradient(y, DT) if len(y) > 1 else 0.0
            t.loc[t.index[idx], "a_s"] = 0.0
    return t


def load_signals(sig_dir: str) -> pd.DataFrame:
    """All Thru-phase cycles: period, int_id, dir, f_green, f_yellow, f_red, next_green."""
    rows = []
    inv_int = {v: k for k, v in INTERSECTIONS.items()}
    for f in sorted(glob.glob(os.path.join(sig_dir, "Peachtree_*.txt"))):
        _, street, d, per = os.path.basename(f)[:-4].split("_")
        if d not in ("NB", "SB"):
            continue
        df = pd.read_csv(f, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        g, y, r = df["BG_Thru"].values, df["BY_Thru"].values, df["BR_Thru"].values
        for i in range(len(df)):
            nxt = g[i + 1] if i + 1 < len(df) else np.nan
            rows.append(dict(period=per, int_id=inv_int[street], dir=d, f_green=g[i], f_yellow=y[i],
                             f_red=r[i], next_green=nxt))
    s = pd.DataFrame(rows).dropna(subset=["f_yellow", "f_red"])
    s["yellow_s"] = (s["f_red"] - s["f_yellow"]) * DT
    return s.reset_index(drop=True)


def signed_s(t: pd.DataFrame, d: str) -> np.ndarray:
    """Travel coordinate (m), increasing in the direction of travel."""
    return t["y_m"].values * (1 if d == "NB" else -1)


def estimate_stop_lines(t: pd.DataFrame, sig: pd.DataFrame, q: float = 0.95) -> pd.DataFrame:
    """Stop line ~ the most downstream position (q-quantile) at which through vehicles are
    standing still (v < 0.3 m/s) during red, while still outside the intersection box
    (int_id == 0) and within 60 m upstream of the box."""
    rows = []
    for (i, d), sg in sig.groupby(["int_id", "dir"]):
        dn = 2 if d == "NB" else 4
        box = t[(t["int_id"] == i) & (t["direction"] == dn)]
        if box.empty:
            continue
        edge = np.quantile(signed_s(box, d), 0.02)  # upstream edge of the box in travel coords
        cand = t[(t["direction"] == dn) & (t["int_id"] == 0) & (t["movement"] == 1)
                 & (t["v_s"].abs() < 0.3)]
        s = signed_s(cand, d)
        cand = cand[(s > edge - 60) & (s < edge + 5)]
        red = np.zeros(len(cand), bool)
        for per, sp in sg.groupby("period"):
            m = (cand["period"] == per).values
            fr = cand["frame_id"].values[m]
            k = np.searchsorted(sp["f_red"].values, fr, side="right") - 1
            ok = k >= 0
            ng = sp["next_green"].values[np.clip(k, 0, None)]
            red[np.where(m)[0][ok]] = fr[ok] < np.nan_to_num(ng[ok], nan=np.inf)
        s = signed_s(cand[red], d)
        rows.append(dict(int_id=i, dir=d, box_edge=edge, stop_s=np.quantile(s, q) if len(s) >= 20 else np.nan,
                         n_stopped=len(s)))
    return pd.DataFrame(rows)


def fill_missing_stop_lines(stops: pd.DataFrame) -> pd.DataFrame:
    """Approaches with too few stopped vehicles: box edge + mean (stop - edge) of the others."""
    s = stops.copy()
    off = (s["stop_s"] - s["box_edge"]).mean()
    s["estimated_from"] = np.where(s["stop_s"].isna(), "box_edge+mean_offset", "queue_front_q95")
    s["stop_s"] = s["stop_s"].fillna(s["box_edge"] + off)
    return s


def _through_tracks(t: pd.DataFrame, dn: int) -> pd.DataFrame:
    return t[(t["direction"] == dn) & (t["v_class"] != 1)]  # drop motorcycles (class 1)


def crossings(t: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """Frame (fractional) at which each vehicle's front crosses each stop line, + its movement."""
    out = []
    for _, st in stops.iterrows():
        dn = 2 if st["dir"] == "NB" else 4
        tt = _through_tracks(t, dn)
        d = st["stop_s"] - signed_s(tt, st["dir"])
        same = np.r_[False, tt["track"].values[1:] == tt["track"].values[:-1]]
        pd_ = np.r_[np.nan, d[:-1]]
        hit = same & (pd_ > 0) & (d <= 0)
        frac = pd_[hit] / (pd_[hit] - d[hit])
        out.append(pd.DataFrame(dict(track=tt["track"].values[hit], period=tt["period"].values[hit],
                                     frame=tt["frame_id"].values[hit] - 1 + frac, int_id=st["int_id"],
                                     dir=st["dir"], movement=tt["movement"].values[hit],
                                     lane_id=tt["lane_id"].values[hit])))
    return pd.concat(out, ignore_index=True)


def signal_state(sig: pd.DataFrame, period: str, int_id: int, d: str, frames: np.ndarray, k: float = 0.0):
    """'G'/'Y'/'R'/'?' of the through phase at trajectory frames (signal frame = frame + k)."""
    sp = sig[(sig["period"] == period) & (sig["int_id"] == int_id) & (sig["dir"] == d)].sort_values("f_green")
    f = frames + k
    out = np.full(len(f), "?", dtype="<U1")
    for _, c in sp.iterrows():
        out[(f >= c["f_green"]) & (f < c["f_yellow"])] = "G"
        out[(f >= c["f_yellow"]) & (f < c["f_red"])] = "Y"
        if np.isfinite(c["next_green"]):
            out[(f >= c["f_red"]) & (f < c["next_green"])] = "R"
    return out


def extract_events(t: pd.DataFrame, sig: pd.DataFrame, stops: pd.DataFrame, k: float = 0.0):
    """Same event definition / sequence format as dzp.events.extract_events (SinD).

    Candidates: Peachtree NB/SB, through movement, regular lane (lane_id 1-9, no turn bays),
    not motorcycle, 0 < d <= D_MAX, v >= V_MIN at the first frame at/after yellow onset.
    """
    from .events import D_MAX, HIST_S, POST_S, V_MIN, _window, label_event

    hist, post = int(round(HIST_S / DT)), int(round(POST_S / DT))
    cr = crossings(t, stops)
    first_cross = cr.groupby(["track", "int_id", "dir"])["frame"].apply(np.sort).to_dict()
    groups = t.groupby("track").indices
    rows, seqs = [], []
    stats = {"candidates": 0, "short_history": 0, "unresolved": 0, "not_through_lane": 0}
    for _, st in stops.iterrows():
        i, dname = st["int_id"], st["dir"]
        dn = 2 if dname == "NB" else 4
        for _, c in sig[(sig["int_id"] == i) & (sig["dir"] == dname)].iterrows():
            f_y, f_r, f_g = c["f_yellow"] - k, c["f_red"] - k, c["next_green"] - k
            k0 = int(np.ceil(f_y))
            snap = t[(t["period"] == c["period"]) & (t["frame_id"] == k0) & (t["direction"] == dn)
                     & (t["v_class"] != 1)]
            d0 = st["stop_s"] - signed_s(snap, dname)
            snap = snap[(d0 > 0) & (d0 <= D_MAX) & (snap["v_s"].abs().values >= V_MIN)]
            for _, r in snap.iterrows():
                stats["candidates"] += 1
                if r["movement"] != 1 or not (1 <= r["lane_id"] <= 9):
                    stats["not_through_lane"] += 1
                    continue
                g = t.iloc[groups[r["track"]]]
                fr = g["frame_id"].values
                assert np.all(np.diff(fr) == 1), "gap in NGSIM track"
                i0 = k0 - fr[0]
                if i0 < hist - 1:
                    stats["short_history"] += 1
                    continue
                d = st["stop_s"] - signed_s(g, dname)
                speed = np.abs(g["v_s"].values)
                acc = g["a_s"].values * (1 if dname == "NB" else -1)
                cs = [x for x in first_cross.get((r["track"], i, dname), []) if x >= f_y - 1]
                t_c = cs[0] if cs else np.nan
                res = label_event(fr, d, speed, i0, t_c, f_r, f_g, DT)
                if res is None:
                    stats["unresolved"] += 1
                    continue
                label, phase = res
                t_red = (f_r - fr) * DT
                seqs.append(np.stack([_window(x, i0, hist, post) for x in (d, speed, acc, t_red)], -1))
                rows.append(dict(
                    intersection=INTERSECTIONS[i], period=c["period"], approach=dname,
                    track_id=r["track"], vehicle_id=r["vehicle_id"], f_yellow=f_y, f_red=f_r,
                    yellow_s=c["yellow_s"], k0=k0, post_frames_observed=min(post, len(fr) - 1 - i0),
                    d0=d[i0], v0=speed[i0], a0=acc[i0], lane_id=r["lane_id"],
                    has_leader_ahead=bool(r["preceding"] != 0), label=label, cross_phase=phase,
                    t_cross_s=(t_c - f_y) * DT if np.isfinite(t_c) else np.nan,
                    stop_line_source=st["estimated_from"],
                ))
    seq = np.array(seqs).reshape(-1, hist + post, 4)
    ev = pd.DataFrame(rows)
    ev["cycle_id"] = ev["period"] + ":" + ev["f_yellow"].astype(int).astype(str)
    return ev, seq, stats
