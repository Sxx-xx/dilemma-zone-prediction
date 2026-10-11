"""Waterloo Multi-Agent Traffic Dataset: Intersection (University Ave. & Weber St.) -> sind.Recording.

One SQLite file per recording (intsc_data_<file_id>.db):
  TRAJECTORIES_0<id>: TRACK_ID, X, Y (UTM 17T, m), SPEED (km/h), TAN_ACC (m/s^2), TIME (s into the file,
    29.97 fps), ANGLE (heading, rad; checked against displacement direction)
  TRAFFIC_LIGHTS: event table, one column per movement L_<from>_<to> with G / Y / R, TIME (s, same clock)
  TRAFFIC_REGIONS_DEF: lane geometry. No stop_line type; the per-approach 'entry_boundary' lines named
    l_<from>_* (entry of the lanes into the intersection box) are used as stop lines (see docs/FINDINGS.md).

Resampled to every 3rd video frame (3 / 29.97 s = 0.1001 s = sind.DT) so the SinD pipeline applies unchanged.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3

import numpy as np
import pandas as pd

from .lanelet_map import LaneletMap
from .sind import DT, GREEN, RED, YELLOW, Recording

STATE = {"G": GREEN, "Y": YELLOW, "R": RED}
TYPE_MAP = {"Car": "car", "Medium Vehicle": "truck", "Heavy Vehicle": "truck", "Bus": "bus"}
# entry_boundary segments forming each approach's stop line (names as in TRAFFIC_REGIONS_DEF)
STOP_SEGMENTS = {"N": ["l_n_s_l", "l_n_s_r"], "S": ["l_s_n_l", "l_s_n_r"], "W": ["l_w_e_l", "l_w_e_r"],
                 "E": ["l_e_-1", "l_e_-2"]}
THROUGH_LIGHT = {"N": "L_N_S", "S": "L_S_N", "W": "L_W_E", "E": "L_E_W"}
# TRACKS.TRACK_LENGTH is NULL in every file -> nominal lengths [m] (assumption) for the front-bumper position
NOMINAL_LENGTH = {"Car": 4.5, "Medium Vehicle": 8.0, "Heavy Vehicle": 12.0, "Bus": 12.0}


def _regions(con, prop: str = "entry_boundary") -> dict[str, np.ndarray]:
    """Regions of one REGION_PROPERTY (names repeat across properties, e.g. a center_line and an entry_boundary)."""
    out = {}
    for name, xs, ys in con.execute("select NAME, X_POSITIONS, Y_POSITIONS from TRAFFIC_REGIONS_DEF "
                                    "where REGION_PROPERTY = ?", (prop,)):
        out[name] = np.c_[json.loads(xs), json.loads(ys)]
    return out


def load_map(con) -> LaneletMap:
    """Stop lines only: concatenated entry_boundary segments per approach (outer end points)."""
    reg = _regions(con)
    nodes, ways = {}, {}
    for a, segs in STOP_SEGMENTS.items():
        pts = np.vstack([reg[s] for s in segs])
        # the two segments share a point; keep the two extreme ends
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        nodes[f"{a}0"], nodes[f"{a}1"] = pts[i], pts[j]
        ways[f"stop_{a}"] = {"nodes": [f"{a}0", f"{a}1"], "tags": {"type": "stop_line", "approach": a}}
    return LaneletMap(nodes=nodes, ways=ways, relations={})


def load_recording(path: str, shift_s: float = 0.0) -> Recording:
    con = sqlite3.connect(path)
    fid = int(os.path.basename(path).split("_")[-1][:-3])
    tr = pd.read_sql(f"select TRACK_ID, X, Y, SPEED, TAN_ACC, TIME, ANGLE from TRAJECTORIES_0{fid}", con)
    meta = pd.read_sql("select TRACK_ID, TYPE, TRACK_LENGTH, ENTRY_GATE, EXIT_GATE from TRACKS", con)
    tr = tr.merge(meta, on="TRACK_ID")
    tr = tr[tr["TYPE"].isin(TYPE_MAP)].copy()
    tr["frame_id"] = np.round(tr["TIME"] / DT).astype(int)
    tr = tr.sort_values(["TRACK_ID", "TIME"]).drop_duplicates(["TRACK_ID", "frame_id"])
    v = tr["SPEED"].values / 3.6
    t = pd.DataFrame({"track_id": tr["TRACK_ID"].values, "frame_id": tr["frame_id"].values,
                      "x": tr["X"].values.astype(float), "y": tr["Y"].values.astype(float),
                      "vx": v * np.cos(tr["ANGLE"].values), "vy": v * np.sin(tr["ANGLE"].values),
                      "heading_rad": tr["ANGLE"].values.astype(float), "a_lon": tr["TAN_ACC"].values.astype(float),
                      "length": tr["TYPE"].map(NOMINAL_LENGTH).values, "agent_type": tr["TYPE"].map(TYPE_MAP).values,
                      "entry_gate": tr["ENTRY_GATE"].values, "exit_gate": tr["EXIT_GATE"].values})
    # keep tracks that are contiguous on the 0.1001 s grid (drop the rare ones with gaps)
    gap = t.groupby("track_id")["frame_id"].apply(lambda f: (np.diff(f.values) != 1).any())
    t = t[~t["track_id"].isin(gap[gap].index)].reset_index(drop=True)
    lt = pd.read_sql("select * from TRAFFIC_LIGHTS order by TIME", con)
    cols = [c for c in lt.columns if c.startswith("L_")]
    lights = pd.DataFrame({"frame": (lt["TIME"].values.astype(float) + shift_s) / DT,
                           **{c: lt[c].map(STATE).values for c in cols}})
    rec = Recording(name=f"waterloo_{fid}", intersection="waterloo", tracks=t, lights=lights, light_cols=cols,
                    map=load_map(con))
    con.close()
    return rec


def files(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "intsc_data_*.db")))
