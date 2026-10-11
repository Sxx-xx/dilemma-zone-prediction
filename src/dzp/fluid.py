"""FLUID (Xuancheng, China; Scientific Data 2026, figshare 10.6084/m9.figshare.29974954) -> sind.Recording.

Per video: derived_data/traj/<video>_Traj.csv (10 FPS, time = frame / 10 s on the video clock, world metres),
derived_data/signal/<video>_signal.csv (state G / y / r per approach direction and turn, begin/end in whole
seconds on the same video clock), derived_data/map/<site>.osm (Lanelet2-like, stop_line ways; nodes in lat/lon
of a local transverse Mercator with lon_0 = lat_0 = 0, README).

Mapped onto the SinD pipeline (dzp.events.extract_events): one light column per (direction, turn) signal group,
codes as SinD (0 red / 1 green / 3 yellow), light-event frames = seconds * 10 (track-frame units).
"""
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyproj

from .lanelet_map import LaneletMap
from .sind import GREEN, RED, YELLOW, Recording

FPS = 10.0
STATE = {"G": GREEN, "y": YELLOW, "r": RED}
TYPE_MAP = {"car": "car", "van": "car", "truck": "truck", "trailer": "truck", "bus": "bus"}  # others: not motor vehicles here
TMERC = "+proj=tmerc +lon_0=0 +lat_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"


def load_map(path: str) -> LaneletMap:
    root = ET.parse(path).getroot()
    p = pyproj.Proj(TMERC)
    nodes = {n.get("id"): np.array(p(float(n.get("lon")), float(n.get("lat")))) for n in root.findall("node")}
    ways = {w.get("id"): {"nodes": [nd.get("ref") for nd in w.findall("nd")],
                          "tags": {t.get("k"): t.get("v") for t in w.findall("tag")}} for w in root.findall("way")}
    return LaneletMap(nodes=nodes, ways=ways, relations={})


def load_signal(path: str, shift_s: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
    """Event table frame -> light_<dir>_<turn> (vehicle groups only; pedestrian 'p' dropped)."""
    s = pd.read_csv(path)
    s = s[s["turn"] != "p"].copy()
    s["col"] = "light_" + s["direction"] + "_" + s["turn"]
    cols = sorted(s["col"].unique())
    times = np.unique(s["begin_time"].values)
    rows = []
    for tb in times:
        row = {"frame": (tb + shift_s) * FPS}
        for c in cols:
            g = s[(s["col"] == c) & (s["begin_time"] <= tb) & (s["end_time"] > tb)]
            row[c] = STATE[g["state"].iloc[0]] if len(g) else -1
        rows.append(row)
    return pd.DataFrame(rows), cols


def load_tracks(path: str) -> pd.DataFrame:
    t = pd.read_csv(path, usecols=["frame", "id", "type", "time", "cx_m", "cy_m", "vx", "vy", "yaw", "length_med",
                                   "speed_smooth", "entry_direction", "exit_direction"])
    t = t[t["type"].isin(TYPE_MAP)].copy()
    t["agent_type"] = t["type"].map(TYPE_MAP)
    t = t.rename(columns={"id": "track_id", "frame": "frame_id", "cx_m": "x", "cy_m": "y", "yaw": "heading_rad",
                          "length_med": "length"})
    t = t.sort_values(["track_id", "frame_id"]).reset_index(drop=True)
    # longitudinal acceleration from the smoothed speed (central difference within each track)
    v = t["speed_smooth"].values
    same_prev = np.r_[False, t["track_id"].values[1:] == t["track_id"].values[:-1]]
    same_next = np.r_[t["track_id"].values[:-1] == t["track_id"].values[1:], False]
    vp = np.where(same_prev, np.r_[np.nan, v[:-1]], v)
    vn = np.where(same_next, np.r_[v[1:], np.nan], v)
    span = (same_prev.astype(int) + same_next.astype(int)) / FPS
    t["a_lon"] = np.where(span > 0, (vn - vp) / np.where(span > 0, span, 1), 0.0)
    return t


def load_recording(root: str, site: str, video: str, shift_s: float = 0.0) -> Recording:
    lights, cols = load_signal(os.path.join(root, "signal", f"{video}_signal.csv"), shift_s)
    return Recording(name=video, intersection=site, tracks=load_tracks(os.path.join(root, "traj", f"{video}_Traj.csv")),
                     lights=lights, light_cols=cols, map=load_map(os.path.join(root, "map", f"{site}.osm")))


def videos(root: str) -> list[str]:
    return sorted(os.path.basename(p)[: -len("_signal.csv")] for p in glob.glob(os.path.join(root, "signal", "*_signal.csv")))
