"""Stop-line geometry: signed distance to a stop line, crossings, approach arms."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lanelet_map import StopLine


@dataclass
class Approach:
    """A stop line oriented so that `n` points from upstream into the intersection."""

    sid: int
    mid: np.ndarray
    t: np.ndarray  # unit vector along the stop line
    n: np.ndarray  # unit normal, upstream -> intersection
    half_len: float
    heading_deg: float  # compass-free label: direction of travel (atan2 of n)

    def name(self) -> str:
        # label by where the approach comes FROM (n points into the intersection)
        a = (self.heading_deg + 360) % 360
        return {0: "W", 1: "S", 2: "E", 3: "N"}[int(((a + 45) % 360) // 90)] + "_approach"


def build_approaches(stop_lines: list[StopLine]) -> list[Approach]:
    center = np.mean([s.mid for s in stop_lines], axis=0)
    out = []
    for i, s in enumerate(stop_lines):
        t = (s.p1 - s.p0) / s.length
        n = np.array([-t[1], t[0]])
        if np.dot(center - s.mid, n) < 0:
            n = -n
        out.append(Approach(i, s.mid, t, n, s.length / 2, float(np.degrees(np.arctan2(n[1], n[0])))))
    return out


def project(ap: Approach, x: np.ndarray, y: np.ndarray):
    """Return (d_center, lateral): d_center > 0 upstream of the stop line [m]."""
    dx, dy = x - ap.mid[0], y - ap.mid[1]
    d = -(dx * ap.n[0] + dy * ap.n[1])
    lat = dx * ap.t[0] + dy * ap.t[1]
    return d, lat


def find_crossings(tracks: pd.DataFrame, ap: Approach, lat_margin: float = 1.5) -> pd.DataFrame:
    """Frames where a vehicle's FRONT bumper crosses the stop line in the travel direction.

    Front position = centre + L/2 along heading; crossing = d_front goes from >0 to <=0 while
    the lateral offset is within the stop-line extent (+margin) and velocity points along n.
    """
    tr = tracks
    fx = tr["x"].values + 0.5 * tr["length"].values * np.cos(tr["heading_rad"].values)
    fy = tr["y"].values + 0.5 * tr["length"].values * np.sin(tr["heading_rad"].values)
    d, lat = project(ap, fx, fy)
    vn = tr["vx"].values * ap.n[0] + tr["vy"].values * ap.n[1]
    same = np.r_[False, tr["track_id"].values[1:] == tr["track_id"].values[:-1]]
    prev_d = np.r_[np.nan, d[:-1]]
    hit = same & (prev_d > 0) & (d <= 0) & (np.abs(lat) <= ap.half_len + lat_margin) & (vn > 0.5)
    # sub-frame interpolation of crossing time
    frac = prev_d[hit] / (prev_d[hit] - d[hit])
    return pd.DataFrame(
        {
            "track_id": tr["track_id"].values[hit],
            "frame": tr["frame_id"].values[hit] - 1 + frac,
            "sid": ap.sid,
            "speed": np.hypot(tr["vx"].values[hit], tr["vy"].values[hit]),
        }
    )


def turn_angle(tracks: pd.DataFrame, k: int = 10) -> pd.Series:
    """Net heading change (deg, CCW positive = left turn) between first and last k frames.

    Uses the velocity direction (not heading_rad) and only frames with speed > 1 m/s.
    """
    tr = tracks[np.hypot(tracks["vx"], tracks["vy"]) > 1.0]

    def f(g):
        if len(g) < 2 * k:
            return np.nan
        a0 = np.arctan2(g["vy"].values[:k].mean(), g["vx"].values[:k].mean())
        a1 = np.arctan2(g["vy"].values[-k:].mean(), g["vx"].values[-k:].mean())
        return np.degrees((a1 - a0 + np.pi) % (2 * np.pi) - np.pi)

    out = tr.groupby("track_id").apply(f)
    return out if isinstance(out, pd.Series) else pd.Series(dtype=float)   # no moving track -> empty DataFrame
