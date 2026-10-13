"""SinD sample-data loader (trajectories, traffic lights, Lanelet2 map).

Verified against the public sample in github.com/SOTIF-AVLab/SinD (see docs/FINDINGS.md):
  * Veh_smoothed_tracks.csv: one row per (track_id, frame_id), 10 Hz (timestamp_ms step 100.1 ms),
    x/y in metres in the map frame, vx/vy, ax/ay, v_lon/a_lon, heading_rad, length, width.
  * Traffic light CSV: *event table* -- one row per change of any light.
    Codes: 0 = red, 1 = green, 3 = yellow (Format.md + observed cycle 1->3->0).
  * Time alignment: the light CSV's `timestamp(ms)` column is on the same clock as the tracks'
    `timestamp_ms`. RawFrameID is NOT (offset differs per recording, and Xi'an's raw video is
    59.94 fps, not 29.97). Verified empirically: with timestamp(ms) alignment, almost no
    non-right-turn vehicle crosses a stop line on red (scripts/sind_check_alignment.py).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .lanelet_map import LaneletMap, load_lanelet_map

RED, GREEN, YELLOW = 0, 1, 3
DT = 0.1001  # s, track sampling period (timestamp_ms step = 100.1 ms)

# recording dir (relative to data/raw/sind) -> (intersection id, map file)
SAMPLE_RECORDINGS = {
    "Changchun/changchun_pudong_507_009": ("cc", "Changchun/Changchun_Pudong.osm"),
    "Chongqing/6_22_NR_1": ("cqNR", "Chongqing/NR_ll2.osm"),
    "Tianjin/8_2_1": ("tj", "Tianjin/map_relink_law_save.osm"),
    "Xi'an/Xi'an_412_m1": ("xasl", "Xi'an/Xi'an_Shanglin.osm"),
}
MOTOR_VEHICLES = ("car", "truck", "bus")


@dataclass
class Recording:
    name: str
    intersection: str
    tracks: pd.DataFrame
    lights: pd.DataFrame  # event table: frame (track-frame units, float), light_1..light_n
    light_cols: list[str]
    map: LaneletMap
    meta: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)
    light_map: dict[int, str] | None = None  # approach sid -> light column when known from the data (else inferred)

    def light_state_at(self, col: str, frame: np.ndarray) -> np.ndarray:
        """Light state at (possibly fractional) track frames; -1 before the first event."""
        idx = np.searchsorted(self.lights["frame"].values, frame, side="right") - 1
        st = self.lights[col].values[np.clip(idx, 0, None)].astype(int)
        return np.where(idx < 0, -1, st)


def _load_lights(path: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    raw = pd.read_csv(path)
    notes = []
    veh_cols = [c for c in raw.columns if "raffic light" in c and not c.startswith("Pedestrian")]
    ts = raw["timestamp(ms)"].astype(float)
    if ts.isna().any():
        # fill from a linear fit RawFrameID -> timestamp (raw fps differs between recordings)
        ok = ts.notna()
        a, b = np.polyfit(raw.loc[ok, "RawFrameID"], ts[ok], 1)
        notes.append(f"{int((~ok).sum())} missing timestamp(ms) filled by fit (raw fps={1000 / a:.2f})")
        ts = ts.fillna(raw["RawFrameID"] * a + b)
    out = pd.DataFrame({"raw_frame": raw["RawFrameID"].astype(int), "t_ms": ts})
    names = []
    for c in veh_cols:
        k = c.split()[-1]
        names.append(f"light_{k}")
        out[f"light_{k}"] = raw[c].astype(int)
    unknown = set(np.unique(out[names].values)) - {RED, GREEN, YELLOW}
    if unknown:
        notes.append(f"unknown light codes {sorted(unknown)}")
    if not out["t_ms"].is_monotonic_increasing:
        notes.append("light events not time-ordered -> sorted (stable)")
        out = out.sort_values("t_ms", kind="stable").reset_index(drop=True)
    out["frame"] = out["t_ms"] / (DT * 1000)
    return out, names, notes


def load_recording(root: str, rec: str) -> Recording:
    inter, map_rel = SAMPLE_RECORDINGS[rec]
    d = os.path.join(root, rec)
    tracks = pd.read_csv(os.path.join(d, "Veh_smoothed_tracks.csv"))
    tracks = tracks.sort_values(["track_id", "frame_id"]).reset_index(drop=True)
    light_file = sorted(glob.glob(os.path.join(d, "Traffic*Light*.csv")))[0]
    lights, cols, notes = _load_lights(light_file)
    meta_path = os.path.join(d, "Veh_tracks_meta.csv")
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    m = load_lanelet_map(os.path.join(root, map_rel))
    return Recording(rec, inter, tracks, lights, cols, m, meta, notes)


def load_all(root: str = "data/raw/sind") -> list[Recording]:
    return [load_recording(root, r) for r in SAMPLE_RECORDINGS if os.path.isdir(os.path.join(root, r))]
