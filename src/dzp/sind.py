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
    lights_end_frame: float | None = None    # when set, the state after this frame is unknown (event table ends early)

    def light_state_at(self, col: str, frame: np.ndarray) -> np.ndarray:
        """Light state at (possibly fractional) track frames; -1 before the first event (and after lights_end_frame)."""
        idx = np.searchsorted(self.lights["frame"].values, frame, side="right") - 1
        st = self.lights[col].values[np.clip(idx, 0, None)].astype(int)
        st = np.where(idx < 0, -1, st)
        if self.lights_end_frame is not None:
            st = np.where(np.asarray(frame) > self.lights_end_frame, -1, st)
        return st


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


# ----------------------------------------------------------------------------------------------- SinD v2.0 full release
FULL_CITIES = {  # Data/<folder> -> (intersection id, map path relative to the delivery root)
    "Changchun": ("cc", "Data/Changchun/Changchun_Pudong.osm"),
    "tianjin": ("tj", "Maps/lanelet2/lanelet2_tj.osm"),
    "Chongqing_IR": ("cqIR", "Data/Chongqing_IR/IR_ll2.osm"),
    "Chongqing_NR": ("cqNR", "Data/Chongqing_NR/NR_ll2.osm"),
    "Chongqing_R": ("cqR", "Data/Chongqing_R/R_ll2.osm"),
    "Xi'an": ("xasl", "Data/Xi'an/Xi'an_Shanglin.osm"),
}
RAW_VIDEO_FPS = 30.0  # official fallback when the light CSV has no usable timestamp(ms) (sind_traffic_lights.py)


def full_offsets(root: str) -> pd.DataFrame:
    """Conversion_Tools/signal_time_offsets_by_recording.csv indexed by recording folder name."""
    p = os.path.join(root, "Conversion_Tools", "signal_time_offsets_by_recording.csv")
    return pd.read_csv(p).set_index("recording")


def full_recordings(root: str) -> list[tuple[str, str]]:
    """[(city folder, recording folder)] that have both tracks and a traffic-light CSV."""
    out = []
    for city in FULL_CITIES:
        d = os.path.join(root, "Data", city)
        for rec in sorted(os.listdir(d)):
            rd = os.path.join(d, rec)
            if os.path.isdir(rd) and os.path.exists(os.path.join(rd, "Veh_smoothed_tracks.csv")) and glob.glob(os.path.join(rd, "Traffic*Light*.csv")):
                out.append((city, rec))
    return out


def _load_lights_full(path: str, offset_ms: float) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Light event table on the track clock, following the official rule (trajdata sind_traffic_lights.py):
    timestamp(ms) - offset when timestamp(ms) varies, else (RawFrameID - first RawFrameID) / 30 fps - offset."""
    raw = pd.read_csv(path)
    raw.columns = [("timestamp(ms)" if c.lower() == "timestamp(ms)" else c) for c in raw.columns]   # tianjin/8_5_3: 'Timestamp(ms)'
    notes = []
    veh_cols = [c for c in raw.columns if "raffic light" in c and not c.startswith("Pedestrian")]
    ts = pd.to_numeric(raw["timestamp(ms)"], errors="coerce")
    fin = ts[ts.notna()]
    if len(fin) > 1 and np.ptp(fin.values) > 1e-6:
        if ts.isna().any():
            notes.append(f"{int(ts.isna().sum())} missing timestamp(ms) interpolated")
        t_ms = ts.interpolate().ffill().bfill().values - offset_ms
        notes.append("clock: timestamp(ms)" + (f" - offset {offset_ms:.0f} ms" if offset_ms else ""))
    else:
        rf = pd.to_numeric(raw["RawFrameID"], errors="coerce").values.astype(float)
        t_ms = (rf - np.nanmin(rf)) / RAW_VIDEO_FPS * 1000.0 - offset_ms
        notes.append(f"clock: RawFrameID relative at {RAW_VIDEO_FPS:g} fps - offset {offset_ms:.0f} ms (timestamp(ms) unusable)")
    out = pd.DataFrame({"raw_frame": pd.to_numeric(raw["RawFrameID"], errors="coerce").fillna(-1).astype(int), "t_ms": t_ms})
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


def load_full_recording(root: str, city: str, rec: str, offsets: pd.DataFrame | None = None, maps: dict | None = None) -> Recording:
    """One SinD v2.0 recording (root = the SinD_v2.0_CSV_Dataset folder). Recording.name = '<city>/<rec>'."""
    inter, map_rel = FULL_CITIES[city]
    d = os.path.join(root, "Data", city, rec)
    tracks = pd.read_csv(os.path.join(d, "Veh_smoothed_tracks.csv"))
    tracks = tracks.sort_values(["track_id", "frame_id"]).reset_index(drop=True)
    offsets = full_offsets(root) if offsets is None else offsets
    key = rec if rec in offsets.index else next((k for k in offsets.index if str(k).split(" ")[0] == rec), None)  # tianjin: '8_2_1 R1'
    off = float(offsets.loc[key, "signal_time_offset_ms"]) if key is not None else 0.0
    light_file = sorted(glob.glob(os.path.join(d, "Traffic*Light*.csv")))[0]
    lights, cols, notes = _load_lights_full(light_file, off)
    if key is not None:
        if bool(offsets.loc[key, "missing_signal_data"]):
            notes.append("official: missing_signal_data")
        qw = str(offsets.loc[key, "quality_warning"])
        if qw not in ("{}", "nan"):
            notes.append("official quality warning: " + qw[:120])
    else:
        notes.append("not in signal_time_offsets_by_recording.csv")
    meta_path = os.path.join(d, "Veh_tracks_meta.csv")
    meta = pd.read_csv(meta_path) if os.path.exists(meta_path) else None
    if maps is not None and map_rel in maps:
        m = maps[map_rel]
    else:
        m = load_lanelet_map(os.path.join(root, map_rel))
        if maps is not None:
            maps[map_rel] = m
    r = Recording(f"{city}/{rec}", inter, tracks, lights, cols, m, meta, notes)
    # the light tables are hand-made event lists that may stop before the recording ends (Chongqing: ~half) ->
    # states after the last listed change are unknown, not held
    r.lights_end_frame = float(lights["frame"].max())
    t_end = tracks["timestamp_ms"].max() / (DT * 1000)
    if r.lights_end_frame < t_end - 60 / DT:
        r.notes.append(f"light table ends at {r.lights_end_frame * DT:.0f} s of {t_end * DT:.0f} s -> later states unknown")
    return r
