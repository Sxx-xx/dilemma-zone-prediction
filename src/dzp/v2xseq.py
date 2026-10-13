"""V2X-Seq TFD (DAIR-V2X-Seq, Beijing Yizhuang, CVPR 2023), single-infrastructure view -> sind.Recording.

Data (extracted V2X-Seq-TFD/): single-infrastructure/trajectories/{train,val}/<scene>.csv (10 Hz, UTM metres,
theta = heading rad, v_x/v_y), single-infrastructure/traffic-light/{train,val}/<scene>.csv (10 Hz, one row per
controlled lane_id: color_1 = current state RED/GREEN/YELLOW, remain_1 = whole seconds), maps/yizhuang_hdmap*.json
(LANE with turn_direction / predecessors / successors / centerline; the STOPLINE table is not usable: a handful of
lines per map, none at the recorded intersections).

Mapping onto the SinD pipeline (dzp.events / dzp.features):
  * sites       : the files' intersect_id is not reliable (scenes labelled e.g. yizhuang#5-1_po lie kilometres
                  apart); a site is a cluster (SITE_RADIUS) of the scenes' signal-position centroids, named after
                  the majority intersect_id (suffix ~b, ~c for further clusters with the same label).
  * scenes      : 9.9 s each; scenes of one site that start 5 s apart overlap with identical rows -> chained
                  into one *segment* (drop duplicate (id, timestamp) rows). Scenes with an empty signal file are
                  left out and end a segment.
  * stop line   : the signal lane_id is the connector lane *inside* the intersection; its first centerline point
                  is where the controlled lane enters the junction (= end of its predecessor lane). The signal
                  file lists only ONE through connector per approach while the approach has several through
                  lanes, so the approach is widened over the predecessor's left/right neighbours that are
                  through-capable CITY_DRIVING lanes (_through_lanes). Through connectors of one direction are
                  clustered into an approach; the stop line is the segment through the lane end points,
                  perpendicular to the lane heading, widened by LANE_W / 2 at both ends, so features.build's lane
                  index (LANE_W bands from the p0 end) coincides with the map lanes.
  * line shift  : the connector start is the map's lane boundary, not the painted stop line: at many approaches
                  through vehicles halt with their front 1-5 m beyond it and would be labelled "go". A per-approach
                  shift table (scripts/v2xseq_calibrate.py: 10th percentile of the leading halt positions - 1 m,
                  clipped to SHIFT_RANGE) moves the line to where vehicles actually stop; matched by site and
                  stop-line midpoint (SHIFT_MATCH_M).
  * lights      : per approach, color_1 of its through lanes (first lane; disagreement counted in Recording.notes)
                  -> SinD codes (0 red / 1 green / 3 yellow), runs shorter than FLICKER_S are merged into the
                  previous state (the record flickers GREEN/YELLOW for 0.1 s), then an event table (frame = 0.1 s
                  track frames from the segment start). Recording.light_map gives approach sid -> light column, so
                  no least-red inference is needed.
  * vehicles    : type VEHICLE, sub_type CAR / VAN -> car, TRUCK -> truck, BUS -> bus (SinD MOTOR_VEHICLES);
                  a_lon = central difference of the speed, as in dzp.fluid.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .lanelet_map import LaneletMap
from .sind import GREEN, RED, YELLOW, Recording

DT = 0.1
STITCH_GAP = 5.05          # s between scene starts that still overlap (scenes are 9.9 s, offsets 5 s)
SITE_RADIUS = 150.0        # m, scenes whose signal centroids are closer than this are one site
SHIFT_RANGE = (-8.0, 3.0)  # m, allowed stop-line shift (negative = downstream, into the junction)
SHIFT_MATCH_M = 6.0        # m, max distance between a line midpoint and a shift-table entry
FLICKER_S = 1.0            # signal runs shorter than this are dropped
LANE_W = 3.5               # m, map lanes here are 3.5 m
CLUSTER_ALONG = 12.0       # m, max spread of connector start points along the travel direction in one approach
CLUSTER_LAT = 2.0 * LANE_W  # m, max gap between neighbouring lane starts across the travel direction
STATE = {"RED": RED, "GREEN": GREEN, "YELLOW": YELLOW}
SUB_TYPE = {"CAR": "car", "VAN": "car", "TRUCK": "truck", "BUS": "bus"}
TL_COLS = ["timestamp", "lane_id", "intersect_id", "color_1"]
TR_COLS = ["timestamp", "id", "type", "sub_type", "x", "y", "length", "theta", "v_x", "v_y", "intersect_id"]


@dataclass
class Segment:
    inter: str
    name: str
    scenes: list[str]      # "<split>/<scene>" in start order
    t0: float
    t1: float


# ----------------------------------------------------------------------------------------------- maps
def _cross(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _pts(lst):
    return np.array([[float(v) for v in p.strip("() ").split(",")] for p in lst], dtype=np.float64)


def lane_index(root: str, cache_dir: str) -> dict:
    """lane_id -> dict(turn, pred, succ, cl (n,2), is_int, ltype, left, right, tc); first occurrence wins (lanes repeat
    across map files)."""
    path = os.path.join(cache_dir, "lane_index_v2.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    idx = {}
    for f in sorted(glob.glob(os.path.join(root, "maps", "*.json"))):
        for k, v in json.load(open(f))["LANE"].items():
            if k not in idx:
                idx[k] = dict(turn=v["turn_direction"], pred=list(v["predecessors"]), succ=list(v["successors"]),
                              cl=_pts(v["centerline"]), is_int=bool(v["is_intersection"]), ltype=v["lane_type"],
                              left=v["l_neighbor_id"], right=v["r_neighbor_id"], tc=bool(v["has_traffic_control"]))
    os.makedirs(cache_dir, exist_ok=True)
    pickle.dump(idx, open(path, "wb"))
    return idx


# ----------------------------------------------------------------------------------------------- scenes
def scene_index(root: str, cache_dir: str) -> pd.DataFrame:
    """One row per scene: split, scene, inter (file label), t0, t1, has_tl, tl_x/tl_y (signal centroid), n_tl_lanes,
    site (signal-position cluster, see module doc)."""
    path = os.path.join(cache_dir, "scene_index.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "site" in df.columns:
            return df
    else:
        rows = []
        for split in ("train", "val"):
            for f in sorted(glob.glob(os.path.join(root, "single-infrastructure", "trajectories", split, "*.csv"))):
                sid = os.path.basename(f)[:-4]
                t = pd.read_csv(f, usecols=["timestamp", "intersect_id"])
                rows.append(dict(split=split, scene=sid, inter=t["intersect_id"].iloc[0], t0=float(t["timestamp"].min()),
                                 t1=float(t["timestamp"].max())))
        df = pd.DataFrame(rows)
    tlx, tly, nl = [], [], []
    for r in df.itertuples(index=False):
        f = os.path.join(root, "single-infrastructure", "traffic-light", r.split, f"{r.scene}.csv")
        d = pd.read_csv(f, usecols=["x", "y", "lane_id"]) if os.path.exists(f) and os.path.getsize(f) > 200 else None
        if d is None or d.empty:
            tlx.append(np.nan), tly.append(np.nan), nl.append(0)
        else:
            tlx.append(float(d["x"].mean())), tly.append(float(d["y"].mean())), nl.append(int(d["lane_id"].nunique()))
    df["tl_x"], df["tl_y"], df["n_tl_lanes"] = tlx, tly, nl
    df["has_tl"] = df["n_tl_lanes"] > 0
    df["site"] = _sites(df)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _sites(df: pd.DataFrame) -> pd.Series:
    """Greedy clustering of the signal centroids; cluster label = majority intersect_id (+ ~b, ~c, ... if taken)."""
    site = pd.Series([None] * len(df), index=df.index, dtype=object)
    ok = df["has_tl"].values
    xy = df[["tl_x", "tl_y"]].values
    order = np.where(ok)[0]
    centers, members = [], []
    for i in order:
        if centers:
            dist = np.hypot(*(np.array(centers) - xy[i]).T)
            j = int(np.argmin(dist))
            if dist[j] <= SITE_RADIUS:
                members[j].append(i)
                centers[j] = xy[members[j]].mean(0)
                continue
        centers.append(xy[i].copy())
        members.append([i])
    taken = {}
    for m in sorted(members, key=len, reverse=True):
        label = df["inter"].iloc[m].mode().iloc[0]
        k = taken.get(label, 0)
        taken[label] = k + 1
        name = label if k == 0 else f"{label}~{chr(ord('a') + k)}"
        site.iloc[m] = name
    return site


def segments(root: str, cache_dir: str) -> list[Segment]:
    """Chain overlapping scenes (same intersection, starts <= STITCH_GAP apart) into segments; scenes without a
    signal record are skipped and break the chain."""
    si = scene_index(root, cache_dir)
    si = si[si["has_tl"]].sort_values(["site", "t0"]).reset_index(drop=True)
    out, k = [], 0
    for inter, g in si.groupby("site", sort=True):
        cur = None
        for r in g.itertuples(index=False):
            if cur is not None and r.t0 - cur[-1].t0 <= STITCH_GAP and r.t0 <= cur[-1].t1 + DT:
                cur.append(r)
            else:
                if cur:
                    out.append(Segment(inter, f"{inter}/seg{k:05d}", [f"{c.split}/{c.scene}" for c in cur], cur[0].t0, cur[-1].t1))
                    k += 1
                cur = [r]
        if cur:
            out.append(Segment(inter, f"{inter}/seg{k:05d}", [f"{c.split}/{c.scene}" for c in cur], cur[0].t0, cur[-1].t1))
            k += 1
    return out


# ----------------------------------------------------------------------------------------------- geometry
def _through_lanes(conn: str, lanes: dict) -> list[str]:
    """Upstream lanes of a through connector: its predecessor plus the left/right neighbours (walked until the first
    lane that is not a through-capable CITY_DRIVING lane, i.e. a turn-only lane, a bike lane or the median)."""
    C = lanes[conn]
    ups = [u for u in C["pred"] if u in lanes and np.hypot(*(lanes[u]["cl"][-1] - C["cl"][0])) < 2.0]
    if not ups:
        return []

    def through_capable(lid):
        L = lanes.get(lid)
        return bool(L) and L["ltype"] == "CITY_DRIVING" and any(lanes[s]["turn"] == "NONE" for s in L["succ"] if s in lanes)

    out = []
    for u in ups:
        if not through_capable(u):
            out.append(u)                      # keep the signalled lane itself in any case
        else:
            out.append(u)
        for side in ("left", "right"):
            cur = lanes[u].get(side)
            while cur and cur != "None" and through_capable(cur) and cur not in out:
                out.append(cur)
                cur = lanes[cur].get(side)
    return out


def approaches_from_lanes(lane_ids: list[str], lanes: dict):
    """Cluster the through connector lanes into approaches and widen each over all through-capable lanes of the
    approach (_through_lanes). Returns (stop_line_xy list, approach_lanes list, notes)."""
    pts, notes = [], []
    for lid in lane_ids:
        L = lanes.get(lid)
        if L is None or L["turn"] != "NONE" or len(L["cl"]) < 2:
            continue
        ups = _through_lanes(lid, lanes)
        if ups:
            ends = [lanes[u]["cl"][-1] for u in ups]
            heads = [lanes[u]["cl"][-1] - lanes[u]["cl"][max(len(lanes[u]["cl"]) - 4, 0)] for u in ups]
            h = np.sum([hh / np.linalg.norm(hh) for hh in heads], axis=0)
        else:
            notes.append(f"{lid}: no upstream lane in map, stop line = connector only")
            ends = [L["cl"][0]]
            h = L["cl"][min(3, len(L["cl"]) - 1)] - L["cl"][0]
        pts.append((lid, ends, h / np.linalg.norm(h), ups or [lid]))
    groups = []
    for lid, ends, h, ups in pts:
        for g in groups:
            hm = g["h"] / np.linalg.norm(g["h"])
            if np.dot(hm, h) < np.cos(np.radians(30)):
                continue
            p0 = g["p"][0]
            if all(abs(np.dot(q - p0, hm)) <= CLUSTER_ALONG for q in ends) and \
                    min(abs(_cross(hm, q - p0) - _cross(hm, r - p0)) for q in ends for r in g["p"]) <= CLUSTER_LAT:
                g["ids"].append(lid)
                g["p"] += [q for q in ends if not any(np.allclose(q, r) for r in g["p"])]
                g["lanes"] += [u for u in ups if u not in g["lanes"]]
                g["h"] = g["h"] + h
                break
        else:
            groups.append(dict(ids=[lid], p=list(ends), h=h.copy(), lanes=list(ups)))
    lines, members = [], []
    for g in groups:
        n = g["h"] / np.linalg.norm(g["h"])
        t = np.array([-n[1], n[0]])
        P = np.array(g["p"])
        mid = P.mean(0)
        lat = (P - mid) @ t
        # endpoint order such that geometry.build_approaches' default normal (-t_y, t_x) is the travel direction n:
        # with a single approach it cannot orient the line from the stop-line centroid, so the order must be right.
        lines.append(np.array([mid + t * (lat.max() + LANE_W / 2), mid + t * (lat.min() - LANE_W / 2)]))
        members.append(dict(signal=g["ids"], lanes=g["lanes"]))
    return lines, members, notes


def _map_from_lines(lines) -> LaneletMap:
    nodes, ways = {}, {}
    for i, xy in enumerate(lines):
        nodes[f"n{i}a"], nodes[f"n{i}b"] = xy[0], xy[1]
        ways[f"sl{i}"] = {"nodes": [f"n{i}a", f"n{i}b"], "tags": {"type": "stop_line"}}
    return LaneletMap(nodes=nodes, ways=ways, relations={})


# ----------------------------------------------------------------------------------------------- loading
def _read(root: str, kind: str, scene: str, cols):
    split, sid = scene.split("/")
    return pd.read_csv(os.path.join(root, "single-infrastructure", kind, split, f"{sid}.csv"), usecols=cols)


def load_tracks(root: str, seg: Segment) -> pd.DataFrame:
    t = pd.concat([_read(root, "trajectories", s, TR_COLS) for s in seg.scenes], ignore_index=True)
    t = t[(t["type"] == "VEHICLE") & t["sub_type"].isin(SUB_TYPE)]
    t = t.drop_duplicates(["id", "timestamp"]).copy()
    t["agent_type"] = t["sub_type"].map(SUB_TYPE)
    t["frame_id"] = np.round((t["timestamp"].values - seg.t0) / DT).astype(int)
    t = t.rename(columns={"id": "track_id", "v_x": "vx", "v_y": "vy", "theta": "heading_rad"})
    t = t.sort_values(["track_id", "frame_id"]).reset_index(drop=True)
    # dzp.events / dzp.features index tracks by frame offset -> a track must be frame-contiguous. Ids that vanish
    # and reappear (occlusion, or absent from a middle scene) are split: new track_id = id * 100 + piece.
    tid0 = t["track_id"].values
    gap = np.r_[False, (tid0[1:] == tid0[:-1]) & (np.diff(t["frame_id"].values) > 1)]
    piece = pd.Series(gap.astype(int)).groupby(tid0).cumsum().values
    t["orig_id"] = tid0
    t["track_id"] = tid0 * 100 + piece
    t = t.sort_values(["track_id", "frame_id"]).reset_index(drop=True)
    v = np.hypot(t["vx"].values, t["vy"].values)
    tid = t["track_id"].values
    same_prev = np.r_[False, tid[1:] == tid[:-1]]
    same_next = np.r_[tid[:-1] == tid[1:], False]
    vp = np.where(same_prev, np.r_[np.nan, v[:-1]], v)
    vn = np.where(same_next, np.r_[v[1:], np.nan], v)
    span = (same_prev.astype(int) + same_next.astype(int)) * DT
    t["a_lon"] = np.where(span > 0, (vn - vp) / np.where(span > 0, span, 1), 0.0)
    return t[["track_id", "orig_id", "frame_id", "timestamp", "agent_type", "x", "y", "vx", "vy", "a_lon", "heading_rad", "length"]]


def _runs_to_events(frames: np.ndarray, states: np.ndarray, col: str) -> pd.DataFrame:
    """State samples -> change table; runs shorter than FLICKER_S are merged into the preceding state."""
    chg = np.r_[True, states[1:] != states[:-1]]
    f, s = frames[chg], states[chg]
    ends = np.r_[f[1:], frames[-1] + 1]
    keep = np.ones(len(f), bool)
    for i in range(1, len(f)):
        if (ends[i] - f[i]) * DT < FLICKER_S and i < len(f) - 1:
            keep[i] = False
    f, s = f[keep], s[keep]
    chg = np.r_[True, s[1:] != s[:-1]]
    return pd.DataFrame({"frame": f[chg].astype(float), col: s[chg]})


def load_shift_table(path: str) -> pd.DataFrame | None:
    return pd.read_csv(path) if path and os.path.exists(path) else None


def apply_shift(lines, site: str, table: pd.DataFrame | None, notes: list) -> list:
    """Move each stop line along its travel direction by the calibrated shift (see module doc)."""
    if table is None or not lines:
        return lines
    tab = table[table["site"] == site]
    out = []
    for xy in lines:
        mid = xy.mean(0)
        t = (xy[1] - xy[0]) / np.linalg.norm(xy[1] - xy[0])
        n = np.array([-t[1], t[0]])
        if len(tab):
            dist = np.hypot(tab["mid_x"].values - mid[0], tab["mid_y"].values - mid[1])
            j = int(np.argmin(dist))
            if dist[j] <= SHIFT_MATCH_M:
                sh = float(np.clip(tab["shift_m"].values[j], *SHIFT_RANGE))
                out.append(xy - n * sh)           # shift < 0 -> downstream (along n)
                continue
        notes.append(f"no stop-line shift entry near ({mid[0]:.0f}, {mid[1]:.0f})")
        out.append(xy)
    return out


def load_lights(root: str, seg: Segment, lanes: dict, shift_s: float = 0.0, shift_table: pd.DataFrame | None = None):
    tl = pd.concat([_read(root, "traffic-light", s, TL_COLS) for s in seg.scenes], ignore_index=True)
    tl = tl.drop_duplicates(["lane_id", "timestamp"])
    lines, members, notes = approaches_from_lanes(sorted(tl["lane_id"].unique()), lanes)
    lines = apply_shift(lines, seg.inter, shift_table, notes)
    tabs, cols = [], []
    for sid, m in enumerate(members):
        col = f"light_{sid}"
        g = tl[tl["lane_id"].isin(m["signal"])].copy()
        g["code"] = g["color_1"].map(STATE).fillna(-1).astype(int)
        piv = g.pivot_table(index="timestamp", columns="lane_id", values="code", aggfunc="first")
        if piv.shape[1] > 1:
            dis = int((piv.nunique(axis=1) > 1).sum())
            if dis:
                notes.append(f"{col}: lanes disagree at {dis} samples")
        frames = np.round((piv.index.values - seg.t0 + shift_s) / DT).astype(int)
        states = piv.iloc[:, 0].ffill().bfill().values.astype(int)
        tabs.append(_runs_to_events(frames, states, col))
        cols.append(col)
    lights = tabs[0] if tabs else pd.DataFrame({"frame": []})
    for t in tabs[1:]:
        lights = lights.merge(t, on="frame", how="outer")
    lights = lights.sort_values("frame").ffill().reset_index(drop=True)
    return lights, cols, lines, members, notes


def load_recording(root: str, seg: Segment, lanes: dict, shift_s: float = 0.0,
                   shift_table: pd.DataFrame | None = None) -> Recording:
    lights, cols, lines, members, notes = load_lights(root, seg, lanes, shift_s, shift_table)
    rec = Recording(name=seg.name, intersection=seg.inter, tracks=load_tracks(root, seg), lights=lights, light_cols=cols,
                    map=_map_from_lines(lines), notes=notes, light_map={i: c for i, c in enumerate(cols)})
    rec.approach_lanes = members
    return rec
