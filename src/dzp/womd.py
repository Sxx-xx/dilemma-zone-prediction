"""Waymo Open Motion Dataset v1.3.1 (scenario protos) -> yellow-onset leader events in the project's table schema.

No TensorFlow: trimmed protos under src/womd_protos, TFRecord read directly. Per shard (`extract_shard`):
  * scenarios parsed into arrays; yellow onsets always come from the ORIGINAL signal table. The Michigan corrected
    dynamic_map_states (WOMD-Traffic-Signal-Data-Improvement, matched by scenario_id) are used only to fill the red
    onset (t_r) when the original yellow ends in UNKNOWN -- their yellow is a synthetic 2 s insert before every
    estimated red, so it cannot serve as an onset (t_r_source = original | michigan);
  * stitching: scenarios of the same drive share the coordinate frame (verified: lane geometry + SDC trajectory
    coincide at a whole-step offset). Windows are grouped by matching the SDC trajectory (median distance < 0.1 m at
    some offset); objects are merged by position, signal lanes by stop_point (map ids are per-scenario);
  * yellow onset = transition GO/ARROW_GO -> CAUTION/ARROW_CAUTION of a signalled lane; stop line = the lane's
    stop_point = start of the signalled connector lane; approach = chain of entry_lanes (<= UP_MAX m upstream);
    movement = turn angle of the connector lane (through < 30 deg); arrow onsets flagged (`arrow`);
  * leader = nearest vehicle to the stop point on the chain (|lateral| <= LAT_M, heading within 45 deg) with no
    vehicle between it and the line; label until the window end: go = crosses the stop point, stop = speed < V_STOP
    while upstream, else unresolved (label_relaxed: unresolved but upstream, v < 3 m/s and braking at the window end ->
    stop; d_end / v_end / a_end / t_end stored so labels can be redefined later); t_r = yellow end (red onset);
  * features as dzp.features.build: d, v, a (1 s moving average of |velocity|; a_raw kept), t_r, tt, dreq ... and the
    2 s (x = -d, lat, v, a) window; acceleration noise |da| before/after smoothing per track.
"""
from __future__ import annotations

import os
import struct
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "womd_protos"))
from waymo_open_dataset.protos import scenario_pb2  # noqa: E402

DT = 0.1
CAUTION, GO, STOP = {2, 5}, {3, 6}, {1, 4}
ARROW = {1, 2, 3}
LAT_M, V_STOP, UP_MAX, HIST, TT_MAX = 2.5, 1.0, 120.0, 20, 12.0
SMOOTH_S = 1.0
STITCH_TOL = 0.1


def records(path):
    with open(path, "rb") as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                return
            n = struct.unpack("<Q", h[:8])[0]
            yield f.read(n)
            f.read(4)


# ----------------------------------------------------------------------------------------------- parsing
def parse(raw):
    sc = scenario_pb2.Scenario()
    sc.ParseFromString(raw)
    T = len(sc.timestamps_seconds)
    tracks = []
    for tr in sc.tracks:
        if tr.object_type != 1:
            continue
        st = tr.states
        tracks.append(dict(id=tr.id, valid=np.array([s.valid for s in st]), x=np.array([s.center_x for s in st]), y=np.array([s.center_y for s in st]),
                           vx=np.array([s.velocity_x for s in st]), vy=np.array([s.velocity_y for s in st]), hd=np.array([s.heading for s in st]),
                           length=float(np.median([s.length for s in st if s.valid]) if any(s.valid for s in st) else 4.5)))
    sdc = sc.tracks[sc.sdc_track_index].states
    lanes = {mf.id: dict(poly=np.array([(q.x, q.y) for q in mf.lane.polyline]), entry=list(mf.lane.entry_lanes), type=mf.lane.type)
             for mf in sc.map_features if mf.WhichOneof("feature_data") == "lane" and len(mf.lane.polyline) >= 2}
    return dict(id=sc.scenario_id, T=T, tracks=tracks, lanes=lanes, sdc=np.array([[s.center_x, s.center_y] for s in sdc]),
                sdc_valid=np.array([s.valid for s in sdc]), signals=signal_table(sc.dynamic_map_states, T))


def signal_table(dynamic_map_states, T):
    """{lane id: (state array over T, stop_point xy)}"""
    seq, sp = {}, {}
    for k, dm in enumerate(dynamic_map_states):
        for ls in dm.lane_states:
            if ls.lane not in seq:
                seq[ls.lane] = np.zeros(T, int)
                sp[ls.lane] = (ls.stop_point.x, ls.stop_point.y)
            seq[ls.lane][k] = ls.state
    return {l: (seq[l], sp[l]) for l in seq}


def michigan_index(folder):
    """scenario_id -> (file, byte offset, length) over the corrected-signal TFRecords of one split."""
    idx = {}
    if not folder or not os.path.isdir(folder):
        return idx
    for fn in sorted(os.listdir(folder)):
        p = os.path.join(folder, fn)
        if not fn.endswith("-new"):
            continue
        with open(p, "rb") as f:
            while True:
                pos = f.tell()
                h = f.read(12)
                if len(h) < 12:
                    break
                n = struct.unpack("<Q", h[:8])[0]
                raw = f.read(n)
                f.read(4)
                sc = scenario_pb2.Scenario()
                sc.ParseFromString(raw)
                idx[sc.scenario_id] = (p, pos + 12, n)
    return idx


def michigan_signals(entry, T):
    p, off, n = entry
    with open(p, "rb") as f:
        f.seek(off)
        raw = f.read(n)
    sc = scenario_pb2.Scenario()
    sc.ParseFromString(raw)
    return signal_table(sc.dynamic_map_states, T)


# ----------------------------------------------------------------------------------------------- stitching
def _offset(a, b, max_off=200):
    """Whole-step offset such that b's SDC at step k matches a's at k + off (median error < STITCH_TOL), else None."""
    pa, pb = a["sdc"], b["sdc"]
    tb = cKDTree(pa)
    d, j = tb.query(pb)
    if (d < 0.3).sum() < 10:
        return None
    cand = np.unique((j - np.arange(len(pb)))[d < 0.3])
    best = None
    for off in cand:
        kb = np.arange(len(pb))
        ka = kb + off
        m = (ka >= 0) & (ka < len(pa))
        if m.sum() < 10:
            continue
        e = np.hypot(*(pa[ka[m]] - pb[kb[m]]).T)
        if np.median(e) < STITCH_TOL and (best is None or m.sum() > best[1]):
            best = (int(off), int(m.sum()))
    return best[0] if best else None


def stitch_groups(scs):
    """Union-find over scenarios whose SDC trajectories coincide (shared frame); returns [(members, offsets)] with
    offsets in steps relative to the group's earliest window."""
    n = len(scs)
    starts = np.array([s["sdc"][0] for s in scs])
    tree = cKDTree(starts)
    parent = list(range(n))
    rel = {}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, k in sorted(tree.query_pairs(250.0)):
        off = _offset(scs[i], scs[k])
        if off is None:
            continue
        rel[(i, k)] = off          # b(k) step j == a(i) step j + off
        parent[find(k)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for root, mem in groups.items():
        if len(mem) == 1:
            out.append((mem, {mem[0]: 0}))
            continue
        # propagate offsets from the root through known pairs
        offs = {mem[0]: 0}
        changed = True
        while changed:
            changed = False
            for (i, k), off in rel.items():
                if i in offs and k not in offs:
                    offs[k] = offs[i] + off          # k's step j is i's step j+off -> k starts at offs[i] + off
                    changed = True
                elif k in offs and i not in offs:
                    offs[i] = offs[k] - off
                    changed = True
        for m in mem:
            offs.setdefault(m, None)
        mem = [m for m in mem if offs[m] is not None]
        base = min(offs[m] for m in mem)
        out.append((mem, {m: offs[m] - base for m in mem}))
    return out


def merge_group(scs, mem, offs):
    """Stitch windows into one: tracks matched by position in overlapping steps, signal lanes by stop point."""
    if len(mem) == 1:
        s = scs[mem[0]]
        return dict(id=s["id"], ids=[s["id"]], T=s["T"], tracks=s["tracks"], lanes=s["lanes"], signals=s["signals"],
                    signals_mi=s.get("signals_mi", {}), n_windows=1, sdc=s["sdc"])
    T = max(offs[m] + scs[m]["T"] for m in mem)
    mem = sorted(mem, key=lambda m: offs[m])
    tracks, lanes, signals, signals_mi = [], {}, {}, {}
    lane_off = 0
    for m in mem:
        s, o = scs[m], offs[m]
        # tracks
        for tr in s["tracks"]:
            placed = False
            for ex in tracks:
                both = np.zeros(T, bool)
                both[o:o + s["T"]] = tr["valid"]
                both &= ex["valid"]
                if both.sum() >= 3:
                    kk = np.where(both)[0]
                    e = np.hypot(ex["x"][kk] - tr["x"][kk - o], ex["y"][kk] - tr["y"][kk - o])
                    if np.median(e) < 0.2:
                        new = ~ex["valid"].copy()
                        new[:o] = False
                        new[o + s["T"]:] = False
                        new[o:o + s["T"]] &= tr["valid"]
                        for c in ("x", "y", "vx", "vy", "hd"):
                            ex[c][new] = tr[c][np.where(new)[0] - o]
                        ex["valid"] |= new
                        placed = True
                        break
            if not placed:
                nt = dict(id=tr["id"], length=tr["length"])
                for c in ("x", "y", "vx", "vy", "hd"):
                    arr = np.zeros(T)
                    arr[o:o + s["T"]] = tr[c]
                    nt[c] = arr
                v = np.zeros(T, bool)
                v[o:o + s["T"]] = tr["valid"]
                nt["valid"] = v
                tracks.append(nt)
        # lanes: keep per-window ids offset to avoid clashes (chains only follow within a window's own map)
        for lid, ln in s["lanes"].items():
            lanes[lid + lane_off] = dict(poly=ln["poly"], entry=[e + lane_off for e in ln["entry"]], type=ln["type"])
        for table, target in ((s["signals"], signals), (s.get("signals_mi", {}), signals_mi)):
            for lid, (sq, sp) in table.items():
                key = next((k for k, (_, sp2) in target.items() if abs(sp2[0] - sp[0]) < 0.5 and abs(sp2[1] - sp[1]) < 0.5), None)
                if key is None:
                    arr = np.zeros(T, int)
                    arr[o:o + s["T"]] = sq
                    target[lid + lane_off] = (arr, sp)
                else:
                    arr = target[key][0]
                    seg = arr[o:o + s["T"]]
                    seg[seg == 0] = sq[seg == 0]
        lane_off += 10 ** 6
    sdc = np.full((T, 2), np.nan)
    for m in mem:
        sdc[offs[m]:offs[m] + scs[m]["T"]] = scs[m]["sdc"]
    return dict(id=scs[mem[0]]["id"], ids=[scs[m]["id"] for m in mem], T=T, tracks=tracks, lanes=lanes, signals=signals, signals_mi=signals_mi,
                n_windows=len(mem), sdc=sdc)


# ----------------------------------------------------------------------------------------------- geometry
def lane_frame(poly):
    p = np.asarray(poly, float)
    seg = np.diff(p, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    return p, np.r_[0, np.cumsum(L)], seg / np.maximum(L, 1e-9)[:, None]


def project(p, s, tang, xy):
    a, b = p[:-1], p[1:]
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(1), 1e-9)
    xy = np.atleast_2d(xy)
    out_s, out_lat, out_t = np.empty(len(xy)), np.empty(len(xy)), np.empty((len(xy), 2))
    for i, q in enumerate(xy):
        t = np.clip(((q - a) * ab).sum(1) / ab2, 0, 1)
        proj = a + t[:, None] * ab
        j = int(np.argmin(((q - proj) ** 2).sum(1)))
        out_s[i] = s[j] + t[j] * np.sqrt(ab2[j])
        r = q - proj[j]
        out_lat[i] = tang[j][0] * r[1] - tang[j][1] * r[0]
        out_t[i] = tang[j]
    return out_s, out_lat, out_t


def upstream_chains(lanes, lid, max_len=UP_MAX, depth=4):
    """[(entry lane id, polyline ending at the start of `lid`)] -- one longest chain per direct entry lane."""
    def walk(cur, pts, length, d):
        ents = [e for e in lanes[cur]["entry"] if e in lanes]
        if not ents or length >= max_len or d >= depth:
            return [(pts, length)]
        out = []
        for e in ents:
            q = lanes[e]["poly"]
            out += walk(e, [q] + pts, length + np.hypot(*np.diff(q, axis=0).T).sum(), d + 1)
        return out

    for e in lanes[lid]["entry"]:
        if e in lanes:
            q = lanes[e]["poly"]
            chains = walk(e, [q], np.hypot(*np.diff(q, axis=0).T).sum(), 1)
            pts, _ = max(chains, key=lambda c: c[1])
            yield e, np.vstack([pts[0]] + [p[1:] for p in pts[1:]])


def lane_turn_deg(poly):
    p = np.asarray(poly)
    if len(p) < 3:
        return np.nan
    t0, t1 = p[1] - p[0], p[-1] - p[-2]
    return float(np.degrees(np.arctan2(t0[0] * t1[1] - t0[1] * t1[0], t0 @ t1)))


def smooth_speed(v, valid, w=SMOOTH_S):
    """Centred moving average over valid steps (window w seconds); NaN where the window has no valid data."""
    k = int(round(w / DT))
    vv = np.where(valid, v, 0.0)
    c = np.convolve(vv, np.ones(k), "same")
    n = np.convolve(valid.astype(float), np.ones(k), "same")
    return np.where(n > 0, c / np.maximum(n, 1), np.nan)


# ----------------------------------------------------------------------------------------------- events
LEADS = (0.0,)


def extract_window(win, source, accel_rows, leads=LEADS):
    """Yellow-onset leader events of one (stitched) window -> (event rows, feature rows, sequences)."""
    T, tracks, lanes = win["T"], win["tracks"], win["lanes"]
    for tr in tracks:
        v = np.hypot(tr["vx"], tr["vy"])
        tr["v"] = v
        tr["vs"] = smooth_speed(v, tr["valid"])
        a_raw = np.full(T, np.nan)
        ok = tr["valid"][1:] & tr["valid"][:-1]
        a_raw[1:][ok] = (v[1:][ok] - v[:-1][ok]) / DT
        a_s = np.full(T, np.nan)
        a_s[1:] = np.diff(tr["vs"]) / DT
        tr["a_raw"], tr["a"] = a_raw, a_s
        m = np.isfinite(a_raw[1:]) & np.isfinite(a_raw[:-1]) & (v[1:] > 2)
        if m.sum() >= 10:
            ms = np.isfinite(a_s[1:]) & np.isfinite(a_s[:-1]) & (v[1:] > 2)
            accel_rows.append(dict(window=win["id"], track=tr["id"], n=int(m.sum()), da_raw=float(np.abs(np.diff(a_raw))[m].mean()),
                                   da_smooth=float(np.abs(np.diff(a_s))[ms].mean()) if ms.any() else np.nan))
    ev_rows, feat_rows, seqs = [], [], {}
    for lid, (sq, sp) in win["signals"].items():
        if lid not in lanes:
            continue
        known = sq != 0
        caution = np.isin(sq, list(CAUTION))
        turn = lane_turn_deg(lanes[lid]["poly"])
        mi = win.get("signals_mi", {})
        msq = mi[lid][0] if lid in mi else next((v[0] for v in mi.values() if abs(v[1][0] - sp[0]) < 0.5 and abs(v[1][1] - sp[1]) < 0.5), None)
        chains = list(upstream_chains(lanes, lid))
        for k in range(1, T):
            if not (caution[k] and known[k - 1] and sq[k - 1] in GO):
                continue
            arrow = sq[k] in ARROW
            end = next((j for j in range(k + 1, T) if not caution[j]), None)
            end_state = int(sq[end]) if end is not None else -1
            f_red = end if (end is not None and end_state in STOP) else np.nan
            t_r_source = "original" if np.isfinite(f_red) else "none"
            mi_red = np.nan
            if msq is not None:
                mi_red = next((j for j in range(k + 1, min(T, k + 80)) if msq[j] in STOP), np.nan)
            if not np.isfinite(f_red) and np.isfinite(mi_red):
                f_red, t_r_source = mi_red, "michigan"
            yellow_s = (f_red - k) * DT if np.isfinite(f_red) else np.nan
            seen = {}
            for ent, poly in chains:
                p, s, tang = lane_frame(poly)
                s_stop = s[-1]
                cands = []
                for tr in tracks:
                    if not tr["valid"][k]:
                        continue
                    ss, lat, tg = project(p, s, tang, [[tr["x"][k], tr["y"][k]]])
                    d = s_stop - ss[0]
                    if abs(lat[0]) > LAT_M or d <= 0 or d > UP_MAX:
                        continue
                    if np.cos(tr["hd"][k]) * tg[0][0] + np.sin(tr["hd"][k]) * tg[0][1] < np.cos(np.radians(45)):
                        continue
                    cands.append((d, tr, lat[0]))
                if not cands:
                    continue
                cands.sort(key=lambda c: c[0])
                d0, tr, lat0 = cands[0]
                if tr["id"] in seen and seen[tr["id"]] <= d0:
                    continue
                seen[tr["id"]] = d0
                v0 = tr["v"][k]
                tt = d0 / max(v0, 0.1)
                label, t_label, f_cross = -1, np.nan, np.nan
                for j in range(k + 1, T):
                    if not tr["valid"][j]:
                        break
                    sj = project(p, s, tang, [[tr["x"][j], tr["y"][j]]])[0][0]
                    if s_stop - sj <= 0:
                        label, t_label, f_cross = 1, (j - k) * DT, j
                        break
                    if tr["v"][j] < V_STOP:
                        label, t_label = 0, (j - k) * DT
                        break
                # end-of-window state (lets labels be redefined later without re-extraction)
                jl = k
                while jl + 1 < T and tr["valid"][jl + 1]:
                    jl += 1
                if label == 1:
                    jl = int(f_cross)
                s_end = project(p, s, tang, [[tr["x"][jl], tr["y"][jl]]])[0][0]
                d_end, v_end = float(s_stop - s_end), float(tr["v"][jl])
                a_end = float(tr["a"][jl]) if np.isfinite(tr["a"][jl]) else np.nan
                # relaxed stop: still upstream at the window end, slow and braking (v < 3 m/s, a < -0.5 m/s2) -> stop
                label_relaxed = label if label >= 0 else (0 if (d_end > 0 and v_end < 3.0 and np.isfinite(a_end) and a_end < -0.5) else -1)
                hist = tr["valid"][max(0, k - HIST + 1):k + 1].all() and k >= HIST - 1
                # features and 2 s window at each lead time (k + L); lead 0 first
                ev_i = len(ev_rows)
                for L in leads:
                    kL = k + int(round(L / DT))
                    if kL >= T or not tr["valid"][kL]:
                        feat_rows.append(dict(ev=ev_i, lead_s=L, d=np.nan, v=np.nan, a=np.nan, a_raw=np.nan, t_r=np.nan, tt=np.nan, dreq=np.nan, crossed=np.nan,
                                              lead_present=0, lead_gap=np.nan, lead_dv=np.nan, lane=0, clamped=1, ldr_id=-1, ldr_d=np.nan, ldr_gap=np.nan,
                                              ldr_dv=np.nan, ldr_a=np.nan, ldr_label=np.nan, pad=HIST, lat=np.nan, n_lead=0))
                        seqs.setdefault(L, []).append(np.zeros((HIST, 4), np.float32))
                        continue
                    kk = np.clip(np.arange(kL - HIST + 1, kL + 1), 0, T - 1)
                    ss, lat, _ = project(p, s, tang, np.c_[tr["x"][kk], tr["y"][kk]])
                    a_seq = np.where(np.isfinite(tr["a"][kk]), tr["a"][kk], 0.0)
                    seqs.setdefault(L, []).append(np.stack([-(s_stop - ss), lat, tr["v"][kk], a_seq], -1).astype(np.float32))
                    dL, vL = float(s_stop - ss[-1]), float(tr["v"][kL])
                    aL = tr["a"][kL] if np.isfinite(tr["a"][kL]) else 0.0
                    feat_rows.append(dict(ev=ev_i, lead_s=L, d=dL, v=vL, a=float(aL), a_raw=float(tr["a_raw"][kL]) if np.isfinite(tr["a_raw"][kL]) else np.nan,
                                          t_r=(f_red - kL) * DT if np.isfinite(f_red) else np.nan, tt=dL / max(vL, 0.1), dreq=vL ** 2 / (2 * max(dL, 0.1)),
                                          crossed=int(dL <= 0), lead_present=0, lead_gap=np.nan, lead_dv=np.nan, lane=0,
                                          clamped=int(not (tr["valid"][max(0, kL - HIST + 1):kL + 1].all() and kL >= HIST - 1)), ldr_id=-1, ldr_d=np.nan,
                                          ldr_gap=np.nan, ldr_dv=np.nan, ldr_a=np.nan, ldr_label=np.nan, pad=int(max(0, HIST - 1 - kL)), lat=float(lat[-1]),
                                          n_lead=sum(1 for c in cands[1:] if c[0] < d0)))
                ev_rows.append(dict(scenario=win["id"], windows=win["n_windows"], source=source, lane=lid, entry_lane=ent, k0=k, f_yellow=k, f_red=f_red,
                                    yellow_s=yellow_s, end_state=end_state, t_r_source=t_r_source, mi_red_s=(mi_red - k) * DT if np.isfinite(mi_red) else np.nan,
                                    arrow=bool(arrow), turn_deg=turn,
                                    movement="through" if abs(turn) < 30 else ("left" if turn > 0 else "right"), track_id=int(tr["id"]),
                                    label=label, label_relaxed=label_relaxed, t_label=t_label, d_end=d_end, v_end=v_end, a_end=a_end, t_end=(jl - k) * DT,
                                    hist_ok=bool(hist), t_left=(T - 1 - k) * DT, stop_x=sp[0], stop_y=sp[1],
                                    n_on_chain=len(cands), approach_heading=float(np.degrees(np.arctan2(tang[-1][1], tang[-1][0])))))
    return ev_rows, feat_rows, seqs


def extract_shard(path, michigan_idx=None, stitch=True, leads=LEADS):
    """All windows of a shard -> (events df, features df, sequences (N,20,4), meta dict, accel df, stitch stats)."""
    scs = [parse(r) for r in records(path)]
    meta = dict(shard=os.path.basename(path), n_scenarios=len(scs), n_signal=sum(1 for s in scs if s["signals"]),
                n_michigan=0, n_groups=0, n_multi=0, max_windows=1, yellow_end_orig={}, yellow_end_used={})
    end_orig, end_used = {}, {}

    def ends(sig, T):
        c = {}
        for sq, _ in sig.values():
            for k in range(1, T):
                if sq[k] in CAUTION and sq[k - 1] in GO:
                    e = next((int(sq[j]) for j in range(k + 1, T) if sq[j] not in CAUTION), -1)
                    c[e] = c.get(e, 0) + 1
        return c

    for s in scs:
        for e, n in ends(s["signals"], s["T"]).items():
            end_orig[e] = end_orig.get(e, 0) + n
        if michigan_idx and s["id"] in michigan_idx:
            mi = michigan_signals(michigan_idx[s["id"]], s["T"])
            if mi:
                s["signals_mi"] = mi
                meta["n_michigan"] += 1
        for e, n in ends(s.get("signals_mi", s["signals"]), s["T"]).items():
            end_used[e] = end_used.get(e, 0) + n
    meta["yellow_end_orig"], meta["yellow_end_used"] = end_orig, end_used
    groups = stitch_groups(scs) if stitch else [([i], {i: 0}) for i in range(len(scs))]
    meta["n_groups"] = len(groups)
    meta["n_multi"] = sum(1 for g, _ in groups if len(g) > 1)
    meta["max_windows"] = max(len(g) for g, _ in groups)
    E, F, S, A = [], [], {}, []
    for mem, offs in groups:
        win = merge_group(scs, mem, offs)
        e, f, q = extract_window(win, "michigan_fill" if any(("signals_mi" in scs[m]) for m in mem) else "original", A, leads)
        for r in f:
            r["ev"] += len(E)
        E += e
        F += f
        for L in leads:
            S.setdefault(L, []).extend(q.get(L, []))
    ev = pd.DataFrame(E)
    feat = pd.DataFrame(F)
    seq = {L: (np.stack(S[L]) if S.get(L) else np.zeros((0, HIST, 4), np.float32)) for L in leads}
    return ev, feat, seq, meta, pd.DataFrame(A)
