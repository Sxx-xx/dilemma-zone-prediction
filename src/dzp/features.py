"""Per-event features at an arbitrary prediction time (yellow onset + lead), dataset-independent.

Works on any sind.Recording-shaped data (SinD, FLUID, Waterloo adapters) plus the event table of
events.extract_events (needs columns track_id, sid, k0, f_red).

Instantaneous features at the prediction frame k = k0 + round(lead / dt):
  d (front bumper distance to the stop line, m, >0 upstream), v (m/s), a (longitudinal, m/s^2),
  t_r (time to red from the signal, s), tt = d / v, dreq = v^2 / (2 d) (deceleration needed to stop at the line),
  lead vehicle in the same lane ahead (present, gap = rear-of-lead to front-of-subject in m, dv = v - v_lead),
  lane (index across the stop line from its p0 end, lane width LANE_W - an estimate, maps have no lane ids here).
Leader (ldr_*, the leader/follower split): the nearest motor vehicle ahead in the same lane band that has not yet
  crossed the stop line (front d > 0). If the nearest vehicle ahead is already past the line, or there is none, the
  subject is a platoon leader (ldr_id = -1). ldr_gap = rear-of-leader to front-of-subject (m), ldr_dv = v - v_leader,
  ldr_a = leader's longitudinal acceleration, ldr_label = leader's own stop/go outcome for this yellow
  (events.label_event on its trajectory; NaN if unresolved).
Sequence: the HIST_STEPS frames ending at k of (x, y, v, a) in the approach frame:
  x = -d (longitudinal, 0 at the stop line, negative upstream), y = lateral offset along the stop line.
Frames before the first observation are padded with the first observed value; `pad` counts them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import sind
from .events import label_event
from .geometry import Approach, project

HIST_STEPS = 20          # 2 s at 10 Hz
LANE_W = 3.5             # m, for the lane index estimate
LEAD_MAX = 80.0          # m, lead search range ahead of the subject
NO_LEADER = dict(ldr_id=-1, ldr_d=np.nan, ldr_gap=np.nan, ldr_dv=np.nan, ldr_a=np.nan, ldr_label=np.nan)


def _front(tr: pd.DataFrame):
    fx = tr["x"].values + 0.5 * tr["length"].values * np.cos(tr["heading_rad"].values)
    fy = tr["y"].values + 0.5 * tr["length"].values * np.sin(tr["heading_rad"].values)
    return fx, fy


def _leader(o, od, olat, vn, dj, latj, by_track, ap, f_red, dt):
    """Leader among the other agents `o` at the subject's frame (see module doc); returns a dict of ldr_* columns."""
    out = dict(NO_LEADER)
    m = (od < dj - 0.5) & (od > dj - LEAD_MAX) & (np.abs(olat - latj) < LANE_W / 2) & (vn > -0.5) \
        & o["agent_type"].isin(sind.MOTOR_VEHICLES).values
    if not m.any():
        return out
    q = np.argmax(np.where(m, od, -np.inf))
    if od[q] <= 0:                                   # nearest ahead already past the stop line -> leader
        return out
    lt = by_track[o["track_id"].values[q]]
    lfr = lt["frame_id"].values
    ld, _ = project(ap, lt["fx"].values, lt["fy"].values)
    i = int(np.searchsorted(lfr, o["frame_id"].values[q]))
    past = np.where((ld <= 0) & (np.arange(len(lfr)) >= i))[0]
    res = label_event(lfr, ld, lt["speed"].values, i, lfr[past[0]] if len(past) else np.nan, f_red, np.nan, dt)
    out.update(ldr_id=int(o["track_id"].values[q]), ldr_d=float(od[q]), ldr_gap=float(dj - (od[q] + o["length"].values[q])),
               ldr_dv=float(o["speed"].values[q]), ldr_a=float(o["a_lon"].values[q]),
               ldr_label=np.nan if res is None else float(res[0]))
    return out


def build(rec, ev: pd.DataFrame, aps: list[Approach], leads=(0.0,), dt: float = 0.1001):
    """Returns (feat, seqs): feat = long table (row = event index, lead); seqs = {lead: (N, HIST_STEPS, 4)}."""
    t = rec.tracks
    fx, fy = _front(t)
    t = t.assign(fx=fx, fy=fy, speed=np.hypot(t["vx"].values, t["vy"].values))
    by_track = {tid: g for tid, g in t.groupby("track_id")}
    by_frame = t.groupby("frame_id").indices
    rows, seqs = [], {L: [] for L in leads}
    for i, e in enumerate(ev.itertuples(index=False)):
        ap = aps[e.sid]
        g = by_track[e.track_id]
        fr = g["frame_id"].values
        d, lat = project(ap, g["fx"].values, g["fy"].values)
        v, a = g["speed"].values, g["a_lon"].values
        lane0 = (lat + ap.half_len) // LANE_W
        for L in leads:
            k = int(e.k0) + int(round(L / dt))
            j = int(np.clip(k - fr[0], 0, len(fr) - 1))
            clamped = int(k - fr[0] != j)
            dj, vj = float(d[j]), float(v[j])
            # lead vehicle at the same frame, same lane band, ahead (smaller d)
            present, gap, dv = 0, np.nan, np.nan
            ldr = dict(NO_LEADER)
            idx = by_frame.get(fr[j])
            if idx is not None:
                o = t.iloc[idx]
                o = o[o["track_id"] != e.track_id]
                od, olat = project(ap, o["fx"].values, o["fy"].values)
                vn = o["vx"].values * ap.n[0] + o["vy"].values * ap.n[1]
                m = (od < dj - 0.5) & (od > dj - LEAD_MAX) & (np.abs(olat - lat[j]) < LANE_W / 2) & (vn > -0.5)
                if m.any():
                    q = np.argmax(np.where(m, od, -np.inf))          # nearest ahead
                    present = 1
                    gap = dj - (od[q] + o["length"].values[q])
                    dv = vj - float(o["speed"].values[q])
                ldr = _leader(o, od, olat, vn, dj, lat[j], by_track, ap, e.f_red, dt)
                ldr["ldr_dv"] = vj - ldr["ldr_dv"]
            rows.append(dict(ev=i, lead_s=L, d=dj, v=vj, a=float(a[j]), t_r=(e.f_red - fr[j]) * dt,
                             tt=dj / max(vj, 0.1), dreq=vj ** 2 / (2 * max(dj, 0.1)) if dj > 0 else np.inf,
                             lead_present=present, lead_gap=gap, lead_dv=dv, lane=int(lane0[j]), clamped=clamped, **ldr))
            jj = np.clip(np.arange(k - HIST_STEPS + 1, k + 1) - fr[0], 0, len(fr) - 1)
            seqs[L].append(np.stack([-d[jj], lat[jj], v[jj], a[jj]], -1))
            rows[-1]["pad"] = int(np.sum(np.arange(k - HIST_STEPS + 1, k + 1) - fr[0] < 0))
    feat = pd.DataFrame(rows)
    return feat, {L: np.asarray(s, dtype=np.float32).reshape(-1, HIST_STEPS, 4) for L, s in seqs.items()}
