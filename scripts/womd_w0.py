"""W0: Waymo Open Motion Dataset v1.3.1 feasibility check on a few scenario shards (no TensorFlow; trimmed protos
compiled under src/womd_protos, TFRecord read directly).

Per scenario (91 steps at 10 Hz, 1 s history + 8 s future): signal lanes from dynamic_map_states; yellow onset =
transition into CAUTION / ARROW_CAUTION from a non-caution known state; stop line = the lane's signal stop_point;
vehicles on the upstream chain of entry lanes (the signalled lane starts at its stop point), |lateral| <= LAT_M, heading
within 45 deg; leader = nearest to the stop point per approach lane; movement from the signalled lane's own turn angle. Labels until the scene end: go = crosses the stop point, stop = speed < V_STOP
while still upstream, else unresolved. Outputs under --out: onsets.csv, events.csv, yellow_durations.csv,
accel_quality.csv, stop_point_check.csv, summary.json.

  .venv/bin/python scripts/womd_w0.py --shards data/raw/womd/sample/training.tfrecord-0000*-of-01000 --out results/womd/w0
"""
import argparse
import glob
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "womd_protos"))

import numpy as np
import pandas as pd

from waymo_open_dataset.protos import scenario_pb2

DT = 0.1
CAUTION = {2, 5}
GO = {3, 6}
STOP = {1, 4}
LAT_M, V_STOP, UP_MAX = 2.5, 0.5, 120.0
HIST = 20


def records(path):
    with open(path, "rb") as f:
        while True:
            h = f.read(12)
            if len(h) < 12:
                return
            n = struct.unpack("<Q", h[:8])[0]
            yield f.read(n)
            f.read(4)


def lane_frame(poly):
    """Arc length and unit tangent along a lane polyline."""
    p = np.asarray(poly, float)
    seg = np.diff(p, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    s = np.r_[0, np.cumsum(L)]
    return p, s, seg / np.maximum(L, 1e-9)[:, None]


def project_to_lane(p, s, tang, xy):
    """Nearest-segment projection: (arc length, signed lateral, tangent at the point) for each row of xy."""
    a, b = p[:-1], p[1:]
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(1), 1e-9)
    out_s, out_lat, out_t = np.empty(len(xy)), np.empty(len(xy)), np.empty((len(xy), 2))
    for i, q in enumerate(xy):
        t = np.clip(((q - a) * ab).sum(1) / ab2, 0, 1)
        proj = a + t[:, None] * ab
        d2 = ((q - proj) ** 2).sum(1)
        j = int(np.argmin(d2))
        out_s[i] = s[j] + t[j] * np.sqrt(ab2[j])
        out_lat[i] = tang[j][0] * (q - proj[j])[1] - tang[j][1] * (q - proj[j])[0]   # 2-D cross (numpy 2 dropped it)
        out_t[i] = tang[j]
    return out_s, out_lat, out_t


def upstream_chains(lanes, lid, max_len=UP_MAX, depth=4):
    """Polylines ending at the start of lane `lid` (= its signal stop point), built from entry_lanes recursively.
    Returns [(entry_lane_id, polyline array)] one per distinct first-upstream lane, longest chain per branch."""
    out = []

    def walk(cur, pts, length, d):
        ents = [e for e in lanes[cur].entry_lanes if e in lanes and len(lanes[e].polyline) >= 2]
        if not ents or length >= max_len or d >= depth:
            out.append((pts, length))
            return
        for e in ents:
            q = np.array([(m.x, m.y) for m in lanes[e].polyline])
            seg = np.hypot(*np.diff(q, axis=0).T).sum()
            walk(e, [q] + pts, length + seg, d + 1)

    for e in lanes[lid].entry_lanes:
        if e in lanes and len(lanes[e].polyline) >= 2:
            q = np.array([(m.x, m.y) for m in lanes[e].polyline])
            out_e = []
            # collect chains starting from this entry lane
            out.clear()
            walk(e, [q], np.hypot(*np.diff(q, axis=0).T).sum(), 1)
            for pts, length in out:
                poly = np.vstack([pts[0]] + [p[1:] for p in pts[1:]])
                out_e.append((length, poly))
            if out_e:
                out_e.sort(key=lambda t: -t[0])
                yield e, out_e[0][1]


def lane_turn_deg(lane):
    p = np.array([(m.x, m.y) for m in lane.polyline])
    if len(p) < 3:
        return np.nan
    t0, t1 = p[1] - p[0], p[-1] - p[-2]
    return float(np.degrees(np.arctan2(t0[0] * t1[1] - t0[1] * t1[0], t0 @ t1)))


def analyse(path, shard_name):
    onsets, events, ydur, accel, spc = [], [], [], [], []
    n_sc = n_signal_sc = 0
    state_counts = np.zeros(9, int)
    for raw in records(path):
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw)
        n_sc += 1
        lanes = {mf.id: mf.lane for mf in sc.map_features if mf.WhichOneof("feature_data") == "lane"}
        T = len(sc.timestamps_seconds)
        # signal state per lane per step
        seq, stop_pt = {}, {}
        for k, dm in enumerate(sc.dynamic_map_states):
            for ls in dm.lane_states:
                seq.setdefault(ls.lane, np.zeros(T, int))[k] = ls.state
                stop_pt[ls.lane] = (ls.stop_point.x, ls.stop_point.y)
                state_counts[ls.state] += 1
        if not seq:
            continue
        n_signal_sc += 1
        # vehicle arrays
        veh = []
        for ti, tr in enumerate(sc.tracks):
            if tr.object_type != 1:
                continue
            st = tr.states
            valid = np.array([s.valid for s in st])
            x = np.array([s.center_x for s in st]); y = np.array([s.center_y for s in st])
            vx = np.array([s.velocity_x for s in st]); vy = np.array([s.velocity_y for s in st])
            hd = np.array([s.heading for s in st]); ln = np.array([s.length for s in st])
            veh.append(dict(i=ti, id=tr.id, valid=valid, x=x, y=y, vx=vx, vy=vy, hd=hd, length=ln))
            v = np.hypot(vx, vy)
            a = np.full(T, np.nan)
            ok = valid[1:] & valid[:-1]
            a[1:][ok] = (v[1:][ok] - v[:-1][ok]) / DT
            da = np.abs(np.diff(a))                      # length T-1, da[i] = |a[i+1] - a[i]|
            m = np.isfinite(da) & (v[1:] > 2)
            if m.sum() >= 10:
                accel.append(dict(scenario=sc.scenario_id, track=tr.id, n=int(m.sum()), da_abs_mean=float(da[m].mean()), a_std=float(np.nanstd(a[v > 2])),
                                  dv_abs_mean=float(np.abs(np.diff(v))[m].mean())))
        # stop point vs lane polyline end
        for lid, sp in stop_pt.items():
            if lid in lanes and len(lanes[lid].polyline) >= 2:
                p = lanes[lid].polyline
                e = np.array([p[-1].x, p[-1].y]); b = np.array([p[0].x, p[0].y])
                spc.append(dict(scenario=sc.scenario_id, lane=lid, d_end=float(np.hypot(*(e - np.array(sp)))), d_start=float(np.hypot(*(b - np.array(sp)))), n_pts=len(p)))
        for lid, sq in seq.items():
            known = sq != 0
            caution = np.isin(sq, list(CAUTION))
            # onsets: step k with caution and previous known non-caution
            for k in range(1, T):
                if caution[k] and known[k - 1] and not caution[k - 1]:
                    prev_go = sq[k - 1] in GO
                    # yellow end
                    end = next((j for j in range(k + 1, T) if not caution[j]), None)
                    end_state = int(sq[end]) if end is not None else -1
                    onsets.append(dict(shard=shard_name, scenario=sc.scenario_id, lane=lid, k=k, t_left=(T - 1 - k) * DT, prev_state=int(sq[k - 1]),
                                       prev_go=prev_go, end_visible=end is not None, end_state=end_state, history_ok=k >= HIST))
                    if end is not None and end_state in STOP:
                        ydur.append(dict(shard=shard_name, scenario=sc.scenario_id, lane=lid, yellow_s=(end - k) * DT))
                    if lid not in lanes or len(lanes[lid].polyline) < 2:
                        continue
                    turn = lane_turn_deg(lanes[lid])
                    sp = np.array(stop_pt[lid])
                    seen = {}
                    for ent, poly in upstream_chains(lanes, lid):
                        p, s, tang = lane_frame(poly)
                        s_stop = s[-1]
                        stop_err = float(np.hypot(*(p[-1] - sp)))
                        cands = []
                        for v in veh:
                            if not v["valid"][k]:
                                continue
                            ss, lat, tg = project_to_lane(p, s, tang, np.array([[v["x"][k], v["y"][k]]]))
                            d = s_stop - ss[0]
                            if abs(lat[0]) > LAT_M or d <= 0 or d > UP_MAX:
                                continue
                            hv = np.array([np.cos(v["hd"][k]), np.sin(v["hd"][k])])
                            if hv @ tg[0] < np.cos(np.radians(45)):
                                continue
                            cands.append((d, v, lat[0]))
                        if not cands:
                            continue
                        cands.sort(key=lambda c: c[0])
                        d0, v, lat0 = cands[0]
                        if v["id"] in seen and seen[v["id"]] <= d0:
                            continue
                        seen[v["id"]] = d0
                        spd = np.hypot(v["vx"], v["vy"])
                        label, t_label = "unresolved", np.nan
                        for j in range(k + 1, T):
                            if not v["valid"][j]:
                                break
                            sj = project_to_lane(p, s, tang, np.array([[v["x"][j], v["y"][j]]]))[0][0]
                            if s_stop - sj <= 0:
                                label, t_label = "go", (j - k) * DT
                                break
                            if spd[j] < V_STOP:
                                label, t_label = "stop", (j - k) * DT
                                break
                        hist_valid = bool(v["valid"][max(0, k - HIST + 1):k + 1].all()) and k >= HIST - 1
                        events.append(dict(shard=shard_name, scenario=sc.scenario_id, lane=lid, entry_lane=ent, k=k, track=v["id"], d=d0, v=float(spd[k]), lat=lat0,
                                           n_on_lane=len(cands), label=label, t_label=t_label, hist_valid=hist_valid, t_left=(T - 1 - k) * DT, prev_go=prev_go,
                                           lane_turn_deg=turn, chain_len=float(s_stop), stop_err=stop_err))
    return dict(n_scenarios=n_sc, n_signal_scenarios=n_signal_sc, state_counts=state_counts.tolist()), onsets, events, ydur, accel, spc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=sorted(glob.glob("data/raw/womd/sample/training.tfrecord-*")))
    ap.add_argument("--out", default="results/womd/w0")
    ap.add_argument("--n-train-shards", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tot = dict(n_scenarios=0, n_signal_scenarios=0, state_counts=np.zeros(9, int))
    O, E, Y, A, S = [], [], [], [], []
    for sh in args.shards:
        name = os.path.basename(sh)
        meta, o, e, y, a, s = analyse(sh, name)
        print(name, meta, "onsets", len(o), "events", len(e), flush=True)
        tot["n_scenarios"] += meta["n_scenarios"]; tot["n_signal_scenarios"] += meta["n_signal_scenarios"]; tot["state_counts"] += np.array(meta["state_counts"])
        O += o; E += e; Y += y; A += a; S += s
    on, ev, yd, ac, sp = map(pd.DataFrame, (O, E, Y, A, S))
    for df, nm in ((on, "onsets"), (ev, "events"), (yd, "yellow_durations"), (ac, "accel_quality"), (sp, "stop_point_check")):
        df.to_csv(f"{args.out}/{nm}.csv", index=False)
    n_sc, n_sig = tot["n_scenarios"], tot["n_signal_scenarios"]
    sc_with_onset = on["scenario"].nunique() if len(on) else 0
    ev_ok = ev[ev["label"].isin(["stop", "go"])] if len(ev) else ev
    ev_ok_hist = ev_ok[ev_ok["hist_valid"]] if len(ev_ok) else ev_ok
    thr = ev_ok_hist[ev_ok_hist["lane_turn_deg"].abs() < 30] if len(ev_ok_hist) else ev_ok_hist
    states = ["UNKNOWN", "ARROW_STOP", "ARROW_CAUTION", "ARROW_GO", "STOP", "CAUTION", "GO", "FLASHING_STOP", "FLASHING_CAUTION"]
    sizes = [os.path.getsize(s) for s in args.shards]
    summ = dict(shards=[os.path.basename(s) for s in args.shards], shard_bytes=sizes, n_scenarios=int(n_sc), n_signal_scenarios=int(n_sig),
                lane_state_share={st: float(c / max(tot["state_counts"].sum(), 1)) for st, c in zip(states, tot["state_counts"])},
                n_onsets=int(len(on)), scenarios_with_onset=int(sc_with_onset), share_signal_scenarios_with_onset=float(sc_with_onset / max(n_sig, 1)),
                onsets_from_go=int(on["prev_go"].sum()) if len(on) else 0, onsets_end_visible=int(on["end_visible"].sum()) if len(on) else 0,
                onsets_history_ok=int(on["history_ok"].sum()) if len(on) else 0,
                yellow_s=dict(n=int(len(yd)), median=float(yd["yellow_s"].median()) if len(yd) else None, q10=float(yd["yellow_s"].quantile(.1)) if len(yd) else None,
                              q90=float(yd["yellow_s"].quantile(.9)) if len(yd) else None, counts=yd["yellow_s"].round(1).value_counts().sort_index().to_dict() if len(yd) else {}),
                events=dict(leaders_at_onset=int(len(ev)), labeled=int(len(ev_ok)), stop=int((ev_ok["label"] == "stop").sum()) if len(ev_ok) else 0,
                            go=int((ev_ok["label"] == "go").sum()) if len(ev_ok) else 0, unresolved=int((ev["label"] == "unresolved").sum()) if len(ev) else 0,
                            labeled_with_2s_history=int(len(ev_ok_hist)), through_or_stop_with_history=int(len(thr)),
                            go_through_with_history=int(((thr["label"] == "go")).sum()) if len(thr) else 0,
                            labeled_by_movement=ev_ok.assign(mv=np.where(ev_ok["lane_turn_deg"].abs() < 30, "through", np.where(ev_ok["lane_turn_deg"] > 0, "left", "right"))).groupby("mv")["label"].value_counts().rename(index=str).to_dict() if len(ev_ok) else {},
                            dz_band=int(((thr["d"] / thr["v"].clip(lower=0.1) >= 2.5) & (thr["d"] / thr["v"].clip(lower=0.1) <= 5.5)).sum()) if len(thr) else 0),
                per_scenario=dict(onsets=float(len(on) / max(n_sc, 1)), labeled_events=float(len(ev_ok) / max(n_sc, 1)), usable_events=float(len(thr) / max(n_sc, 1))),
                extrapolation=dict(train_shards=args.n_train_shards, scenarios_per_shard=float(n_sc / len(args.shards)),
                                   usable_events_training=float(len(thr) / len(args.shards) * args.n_train_shards),
                                   labeled_events_training=float(len(ev_ok) / len(args.shards) * args.n_train_shards),
                                   training_bytes=float(np.mean(sizes) * args.n_train_shards)),
                accel=dict(n_tracks=int(len(ac)), da_abs_mean=float(ac["da_abs_mean"].mean()) if len(ac) else None, a_std=float(ac["a_std"].mean()) if len(ac) else None,
                           dv_abs_mean=float(ac["dv_abs_mean"].mean()) if len(ac) else None),
                stop_point=dict(n=int(len(sp)), d_to_lane_end_median=float(sp["d_end"].median()) if len(sp) else None, d_to_lane_end_q90=float(sp["d_end"].quantile(.9)) if len(sp) else None,
                                share_within_1m_of_end=float((sp["d_end"] < 1).mean()) if len(sp) else None))
    summ["events"]["labeled_by_movement"] = {f"{k[0]}:{k[1]}": int(v) for k, v in summ["events"]["labeled_by_movement"].items()}
    json.dump(summ, open(f"{args.out}/summary.json", "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
