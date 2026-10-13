"""Review aids for the V2X-Seq check run (scripts/v2xseq_check.py output).

  --sample N   : draw N/2 stop + N/2 go through events (seed), plot d(t), v(t) and the light for each (one PNG
                 per event + a contact sheet) and print a label audit line per event (halt / crossing facts
                 recomputed from the trajectory, independent of dzp.events)
  --table      : per-site table (segments, hours, onsets, events by role/label, red-crossing, visibility, pass)
  --audit-all  : recompute, for every through event, crossing speed / halt / min d after onset from the trajectory
                 (events_audit.csv) -- flags 'go' labels that are slow creeps over a mis-placed line

  .venv/bin/python scripts/v2xseq_review.py --root data/raw/v2xseq/full/extracted/V2X-Seq-TFD \
      --check data/raw/v2xseq/results/check --out results/v2xseq/review --sample 20 --table
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dzp import events, sind, v2xseq
from dzp.geometry import build_approaches, project

RED_MAX, MIN_CROSS, DZ_HI = 0.08, 30, 5.5


def plot_event(root, lanes, seg, e, out_png, shift=None):
    rec = v2xseq.load_recording(root, seg, lanes, shift_table=shift)
    aps = build_approaches(rec.map.stop_lines())
    ap = aps[int(e.sid)]
    g = rec.tracks[rec.tracks["track_id"] == e.track_id].sort_values("frame_id")
    if g.empty:
        raise KeyError(f"track {e.track_id} not in {seg.name} (stale check output?)")
    fx = g["x"].values + 0.5 * g["length"].values * np.cos(g["heading_rad"].values)
    fy = g["y"].values + 0.5 * g["length"].values * np.sin(g["heading_rad"].values)
    d, lat = project(ap, fx, fy)
    v = np.hypot(g["vx"].values, g["vy"].values)
    tt = (g["frame_id"].values - e.f_yellow) * v2xseq.DT
    col = rec.light_map[int(e.sid)]
    st = rec.light_state_at(col, g["frame_id"].values.astype(float))
    fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    ax[0].plot(tt, d, "k.-", ms=3)
    ax[0].axhline(0, color="r", lw=1)
    ax[0].axvline(0, color="orange", lw=1)
    ax[0].set_ylabel("front d to stop line [m]")
    ax[1].plot(tt, v, "b.-", ms=3)
    ax[1].axhline(events.V_STOP, color="gray", ls="--", lw=1)
    ax[1].axvline(0, color="orange", lw=1)
    ax[1].set_ylabel("speed [m/s]")
    cmap = {sind.RED: "red", sind.GREEN: "green", sind.YELLOW: "gold", -1: "lightgray"}
    ax[2].scatter(tt, np.zeros_like(tt), c=[cmap[int(s)] for s in st], s=12)
    ax[2].set_yticks([])
    ax[2].set_xlabel("time from yellow onset [s]")
    lab = "go" if e.label == 1 else "stop"
    ax[0].set_title(f"{seg.name} track {e.track_id} sid {int(e.sid)} {ap.name()}  label={lab} ({e.cross_phase})  "
                    f"d0={e.d0:.1f} v0={e.v0:.1f} t_r={e.t_r:.1f}  {'leader' if e.ldr_id < 0 else 'follower'}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    # independent audit: halt upstream after onset? crossing time? track end relative to red
    after = g["frame_id"].values >= e.k0
    halt = bool(np.any((v[after] < events.V_STOP) & (d[after] > 0)))
    crossed = np.where((d[:-1] > 0) & (d[1:] <= 0) & after[1:])[0]
    t_cross = tt[crossed[0] + 1] if len(crossed) else np.nan
    return dict(halt_after_onset=halt, t_cross_s=t_cross, t_end_s=tt[-1], red_at_s=(e.f_red - e.f_yellow) * v2xseq.DT,
                lat_at_onset=float(lat[np.searchsorted(g["frame_id"].values, e.k0)]), min_d_after=float(d[after].min()))


def _pos_clusters(xy, names, radius=25.0):
    keys, centers = [], []
    for p, nm in zip(xy, names):
        j = next((k for k, (c, n) in enumerate(centers) if n == nm and np.hypot(*(c - p)) <= radius), None)
        if j is None:
            centers.append((p.copy(), nm))
            j = len(centers) - 1
        keys.append(f"{nm}_{j}")
    return keys


def audit_all(root, check, out, cache, shift=None):
    E = pd.read_csv(f"{check}/events_all.csv")
    E = E[E["movement"] == "through"]
    lanes = v2xseq.lane_index(root, cache)
    segs = {s.name: s for s in v2xseq.segments(root, cache)}
    rows = []
    for name, g in E.groupby("group"):
        rec = v2xseq.load_recording(root, segs[name], lanes, shift_table=shift)
        aps = build_approaches(rec.map.stop_lines())
        t = rec.tracks
        for e in g.itertuples(index=False):
            k = t[t["track_id"] == e.track_id].sort_values("frame_id")
            ap = aps[int(e.sid)]
            fx = k["x"].values + 0.5 * k["length"].values * np.cos(k["heading_rad"].values)
            fy = k["y"].values + 0.5 * k["length"].values * np.sin(k["heading_rad"].values)
            d, _ = project(ap, fx, fy)
            v = np.hypot(k["vx"].values, k["vy"].values)
            after = k["frame_id"].values >= e.k0
            cr = np.where((d[:-1] > 0) & (d[1:] <= 0) & after[1:])[0]
            rows.append(dict(group=name, track_id=e.track_id, label=int(e.label), phase=e.cross_phase, leader=bool(e.ldr_id < 0),
                             d0=e.d0, v0=e.v0, t_r=e.t_r, min_d_after=float(d[after].min()), min_v_after=float(v[after].min()),
                             v_cross=float(v[cr[0] + 1]) if len(cr) else np.nan,
                             t_cross=float((k["frame_id"].values[cr[0] + 1] - e.f_yellow) * v2xseq.DT) if len(cr) else np.nan,
                             t_end=float((k["frame_id"].values[-1] - e.f_yellow) * v2xseq.DT),
                             halt_after=bool(np.any((v[after] < events.V_STOP) & (d[after] > 0)))))
    A = pd.DataFrame(rows)
    A.to_csv(f"{out}/events_audit.csv", index=False)
    return A


def site_table(check, out):
    S = pd.read_csv(f"{check}/segments.csv")
    A = pd.read_csv(f"{check}/approaches.csv")
    O = pd.read_csv(f"{check}/onsets.csv")
    E = pd.read_csv(f"{check}/events_all.csv")
    V = pd.read_json(f"{check}/visibility_samples.json")
    E = E[E["movement"] == "through"].copy()
    E["leader"] = E["ldr_id"] < 0
    rows = []
    for site, s in S.groupby("inter"):
        a = A[A["inter"] == site]
        # approaches are numbered per segment: aggregate by stop-line position (greedy clusters, 25 m) instead
        a = a.assign(key=_pos_clusters(a[["mid_x", "mid_y"]].values, a["name"].values))
        ap_rows = []
        for key, g in a.groupby("key"):
            vs = V[(V["inter"] == site) & V["segment"].isin(g["segment"]) & V["sid"].isin(g["sid"])]
            vs = vs.merge(g[["segment", "sid"]], on=["segment", "sid"])
            dd = np.concatenate([np.array(x) for x in vs["d"]]) if len(vs) else np.array([])
            vv = np.concatenate([np.array(x) for x in vs["v"]]) if len(vs) else np.array([])
            vis = float(np.quantile(dd, 0.99)) if len(dd) else np.nan
            va = float(vv.mean()) if len(vv) else np.nan
            ap_rows.append(dict(site=site, key=key, name=g["name"].mode().iloc[0], segs=len(g), lanes=int(g["lanes"].max()),
                                cross_through=int(g["cross_through"].sum()), cross_red=int(g["cross_red"].sum()),
                                cross_known=int(g["cross_known"].sum()), visible_m=vis, v_app=va,
                                passes=bool(np.isfinite(va) and vis >= DZ_HI * va and g["cross_through"].sum() >= MIN_CROSS)))
        ap_df = pd.DataFrame(ap_rows, columns=["site", "key", "name", "segs", "lanes", "cross_through", "cross_red", "cross_known",
                                               "visible_m", "v_app", "passes"])
        e = E[E["intersection"] == site]
        o = O[O["inter"] == site]
        red = ap_df["cross_red"].sum() / max(ap_df["cross_known"].sum(), 1)
        rows.append(dict(site=site, segments=len(s), scenes=int(s["n_scenes"].sum()), hours=round(s["dur_s"].sum() / 3600, 2),
                         approaches=len(ap_df), approaches_pass=int(ap_df["passes"].sum()),
                         through_onsets=len(o), onsets_phase=int(np.sum(np.diff(np.sort(o["t_abs"].values)) > 1.0) + (len(o) > 0)),
                         candidates=int(s["n_candidates"].sum()), short_hist=int(s["n_short_history"].sum()),
                         unresolved=int(s["n_unresolved"].sum()),
                         events=len(e), stop=int((e["label"] == 0).sum()), go=int(e["label"].sum()),
                         leader=int(e["leader"].sum()), leader_go=int(e[e["leader"]]["label"].sum()),
                         follower=int((~e["leader"]).sum()), follower_go=int(e[~e["leader"]]["label"].sum()),
                         red_rate=round(red, 3), sync_pass=red <= RED_MAX,
                         visible_m=round(ap_df["visible_m"].median(), 1) if len(ap_df) else np.nan,
                         v_app=round(ap_df["v_app"].median(), 1) if len(ap_df) else np.nan,
                         events_pass_ap=int(e["sid"].notna().sum()) if len(ap_df) else 0))
        ap_df.to_csv(f"{out}/approaches_{site.replace('#', '_')}.csv", index=False)
    T = pd.DataFrame(rows).sort_values("events", ascending=False)
    T.to_csv(f"{out}/per_site.csv", index=False)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--check", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="data/raw/v2xseq/results/cache")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--audit-all", action="store_true")
    ap.add_argument("--shift-table", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 250)
    if args.table:
        print(site_table(args.check, args.out).to_string(index=False))
    shift = v2xseq.load_shift_table(args.shift_table)
    if args.audit_all:
        A = audit_all(args.root, args.check, args.out, args.cache, shift)
        go = A[A.label == 1]
        print(f"through events {len(A)}; go {len(go)}: crossing speed < 2 m/s {int((go.v_cross < 2).sum())}, "
              f"crossed after red {int((go.phase == 'red').sum())}; stop {int((A.label == 0).sum())}: "
              f"halt beyond the line later (min_d_after < 0) {int(((A.label == 0) & (A.min_d_after < 0)).sum())}")
    if args.sample:
        E = pd.read_csv(f"{args.check}/events_all.csv")
        E = E[E["movement"] == "through"]
        rng = np.random.default_rng(args.seed)
        pick = pd.concat([E[E["label"] == 0].sample(args.sample // 2, random_state=args.seed),
                          E[E["label"] == 1].sample(args.sample - args.sample // 2, random_state=args.seed)])
        lanes = v2xseq.lane_index(args.root, args.cache)
        segs = {s.name: s for s in v2xseq.segments(args.root, args.cache)}
        os.makedirs(f"{args.out}/events", exist_ok=True)
        audit = []
        for i, e in enumerate(pick.itertuples(index=False)):
            png = f"{args.out}/events/{i:02d}_{'stop' if e.label == 0 else 'go'}_{e.track_id}.png"
            a = plot_event(args.root, lanes, segs[e.group], e, png, shift)
            audit.append(dict(i=i, group=e.group, track_id=e.track_id, label=int(e.label), phase=e.cross_phase, d0=round(e.d0, 1),
                              v0=round(e.v0, 1), t_r=round(e.t_r, 1), leader=bool(e.ldr_id < 0), **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items()}))
        au = pd.DataFrame(audit)
        au.to_csv(f"{args.out}/sample_audit.csv", index=False)
        print(au.to_string(index=False))


if __name__ == "__main__":
    main()
