"""Feasibility checks on the AI Hub "AI 기반 신호 최적화를 위한 데이터" sample (data/raw/Sample).

Prints the numbers quoted in docs/E_aihub_check.md. The data itself is never written to results/
(AI Hub terms forbid redistribution).
"""
import glob
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1] / "data/raw/Sample/02.라벨링데이터"
VIDEO = ROOT / "영상데이터/교차로_14"
SIG = ROOT / "신호최적화데이터/부천시"
TRA = SIG / "교통량데이터(개별차량교통량기반)/교차로_all/20220810/90001220220810050000tra.csv"
NET = SIG / "네트워크데이터/1200012n.net.xml"
NET_OFFSET = (-300248.30, -4148110.73)   # <location netOffset> of NET


def video_labels():
    fs = sorted(VIDEO.glob("*.csv"))
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig").assign(src=f.name) for f in fs], ignore_index=True)
    df["t"] = df.departure_time.str.split(":").apply(lambda x: int(x[0]) * 60 + float(x[1]))
    fn = df.photo_id.str.extract(r"_(\d+)\.jpg")[0].astype(int)
    print(f"[video] files={len(fs)} rows={len(df)}  max|t - 0.5*photo_idx|={np.abs(df.t - 0.5 * fn).max()}")
    print("[video] Tracking_ID values:", df.Tracking_ID.unique())
    print(pd.crosstab(df["signal_info.movement"].values, df["car_info.movement"].values))


def load_signal_csv():
    fs = sorted(glob.glob(str(SIG / "신호데이터/교차로_all/csv/20220810/*.csv")))
    s = pd.concat([pd.read_csv(f, dtype={"interid": str}) for f in fs]).sort_values(["interid", "unix_time"])
    print(f"[signal] files={len(fs)} rows={len(s)} intersections={s.interid.nunique()} "
          f"chars={sorted(set(''.join(s.signalstate)))}")
    has_y = s.signalstate.str.contains("y")
    prev_gap = s.unix_time - s.groupby("interid").unix_time.shift()
    next_gap = s.groupby("interid").unix_time.shift(-1) - s.unix_time
    print("[signal] y-row duration if row time = END of state:", prev_gap[has_y].value_counts().head(4).to_dict())
    print("[signal] y-row duration if row time = START of state:", next_gap[has_y].value_counts().head(4).to_dict())
    return {k: (d.unix_time.values, d.signalstate.values) for k, d in s.groupby("interid")}


def load_net():
    net = ET.parse(NET).getroot()
    via, tl_in = {}, {}
    for c in net.iter("connection"):
        if c.get("tl"):
            tl_in[f"{c.get('from')}_{c.get('fromLane')}"] = c.get("tl")
            if c.get("via"):
                via[c.get("via")] = (c.get("tl"), int(c.get("linkIndex")), c.get("dir"))
    shapes = {l.get("id"): np.array([tuple(map(float, p.split(","))) for p in l.get("shape").split()])
              for e in net.iter("edge") for l in e.iter("lane")}
    print(f"[net] tlLogics={sum(1 for _ in net.iter('tlLogic'))} junctions={sum(1 for _ in net.iter('junction'))} "
          f"stopOffset={sum(1 for _ in net.iter('stopOffset'))}")
    return via, tl_in, shapes


def trajectories(sig, via, tl_in, shapes):
    tr = pd.read_csv(TRA, low_memory=False, dtype={"to_inter_id": str}).sort_values(["vhcl_id", "unix_time"])
    g = tr.groupby("vhcl_id")
    dt = g.unix_time.diff()
    acc = g.spd.diff() / dt
    print(f"[tra] rows={len(tr)} vehicles={tr.vhcl_id.nunique()} dt={dt.value_counts().to_dict()} "
          f"span={pd.to_datetime([tr.unix_time.min(), tr.unix_time.max()], unit='s', utc=True).tz_convert('Asia/Seoul').tolist()}")
    print("[tra] accel q0.001/q0.999/min/max:", acc.quantile([.001, .999]).round(2).tolist(), acc.min(), acc.max())
    print("[tra] all-NaN columns:", [c for c in tr if tr[c].isna().all()])

    # stop line = end of incoming lane shape (SUMO convention); check where stopped vehicles sit
    inc = tr[tr.lane_id.isin(tl_in)]
    st = inc[inc.spd < 0.1]
    x, y = Transformer.from_crs(4326, 32652, always_xy=True).transform(st.lon.values, st.lat.values)
    p = np.c_[x + NET_OFFSET[0], y + NET_OFFSET[1]]
    lon_d, lat_d = [], []
    for pi, l in zip(p, st.lane_id):
        a, b = shapes[l][-2], shapes[l][-1]
        u = (b - a) / np.linalg.norm(b - a)
        r = pi - b
        lon_d.append(-r @ u)
        lat_d.append(abs(r[0] * u[1] - r[1] * u[0]))
    print(f"[tra] stopped share on signalised approaches={len(st) / len(inc):.3f}; "
          f"dist upstream of lane end q0/q.01/q.5={np.quantile(lon_d, [0, .01, .5]).round(2)}; "
          f"lateral q.95={np.quantile(lat_d, .95):.2f} m")

    # junction entries (first sample on an internal lane), non-right-turn
    ent = tr[tr.lane_id.isin(via)].groupby(["vhcl_id", "lane_id"]).unix_time.min().reset_index()
    ent = ent[[via[l][2] != "r" for l in ent.lane_id]]

    def state(tl, t, mode):
        ts, ss = sig[tl]
        i = np.searchsorted(ts, t, side="right") - (1 if mode == "start" else 0)
        return ss[i] if 0 <= i < len(ss) else None

    for mode in ("end", "start"):
        v = pd.Series([state(via[l][0], t, mode)[via[l][1]] for l, t in zip(ent.lane_id, ent.unix_time)])
        print(f"[align] mode={mode}: n={len(v)} state at entry", v.value_counts(normalize=True).round(3).to_dict())

    # entry time relative to the link's yellow onset ("end" reading, which matches the XML programs)
    rel = []
    for l, t in zip(ent.lane_id, ent.unix_time):
        tl, li, _ = via[l]
        ts, ss = sig[tl]
        ch = np.array([s[li] for s in ss])
        starts = np.r_[ts[0] - 1, ts[:-1]]
        y0 = np.array([starts[i] for i in range(1, len(ch)) if ch[i] == "y" and ch[i - 1] in "Gg"])
        if len(y0):
            rel.append(t - y0[np.argmin(np.abs(y0 - t))])
    rel = np.array(rel)
    print(f"[align] entries vs yellow onset: n={len(rel)} share in [-2, +5] s = {np.mean((rel >= -2) & (rel <= 5)):.3f}")


if __name__ == "__main__":
    if not ROOT.exists():
        sys.exit(f"missing {ROOT}")
    video_labels()
    sig = load_signal_csv()
    via, tl_in, shapes = load_net()
    trajectories(sig, via, tl_in, shapes)
