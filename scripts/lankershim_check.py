"""NGSIM Lankershim (Wayback copy of Lankershim-Main-Data.zip): can yellow onset be reconstructed from the
LADOT RSM cycle reports, and how many yellow-onset events does the existing pipeline find?

No model comparison. Outputs results/ngsim_lankershim/{hypotheses.csv, stop_lines.csv, events_count.json}.

RSM facts used (docs/FINDINGS.md): one row per cycle, splits are GREEN only (cycle = sum of served greens +
~5 s yellow+all-red per served phase), Lankershim through = phase 2 / 6 (timing sheets p.1), trajectory frame 0
corresponds to RSM 8:27:34 (0830 file) / 8:44:34 (0845 file) (metadata.htm). Unknown: where in the cycle the
RSM row time lies -> tested as hypotheses with the red-crossing invariant.
"""
import itertools
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from dzp import ngsim

RAW = "data/raw/ngsim/lankershim/extracted"
OUT = "results/ngsim_lankershim"
PERIODS = {"0830am-0845am": 8 * 3600 + 27 * 60 + 34, "0845am-0900am": 8 * 3600 + 44 * 60 + 34}  # RSM s at frame 0
YELLOW_S = {89: 3.5}    # phase 2/6 yellow of int089 (timing sheet p.2); others not read -> 3.9 (int086 phase 4)
# through phase per direction (timing sheets p.1)
THRU = {87: {"SB": 2, "NB": 6}, 88: {"NB": 2, "SB": 6}, 89: {"SB": 2, "NB": 6}, 90: {"SB": 2, "NB": 6},
        91: {"SB": 2, "NB": 6}}
MAPPINGS = {"A(087-090)": {1: 87, 2: 88, 3: 89, 4: 90}, "B(088-091)": {1: 88, 2: 89, 3: 90, 4: 91}}
MAPPING = MAPPINGS["B(088-091)"]   # int2 = RSM089 is the only approach pair that aligns (see per_intersection.csv)


def load_tracks():
    cols = ["vehicle_id", "frame_id", "tot", "global_time", "local_x", "local_y", "gx", "gy", "v_length", "v_width",
            "v_class", "v_vel", "v_acc", "lane_id", "o", "d", "int_id", "section_id", "direction", "movement",
            "preceding", "following", "spacing", "headway"]
    ts = []
    for per in PERIODS:
        t = pd.read_csv(f"{RAW}/vehicle-trajectory-data/{per}/trajectories-{per}.txt", sep=r"\s+", header=None,
                        names=cols, usecols=[c for c in cols if c not in ("gx", "gy", "o", "d", "tot")])
        t["period"] = per
        ts.append(t)
    t = pd.concat(ts).sort_values(["period", "vehicle_id", "frame_id"]).reset_index(drop=True)
    t["track"] = t.groupby(["period", "vehicle_id"]).ngroup()
    t["y_m"] = t["local_y"] * ngsim.FT
    t["len_m"] = t["v_length"] * ngsim.FT
    v = np.full(len(t), np.nan)
    a = np.full(len(t), np.nan)
    for _, idx in t.groupby("track").indices.items():   # same smoothing as ngsim.load_tracks
        y = t["y_m"].values[idx]
        if len(y) >= 11:
            v[idx] = savgol_filter(y, 11, 3, deriv=1, delta=ngsim.DT)
            a[idx] = savgol_filter(y, 11, 3, deriv=2, delta=ngsim.DT)
        else:
            v[idx] = np.gradient(y, ngsim.DT) if len(y) > 1 else 0.0
            a[idx] = 0.0
    t["v_s"], t["a_s"] = v, a
    return t


def load_rsm(n):
    rows = []
    for line in open(f"{RAW}/signal-timing/real-time-split-monitor-reports/RSM{n:03d}-061605.txt", errors="ignore"):
        m = re.match(r"\s*(\d\d):(\d\d):(\d\d)\s+(.*)$", line)
        if not m:
            continue
        nums = [int(x) for x in m.group(4).split()]
        if len(nums) < 10:
            continue
        sec = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        cyc, ofs, g = nums[-10], nums[-9], nums[-8:]
        rows.append(dict(t=sec, cyc=cyc, ofs=ofs, **{f"g{i + 1}": g[i] for i in range(8)}))
    return pd.DataFrame(rows)


def build_signals(mapping, boundary, order, shift_s=0.0):
    """Through-phase cycles in ngsim.load_signals format (frames of each period).

    boundary: 'start' = RSM row time is the cycle start, 'end' = the cycle end.
    order: 'lead' = phase 1/5 runs before 2/6 within the cycle, 'coord' = 2/6 green starts at the boundary.
    """
    rows = []
    for int_id, n in mapping.items():
        r = load_rsm(n)
        r = r[(r.t > 8 * 3600 + 20 * 60) & (r.t < 9 * 3600 + 10 * 60)]
        for d, ph in THRU[n].items():
            lead = ph - 1                               # 1 before 2, 5 before 6
            ring = [1, 2, 3, 4] if ph == 2 else [5, 6, 7, 8]
            for _, c in r.iterrows():
                served = [p for p in ring if c[f"g{p}"] > 0]
                clear = (c.cyc - sum(c[f"g{p}"] for p in ring)) / max(len(served), 1)
                b = c.t if boundary == "start" else c.t - c.cyc
                g0 = b + ((c[f"g{lead}"] + clear) if (order == "lead" and c[f"g{lead}"] > 0) else 0)
                y0 = g0 + c[f"g{ph}"]
                for per, t0 in PERIODS.items():
                    f = lambda s: (s + shift_s - t0) / ngsim.DT
                    rows.append(dict(period=per, int_id=int_id, dir=d, f_green=f(g0), f_yellow=f(y0),
                                     f_red=f(y0 + YELLOW_S.get(n, 3.9)), cyc=c.cyc, clear=clear))
    s = pd.DataFrame(rows).sort_values(["period", "int_id", "dir", "f_green"])
    s["next_green"] = s.groupby(["period", "int_id", "dir"])["f_green"].shift(-1)
    s["yellow_s"] = (s["f_red"] - s["f_yellow"]) * ngsim.DT
    return s.dropna(subset=["next_green"]).reset_index(drop=True)


def red_fraction(cr, sig, k=0.0):
    n = red = 0
    for (per, i, d), g in cr.groupby(["period", "int_id", "dir"]):
        s = ngsim.signal_state(sig, per, i, d, g["frame"].values, k)
        v = s != "?"
        n += v.sum()
        red += (s[v] == "R").sum()
    return red / max(n, 1), int(n)


def scan(cr, sig, ks):
    return pd.DataFrame([dict(k_s=k * ngsim.DT, red_frac=red_fraction(cr, sig, k)[0]) for k in ks])


def main():
    os.makedirs(OUT, exist_ok=True)
    t = load_tracks()
    # 1) global hypotheses: mapping x RSM-row-time position x phase order
    res = []
    for (mname, mp), boundary, order in itertools.product(MAPPINGS.items(), ("start", "end"), ("lead", "coord")):
        sig = build_signals(mp, boundary, order)
        stops = ngsim.fill_missing_stop_lines(ngsim.estimate_stop_lines(t, sig))
        cr = ngsim.crossings(t, stops)
        cr = cr[cr["movement"] == 1]
        sc = scan(cr, sig, range(-150, 151, 10))
        b = sc.loc[sc.red_frac.idxmin()]
        res.append(dict(mapping=mname, boundary=boundary, order=order, n_cross=red_fraction(cr, sig, 0)[1],
                        red_frac_k0=round(red_fraction(cr, sig, 0)[0], 3), best_k_s=b.k_s, red_frac_best=round(b.red_frac, 3)))
    pd.DataFrame(res).sort_values("red_frac_k0").to_csv(f"{OUT}/hypotheses.csv", index=False)

    # 2) per intersection / direction, mapping B, row time = cycle start, phase 1/5 leads; offset scan +-60 s
    sig = build_signals(MAPPING, "start", "lead")
    stops = ngsim.fill_missing_stop_lines(ngsim.estimate_stop_lines(t, sig))
    stops.to_csv(f"{OUT}/stop_lines.csv", index=False)
    cr = ngsim.crossings(t, stops)
    cr = cr[cr["movement"] == 1]
    per = []
    for (i, d), g in cr.groupby(["int_id", "dir"]):
        sc = scan(g, sig, range(-600, 601, 10))
        b = sc.loc[sc.red_frac.idxmin()]
        per.append(dict(int_id=i, rsm=MAPPING[i], dir=d, n_cross=len(g),
                        red_frac_k0=round(sc[sc.k_s == 0].red_frac.iloc[0], 3), best_k_s=b.k_s, red_frac_best=round(b.red_frac, 3)))
    per = pd.DataFrame(per)
    per.to_csv(f"{OUT}/per_intersection.csv", index=False)
    print(per.to_string())

    # 3) fine scan for the aligned intersection (int2 = RSM089), 0.1 s steps over +-5 s
    i2 = cr[cr.int_id == 2]
    fine = scan(i2, sig, range(-50, 51, 1))
    fine.to_csv(f"{OUT}/int2_fine_scan.csv", index=False)
    m = fine.red_frac.min()
    flat = fine[fine.red_frac <= m + 0.01].k_s
    print(f"int2 fine scan: min {m:.3f} at k={fine.loc[fine.red_frac.idxmin(), 'k_s']:+.1f} s; "
          f"within +1 pp of min: {flat.min():+.1f}..{flat.max():+.1f} s")

    # 4) events at int2 only (other intersections not aligned), existing labeling pipeline
    ngsim.INTERSECTIONS = {i: f"int{i}_RSM{n:03d}" for i, n in MAPPING.items()}
    out = {}
    for k in (0, int(round(flat.min() / ngsim.DT)), int(round(flat.max() / ngsim.DT))):
        ev, seq, stats = ngsim.extract_events(t, sig[sig.int_id == 2], stops[stops.int_id == 2], k=k)
        out[f"k={k * ngsim.DT:+.1f}s"] = dict(n_events=len(ev), n_stop=int((ev.label == 0).sum()), n_go=int((ev.label == 1).sum()),
                                              by_dir=ev.groupby("approach").size().to_dict() if len(ev) else {}, **stats)
    out["int2_flat_k_range_s"] = [float(flat.min()), float(flat.max())]

    # 5) queue-leader start after the computed green at int2 (same method as scripts/ngsim_green_start.py)
    from ngsim_green_start import find_leaders
    lead = find_leaders(t, sig[sig.int_id == 2], stops[stops.int_id == 2])
    lead.to_csv(f"{OUT}/int2_leaders.csv", index=False)
    ok = lead[(lead["moved_m"] > 3) & lead["delta0"].notna() & (lead["jitter_m"] < 0.15)]
    x = ok["delta0"]
    out["int2_green_start"] = dict(n_leaders=len(ok), median_s=round(x.median(), 2), p25_s=round(x.quantile(.25), 2),
                                   p75_s=round(x.quantile(.75), 2), min_s=round(x.min(), 2),
                                   k_for_R_0p5_to_2s=[round(0.5 - x.median(), 2), round(2.0 - x.median(), 2)],
                                   by_period={p: round(g.median(), 2) for p, g in ok.groupby("period")["delta0"]})
    json.dump(out, open(f"{OUT}/events_count.json", "w"), indent=1, default=int)
    print(json.dumps(out, indent=1, default=int))


if __name__ == "__main__":
    main()
