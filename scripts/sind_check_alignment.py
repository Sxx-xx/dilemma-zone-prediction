"""Sanity checks for the SinD sample: light/track time alignment, light->approach mapping, map overlay.

Idea: under the correct time alignment, (non-right-turn) motor vehicles should almost never
cross a stop line while their light is red. We shift the light timeline by `off` and report
the red-crossing fraction; the minimum should be near off = 0 s.

Outputs: results/sind/alignment.csv, results/sind/approach_light_map.csv, results/sind/map_<id>.png
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dzp import sind
from dzp.events import approach_crossings

OUT = "results/sind"
os.makedirs(OUT, exist_ok=True)


def red_fraction(rec, cr, off_frames):
    tot = red = 0
    best = {}
    for sid, g in cr.groupby("sid"):
        fr = g["frame"].values + off_frames
        scores = []
        for c in rec.light_cols:
            st = rec.light_state_at(c, fr)
            valid = st >= 0
            scores.append(((st == sind.RED) & valid).sum() / max(valid.sum(), 1))
        j = int(np.argmin(scores))
        best[sid] = (rec.light_cols[j], scores[j], len(g))
        red += scores[j] * len(g)
        tot += len(g)
    return red / max(tot, 1), best


def main():
    rows, maprows = [], []
    for rec in sind.load_all():
        aps, cr = approach_crossings(rec)
        cr = cr[~cr["right_turn"]]
        scan = [(o, *red_fraction(rec, cr, o)) for o in range(-400, 401, 5)]
        o_best, f_best, _ = min(scan, key=lambda s: s[1])
        f0, best0 = red_fraction(rec, cr, 0)
        rows.append(dict(intersection=rec.intersection, n_crossings=len(cr), red_frac_at_0=f0,
                         best_off_s=o_best * sind.DT, red_frac_best=f_best, notes="; ".join(rec.notes)))
        for sid, (col, frac, n) in best0.items():
            maprows.append(dict(intersection=rec.intersection, sid=sid, approach=aps[sid].name(),
                                light=col, red_frac=frac, n=n))
        # overlay figure: trajectories (motor vehicles, subsampled) + map + stop lines
        fig, ax = plt.subplots(figsize=(7, 7))
        for typ, xy in rec.map.all_ways_xy():
            ax.plot(xy[:, 0], xy[:, 1], color="0.75", lw=0.6)
        t = rec.tracks[rec.tracks["agent_type"].isin(sind.MOTOR_VEHICLES)].iloc[::5]
        ax.scatter(t["x"], t["y"], s=0.2, c="tab:blue", alpha=0.15)
        for a in aps:
            p0, p1 = a.mid - a.t * a.half_len, a.mid + a.t * a.half_len
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="red", lw=2.5)
            ax.annotate("", xy=a.mid + 6 * a.n, xytext=a.mid, arrowprops=dict(arrowstyle="->", color="red"))
            ax.text(*(a.mid - 6 * a.n), f"{a.sid}:{best0.get(a.sid, ('?',))[0]}", color="red", fontsize=8)
        ax.set_aspect("equal")
        ax.set_title(f"{rec.intersection}: stop lines (red, arrow = travel dir) + motor-vehicle points")
        fig.savefig(f"{OUT}/map_{rec.intersection}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/alignment.csv", index=False)
    pd.DataFrame(maprows).to_csv(f"{OUT}/approach_light_map.csv", index=False)
    print(df.round(3).to_string())
    print(pd.DataFrame(maprows).round(3).to_string())


if __name__ == "__main__":
    main()
