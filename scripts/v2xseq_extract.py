"""V2X-Seq leader event tables for evaluation, in the run_benchmark.py --event-dir format (parallel over segments).

Per segment (dzp.v2xseq with the calibrated stop lines when --shift-table is given): events.extract_events with the
data-given light map, movement by turn angle (run_benchmark.movement), features.build at lead 0, keep through
platoon leaders (ldr_id < 0). Writes to --out:
  events_through.csv    group = site (for leave-one-intersection-out), segment, cycle = segment:onset (bootstrap unit)
  features_through.csv  lead_s = 0 rows, ev = row index into events_through
  sequences_through.npz lead+0.0 -> (N, 20, 4) (x, y, v, a)

  nice -n 19 .venv/bin/python scripts/v2xseq_extract.py --root data/raw/v2xseq/full/extracted/V2X-Seq-TFD \\
      --shift-table results/v2xseq/stopline_shift.csv --out data/raw/v2xseq/results/leaders_v4 --workers 3
"""
import argparse
import os
import sys
import traceback
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import events, features, v2xseq
from dzp.geometry import build_approaches
from run_benchmark import movement

_G = {}


def _init(root, cache, nice, shift_path):
    os.nice(nice)
    _G["root"], _G["lanes"], _G["shift"] = root, v2xseq.lane_index(root, cache), v2xseq.load_shift_table(shift_path)


def one(seg):
    try:
        rec = v2xseq.load_recording(_G["root"], seg, _G["lanes"], shift_table=_G["shift"])
        if not rec.light_cols or rec.tracks.empty:
            return None
        aps = build_approaches(rec.map.stop_lines())
        infer, events.infer_light_map = events.infer_light_map, (lambda _r, _c: rec.light_map)
        try:
            ev, _, _ = events.extract_events(rec)
        finally:
            events.infer_light_map = infer
        if ev.empty:
            return None
        ev["movement"] = movement(rec, ev)
        ev = ev[ev["movement"] == "through"].reset_index(drop=True)
        if ev.empty:
            return None
        f, s = features.build(rec, ev, aps, [0.0], dt=v2xseq.DT)
        keep = np.where(f["ldr_id"].values < 0)[0]
        if not len(keep):
            return None
        ev = ev.iloc[keep].reset_index(drop=True)
        ev["segment"], ev["group"] = seg.name, seg.inter
        ev["cycle"] = seg.name + ":" + ev["f_yellow"].round().astype(int).astype(str)
        ev["t_abs"] = seg.t0 + ev["f_yellow"] * v2xseq.DT
        f = f.iloc[keep].reset_index(drop=True)
        return ev, f, s[0.0][keep]
    except Exception:
        return dict(error=f"{seg.name}: {traceback.format_exc()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="data/raw/v2xseq/results/cache")
    ap.add_argument("--shift-table", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--nice", type=int, default=19)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    segs = v2xseq.segments(args.root, args.cache)
    E, F, S, errs = [], [], [], []
    with Pool(args.workers, initializer=_init, initargs=(args.root, args.cache, args.nice, args.shift_table)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, segs, chunksize=4)):
            if r is None:
                continue
            if isinstance(r, dict):
                errs.append(r["error"])
                continue
            E.append(r[0]), F.append(r[1]), S.append(r[2])
            if i % 2000 == 0:
                print(f"{i}/{len(segs)} leaders {sum(len(e) for e in E)}", flush=True)
    ev = pd.concat(E, ignore_index=True)
    feat = pd.concat(F, ignore_index=True)
    feat["ev"] = np.arange(len(feat))
    seq = np.concatenate(S)
    ev.to_csv(f"{args.out}/events_through.csv", index=False)
    feat.to_csv(f"{args.out}/features_through.csv", index=False)
    np.savez_compressed(f"{args.out}/sequences_through.npz", **{"lead+0.0": seq.astype(np.float32)})
    open(f"{args.out}/errors.txt", "w").write("\n".join(errs))
    print(f"done: leaders {len(ev)} (go {int(ev.label.sum())}), sites {ev.group.nunique()}, errors {len(errs)}", flush=True)


if __name__ == "__main__":
    main()
