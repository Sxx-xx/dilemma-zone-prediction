"""W1 step 1: label-resolution rate with vs without window stitching (and with vs without the Michigan signals)
on a few validation shards. Output: results/womd/w1/stitch_check.csv

  .venv/bin/python scripts/womd_w1_stitch_check.py --shards data/raw/womd/val_sample/validation.tfrecord-0000*-of-00150
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd

from dzp import womd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", default=sorted(glob.glob("data/raw/womd/val_sample/validation.tfrecord-*-of-00150")))
    ap.add_argument("--michigan", default="data/raw/womd/michigan_tls/validation")
    ap.add_argument("--out", default="results/womd/w1")
    args = ap.parse_args()
    idx = womd.michigan_index(args.michigan)
    rows = []
    for signals, mi in (("original", None), ("michigan", idx)):
        for stitch in (False, True):
            E, M = [], []
            for sh in args.shards:
                ev, _, _, meta, _ = womd.extract_shard(sh, mi, stitch=stitch)
                E.append(ev)
                M.append(meta)
            ev = pd.concat(E, ignore_index=True)
            thr = ev[(ev.movement == "through") & ~ev.arrow]
            r = dict(signals=signals, stitch=stitch, scenarios=sum(m["n_scenarios"] for m in M), multi_window_groups=sum(m["n_multi"] for m in M),
                     max_windows=max(m["max_windows"] for m in M), leaders=len(ev), labeled=int((ev.label >= 0).sum()), stop=int((ev.label == 0).sum()),
                     go=int((ev.label == 1).sum()), unresolved=int((ev.label < 0).sum()), through_leaders=len(thr), through_labeled=int((thr.label >= 0).sum()),
                     through_stop=int((thr.label == 0).sum()), through_go=int((thr.label == 1).sum()), through_labeled_hist=int(((thr.label >= 0) & thr.hist_ok).sum()),
                     through_with_t_r=int(((thr.label >= 0) & thr.hist_ok & thr.f_red.notna()).sum()),
                     dz=int((((thr.label >= 0) & thr.hist_ok) & ((thr.t_label.notna()))).sum()))
            rows.append(r)
            print(r, flush=True)
    pd.DataFrame(rows).to_csv(f"{args.out}/stitch_check.csv", index=False)


if __name__ == "__main__":
    main()
