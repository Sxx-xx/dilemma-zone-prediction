"""V2X-Seq evaluation (v4_excl95 sensitivity): list the ambiguous 'go' events from the label audit.

A 'go' event is ambiguous when, after yellow onset, its minimum speed drops below 1 m/s and it is never more than 6 m
past the stop line (min_d_after > -6): a slow creep over the line rather than a clear pass. Input is
results/v2xseq/review/events_audit.csv (scripts/v2xseq_review.py --audit-all); output is the (segment, track_id) list
used by scripts/v2xseq_eval.py --exclude.

  .venv/bin/python scripts/v2xseq_ambiguous_go.py
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="results/v2xseq/review/events_audit.csv")
    ap.add_argument("--out", default="results/v2xseq/review/ambiguous_go.csv")
    args = ap.parse_args()
    A = pd.read_csv(args.audit)
    go = A[A.label == 1]
    sus = go[(go.min_v_after < 1.0) & (go.min_d_after > -6)][["group", "track_id"]].rename(columns={"group": "segment"})
    sus["track_id"] = sus.track_id.astype(int)
    sus.to_csv(args.out, index=False)
    print(f"ambiguous go listed {len(sus)} -> {args.out}")


if __name__ == "__main__":
    main()
