"""SinD sample: extract yellow-onset stop/go events and run the 4-model comparison.

Outputs (results/sind/): events.csv, sequences.npz, extraction_stats.json, metrics_within.csv,
metrics_loio.csv (only if >= 2 intersections have both classes).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import evaluate, sind
from dzp.events import HIST_S, extract_events

OUT = "results/sind"


def main():
    os.makedirs(OUT, exist_ok=True)
    evs, seqs, stats = [], [], {}
    for rec in sind.load_all():
        ev, seq, st = extract_events(rec)
        evs.append(ev)
        seqs.append(seq)
        stats[rec.intersection] = {k: (v if k != "light_map" else {str(a): b for a, b in v.items()}) for k, v in st.items()}
        stats[rec.intersection]["events"] = len(ev)
        stats[rec.intersection]["loader_notes"] = rec.notes
    ev = pd.concat(evs, ignore_index=True)
    seq = np.concatenate(seqs)
    ev["cycle_id"] = ev["f_yellow"].round().astype(int)
    ev.to_csv(f"{OUT}/events.csv", index=False)
    np.savez_compressed(f"{OUT}/sequences.npz", seq=seq)
    json.dump(stats, open(f"{OUT}/extraction_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
    print(ev.groupby(["intersection", "label"]).size().unstack(fill_value=0))

    onset = int(round(HIST_S / sind.DT)) - 1
    per_site = ev.groupby("intersection")["label"].agg(lambda s: s.nunique() == 2 and len(s) >= 20)
    usable = per_site[per_site].index.tolist()
    sub = ev["intersection"].isin(usable).values
    print("sites usable for within-site CV:", usable)
    evs = ev[sub].reset_index(drop=True)
    res, preds = evaluate.run(evs, seq[sub], onset, sind.DT, "within")
    res.to_csv(f"{OUT}/metrics_within.csv", index=False)
    preds.to_csv(f"{OUT}/preds_within.csv", index=False)
    evaluate.bootstrap_vs_baseline(evs, preds).to_csv(f"{OUT}/bootstrap_within.csv", index=False)
    print(res.round(3).to_string())
    if len(usable) >= 2:
        r2, _ = evaluate.run(ev[sub].reset_index(drop=True), seq[sub], onset, sind.DT, "loio")
        r2.to_csv(f"{OUT}/metrics_loio.csv", index=False)
        print(r2.round(3).to_string())
    else:
        print("LOIO skipped: fewer than 2 intersections with enough events of both classes.")


if __name__ == "__main__":
    main()
