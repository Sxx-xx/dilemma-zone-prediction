"""W1 step 7 helpers: evaluation table variants and the unresolved-share table.

From --tables (scripts/womd_w1_aggregate.py output, all splits):
  <tables>_int/  group = intersection (DBSCAN 3 m stop points -> 60 m), cycle = window:onset   (LOIO, --min-site 30 / 10)
  <tables>_gkf/  group = fold0..fold9 from GroupKFold over *regions* (intersection centroids linked within 300 m, a
                 proxy for "same coordinate frame / same drive area"), cycle = region  (group 10-fold, region bootstrap)
  <out>/unresolved_by_tt.csv : through, non-arrow leaders with 2 s history: share unresolved per d/v bin (and in band)
  <out>/regions.csv          : region id, intersections, events, fold

  .venv/bin/python scripts/womd_w1_variants.py --tables data/raw/womd/w1/tables_all --out results/womd/w1/final
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupKFold

DZ_LO, DZ_HI = 2.5, 5.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--region-eps", type=float, default=300.0)
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ev = pd.read_csv(f"{args.tables}/events_through.csv")
    # regions from intersection centroids
    cent = ev.groupby("intersection")[["stop_x", "stop_y"]].mean()
    reg = pd.Series(DBSCAN(eps=args.region_eps, min_samples=1).fit_predict(cent.values), index=cent.index)
    ev["region"] = "reg" + ev["intersection"].map(reg).astype(str)
    gkf = GroupKFold(n_splits=args.folds)
    fold = np.empty(len(ev), int)
    for k, (_, te) in enumerate(gkf.split(ev, groups=ev["region"].values)):
        fold[te] = k
    ev["fold"] = "fold" + pd.Series(fold).astype(str)
    reg_tab = ev.groupby("region").agg(intersections=("intersection", "nunique"), events=("label", "size"), go=("label", "sum"), fold=("fold", "first"))
    reg_tab.to_csv(f"{args.out}/regions.csv")
    print(f"regions {len(reg_tab)}, events per fold: {ev['fold'].value_counts().sort_index().to_dict()}", flush=True)
    for name, group, cycle in (("int", ev["intersection"], ev["segment"].astype(str) + ":" + ev["k0"].astype(str)), ("gkf", ev["fold"], ev["region"])):
        d = f"{args.tables}_{name}"
        os.makedirs(d, exist_ok=True)
        e2 = ev.copy()
        e2["group"], e2["cycle"] = group.values, cycle.values
        e2.to_csv(f"{d}/events_through.csv", index=False)
        shutil.copy(f"{args.tables}/features_through.csv", f"{d}/features_through.csv")
        shutil.copy(f"{args.tables}/sequences_through.npz", f"{d}/sequences_through.npz")
    # unresolved share by d/v among through non-arrow leaders with history (labels: -1 unresolved)
    a = pd.read_csv(f"{args.tables}/events_all.csv")
    thr = a[(a["movement"] == "through") & ~a["arrow"].astype(bool) & a["hist_ok"].astype(bool) & (a["tt"] <= 12)].copy()
    thr["bin"] = pd.cut(thr["tt"], [0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8, 12])
    t = thr.groupby("bin", observed=True).agg(n=("label", "size"), unresolved=("label", lambda s: int((s < 0).sum())), stop=("label", lambda s: int((s == 0).sum())),
                                              go=("label", lambda s: int((s == 1).sum())), with_t_r=("f_red", lambda s: int(s.notna().sum())),
                                              track_end_s=("t_end", "median"), v_end=("v_end", "median"))
    t["unresolved_share"] = (t["unresolved"] / t["n"]).round(3)
    t["go_share_of_resolved"] = (t["go"] / (t["stop"] + t["go"]).replace(0, np.nan)).round(3)
    band = thr[(thr["tt"] >= DZ_LO) & (thr["tt"] <= DZ_HI)]
    t.loc["band 2.5-5.5"] = [len(band), int((band.label < 0).sum()), int((band.label == 0).sum()), int((band.label == 1).sum()), int(band.f_red.notna().sum()),
                             band.t_end.median(), band.v_end.median(), round((band.label < 0).mean(), 3), round((band.label == 1).sum() / max((band.label >= 0).sum(), 1), 3)]
    t.to_csv(f"{args.out}/unresolved_by_tt.csv")
    print(t.to_string(), flush=True)


if __name__ == "__main__":
    main()
