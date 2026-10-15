"""W1 aggregation: per-shard outputs (scripts/womd_w1_run.py) -> project-schema tables + summary.

Tables (--tables, gitignored): events_through.csv / features_through.csv / sequences_through.npz with the columns the
V2X-Seq evaluation scripts expect (label, group = intersection cluster, segment = window id, cycle = window:onset,
track_id, yellow_s, f_yellow, f_red, ...) for the selected leaders: through connector, not an arrow signal, label
resolved, 2 s history, d/v <= 12 s, red onset visible (t_r). events_all.csv keeps every leader with flags.
Intersections: DBSCAN eps 3 m on signal stop points -> approach id; DBSCAN eps 60 m on approach centroids ->
intersection id (shared coordinate frame assumed; see FINDINGS). Summary json + small CSVs under --out.

  .venv/bin/python scripts/womd_w1_aggregate.py --splits validation --tables data/raw/womd/w1/tables_validation --out results/womd/w1/validation
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

DZ_LO, DZ_HI, TT_MAX = 2.5, 5.5, 12.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/womd/w1")
    ap.add_argument("--splits", default="validation")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.tables, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    E, F, S, M = [], [], [], []
    for split in args.splits.split(","):
        for ep in sorted(glob.glob(f"{args.raw}/{split}/*.events.csv")):
            base = ep[:-len(".events.csv")]
            if not os.path.exists(base + ".meta.json"):
                continue
            ev = pd.read_csv(ep)
            meta = json.load(open(base + ".meta.json"))
            meta["split"] = split
            M.append(meta)
            if len(ev) == 0:
                continue
            feat = pd.read_csv(base + ".feat.csv")
            z = np.load(base + ".seq.npz")
            seq = {k: z[k] for k in z.files} if "seq" not in z.files else {"lead+0.0": z["seq"]}
            ev["split"], ev["shard"] = split, os.path.basename(base)
            feat["ev"] += sum(len(e) for e in E)
            E.append(ev)
            F.append(feat)
            S.append(seq)
    ev = pd.concat(E, ignore_index=True)
    feat = pd.concat(F, ignore_index=True)
    lead_keys = list(S[0].keys())
    seq = {k: np.concatenate([q[k] for q in S]) for k in lead_keys}
    meta = pd.DataFrame(M)
    # intersections from stop points
    xy = ev[["stop_x", "stop_y"]].values
    ap_id = DBSCAN(eps=3.0, min_samples=1).fit_predict(xy)
    ev["approach_id"] = ap_id
    cent = pd.DataFrame(xy, columns=["x", "y"]).groupby(ap_id).mean()
    int_of_ap = pd.Series(DBSCAN(eps=60.0, min_samples=1).fit_predict(cent.values), index=cent.index)
    ev["intersection"] = "int" + ev["approach_id"].map(int_of_ap).astype(str)
    # schema columns
    ev["group"] = ev["intersection"]
    ev["segment"] = ev["scenario"]
    ev["cycle"] = ev["scenario"].astype(str) + ":" + ev["k0"].astype(str)
    ev["approach"] = ev["intersection"] + "/" + ev["approach_id"].astype(str)
    f0 = feat[feat["lead_s"] == 0.0]
    tt = f0.set_index("ev")["tt"].reindex(np.arange(len(ev))).values
    ev["tt"] = tt
    sel = (ev["movement"] == "through") & ~ev["arrow"].astype(bool) & (ev["label"] >= 0) & ev["hist_ok"].astype(bool) & (tt <= TT_MAX) & ev["f_red"].notna()
    ev["selected"] = sel
    ev.to_csv(f"{args.tables}/events_all.csv", index=False)
    keep = np.where(sel)[0]
    evs = ev.iloc[keep].reset_index(drop=True)
    fs = feat[feat["ev"].isin(keep)].copy()
    fs["ev"] = fs["ev"].map({o: i for i, o in enumerate(keep)})
    evs.to_csv(f"{args.tables}/events_through.csv", index=False)
    fs.to_csv(f"{args.tables}/features_through.csv", index=False)
    np.savez_compressed(f"{args.tables}/sequences_through.npz", **{k: v[keep] for k, v in seq.items()})
    # summary
    band = (evs["tt"] >= DZ_LO) & (evs["tt"] <= DZ_HI)
    v0 = fs[fs["lead_s"] == 0.0].set_index("ev")["v"].reindex(np.arange(len(evs))).values
    per_int = evs.groupby("intersection")["label"].agg(n="size", go="sum")
    thr = ev[(ev["movement"] == "through") & ~ev["arrow"].astype(bool)]
    summ = dict(
        splits=args.splits, shards=int(len(meta)), scenarios=int(meta["n_scenarios"].sum()), signal_scenarios=int(meta["n_signal"].sum()),
        michigan_scenarios=int(meta["n_michigan"].sum()), stitched_groups=int(meta["n_multi"].sum()), max_windows=int(meta["max_windows"].max()),
        yellow_end_original=_sum_dicts(meta["yellow_end_orig"]), yellow_end_used=_sum_dicts(meta["yellow_end_used"]),
        leaders_all=int(len(ev)), leaders_by_movement=ev["movement"].value_counts().to_dict(), arrow_onsets=int(ev["arrow"].sum()),
        t_r_source=ev["t_r_source"].value_counts().to_dict(), label_relaxed_added=int(((ev["label"] < 0) & (ev["label_relaxed"] >= 0)).sum()),
        through=dict(leaders=int(len(thr)), labeled=int((thr["label"] >= 0).sum()), stop=int((thr["label"] == 0).sum()), go=int((thr["label"] == 1).sum()),
                     unresolved=int((thr["label"] < 0).sum()), with_history=int(((thr["label"] >= 0) & thr["hist_ok"]).sum()),
                     with_t_r=int(((thr["label"] >= 0) & thr["hist_ok"] & thr["f_red"].notna()).sum())),
        selected=dict(n=int(len(evs)), go=int(evs["label"].sum()), go_rate=float(evs["label"].mean()) if len(evs) else None, dz=int(band.sum()),
                      dz_go=int(evs["label"][band].sum()), v_lt10=int((v0 < 10).sum()), v_ge10=int((v0 >= 10).sum()),
                      t_r_original=int((evs["t_r_source"] == "original").sum()), per_scenario=float(len(evs) / max(meta["n_scenarios"].sum(), 1))),
        yellow_s=dict(median=float(evs["yellow_s"].median()) if len(evs) else None, counts=evs["yellow_s"].round(1).value_counts().sort_index().to_dict()),
        intersections=dict(n=int(per_int.shape[0]), n_ge30=int((per_int["n"] >= 30).sum()), events_in_ge30=int(per_int.loc[per_int["n"] >= 30, "n"].sum()),
                           top=per_int.sort_values("n", ascending=False).head(10).to_dict("index")),
        accel=dict(da_raw=float(np.nanmean(meta["accel"].apply(lambda a: a.get("da_raw") or np.nan))), da_smooth=float(np.nanmean(meta["accel"].apply(lambda a: a.get("da_smooth") or np.nan)))),
        bytes_processed=int(meta["bytes"].sum()))
    json.dump(summ, open(f"{args.out}/summary.json", "w"), indent=1, default=_js)
    per_int.to_csv(f"{args.out}/per_intersection.csv")
    meta.drop(columns=["yellow_end_orig", "yellow_end_used", "accel"]).to_csv(f"{args.out}/shards.csv", index=False)
    print(json.dumps(summ, indent=1, default=_js))


def _sum_dicts(col):
    out = {}
    for d in col:
        for k, v in d.items():
            out[str(k)] = out.get(str(k), 0) + int(v)
    return out


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


if __name__ == "__main__":
    main()
