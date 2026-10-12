"""External test on the SinD public sample: models trained on all FLUID FI leaders, applied unchanged to SinD leaders.

1. Screening per sample recording (the paper's selection criteria, section III.4):
   - signal sync: share of through crossings (|turn angle| < 30 deg) made on red, with the light map the event
     extraction uses (events.infer_light_map); recording passes if <= RED_MAX (8 %)
   - visibility per approach, as scripts/fluid_visibility.py: visible_m = 99th pct of upstream front distance of
     through vehicles moving toward the line; v_app = mean crossing speed of "free" through vehicles (never < 2 m/s
     upstream); pass if visible_m >= 5.5 v_app and >= MIN_CROSS through crossings (as scripts/fluid_check.py)
   - yellow-onset events (run_benchmark.extract, onset only): through stop/go, leader/follower
2. Models fitted once on the 88 FLUID FI leaders (--fluid-event-dir, written by run_benchmark.py --split leader):
   LR(d,v,a), LR(d,v), LSTM, Transformer (3 seeds); rules A (t_r = time to red from each recording's signal
   record) and B (3.0 m/s^2). No refitting, no threshold change: go iff P(go) >= 0.5.
3. Scored on the leaders of passing approaches of passing recordings: per recording (only if >= MIN_TABLE
   leaders) and pooled; cycle bootstrap 95 % CIs and paired differences vs rule A.

  .venv/bin/python scripts/external_sind_sample.py --out results/fluid_fi/leader_follower/leader/external_sind_sample
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import bench_eval, events, sind
from dzp.bench_models import LogRegDV, LogRegDVA, LSTMSeq, RuleA, RuleB, TransformerSeq
from dzp.geometry import project, turn_angle
from run_benchmark import at, extract, through

RED_MAX = 0.08
MIN_CROSS = 30
MIN_TABLE = 30
DZ_LO, DZ_HI = 2.5, 5.5


def screen(rec):
    """Red-crossing rate of through vehicles and per-approach visibility for one SinD recording."""
    t = rec.tracks[rec.tracks["agent_type"].isin(sind.MOTOR_VEHICLES)]
    aps, cr = events.approach_crossings(rec)
    lmap = events.infer_light_map(rec, cr)
    ta = turn_angle(t)
    thr = set(ta.index[np.abs(ta) < 30])
    cr = cr[~cr["right_turn"] & cr["track_id"].isin(thr)]
    n = red = 0
    fx = t["x"].values + 0.5 * t["length"].values * np.cos(t["heading_rad"].values)
    fy = t["y"].values + 0.5 * t["length"].values * np.sin(t["heading_rad"].values)
    speed = np.hypot(t["vx"].values, t["vy"].values)
    rows = []
    for a in aps:
        c = cr[cr["sid"] == a.sid]
        if a.sid in lmap and len(c):
            st = rec.light_state_at(lmap[a.sid], c["frame"].values)
            ok = st >= 0
            n += int(ok.sum())
            red += int(((st == sind.RED) & ok).sum())
        d, lat = project(a, fx, fy)
        vn = t["vx"].values * a.n[0] + t["vy"].values * a.n[1]
        on = t["track_id"].isin(c["track_id"]).values & (np.abs(lat) <= a.half_len + 1.5) & (d > 0)
        dd = d[on & (vn > 1)]
        up = pd.DataFrame({"tid": t["track_id"].values[on], "v": speed[on]}).groupby("tid")["v"].min()
        v = c[c["track_id"].isin(set(up.index[up >= 2.0]))]["speed"].values
        vis = float(np.quantile(dd, 0.99)) if len(dd) else np.nan
        va = float(v.mean()) if len(v) else np.nan
        rows.append(dict(recording=rec.name, approach=a.name(), light=lmap.get(a.sid), through_crossings=len(c),
                         free_n=len(v), visible_m=vis, v_app_ms=va, band_lo_m=DZ_LO * va, band_hi_m=DZ_HI * va,
                         passes=bool(np.isfinite(va) and vis >= DZ_HI * va and len(c) >= MIN_CROSS)))
    return red / max(n, 1), n, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sind-root", default="data/raw/sind")
    ap.add_argument("--fluid-event-dir", default="data/raw/fluid/results/benchmark_fi_leader")
    ap.add_argument("--out", required=True)
    ap.add_argument("--event-dir", default="data/raw/sind/results/external_sample")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.event_dir, exist_ok=True)

    # ---- 1. screening + events per recording
    vis, rec_rows = [], []
    for rec in sind.load_all(args.sind_root):
        rate, n, v = screen(rec)
        vis.append(v)
        rec_rows.append(dict(recording=rec.name, intersection=rec.intersection, through_crossings=n, red_rate=rate,
                             sync_pass=rate <= RED_MAX, approaches_pass=",".join(v.loc[v.passes, "approach"])))
    vis = pd.concat(vis, ignore_index=True)
    vis.to_csv(f"{args.out}/visibility.csv", index=False)
    bargs = SimpleNamespace(dataset="sind", root=args.sind_root, group="recording", leads=[0.0], light_map="",
                            approaches="", site="", exclude="")
    ev_all, feat_all, seq_all = extract(bargs, 0.0)
    _, ev, feat, seqs = through(ev_all, feat_all, seq_all, "all")
    F, S = at(feat, seqs, 0.0, np.arange(len(ev)))
    ev["leader"] = F["ldr_id"].values < 0
    ok_ap = set(map(tuple, vis.loc[vis.passes, ["recording", "approach"]].values))
    ok_rec = {r["recording"] for r in rec_rows if r["sync_pass"]}
    ev["approach_pass"] = [(g, a) in ok_ap for g, a in zip(ev["group"], ev["approach"])]
    ev["selected"] = ev["approach_pass"] & ev["group"].isin(ok_rec)
    ev.to_csv(f"{args.event_dir}/events_through.csv", index=False)
    feat.to_csv(f"{args.event_dir}/features_through.csv", index=False)
    for r in rec_rows:
        e = ev[ev["group"] == r["recording"]]
        es = e[e["selected"]]
        r.update(through=len(e), stop=int((e.label == 0).sum()), go=int(e.label.sum()),
                 leader=int(e.leader.sum()), leader_go=int(e[e.leader].label.sum()), follower=int((~e.leader).sum()),
                 follower_go=int(e[~e.leader].label.sum()), selected_leader=int(es.leader.sum()),
                 selected_leader_go=int(es[es.leader].label.sum()),
                 yellow_s=",".join(f"{x:.1f}" for x in sorted(e["yellow_s"].round(1).unique())),
                 passes=bool(r["sync_pass"] and r["approaches_pass"] and es.leader.sum() > 0))
    rec_df = pd.DataFrame(rec_rows)
    rec_df.to_csv(f"{args.out}/screening.csv", index=False)
    print(rec_df.to_string(), flush=True)
    print(vis.round(2).to_string(), flush=True)

    # ---- 2. models fitted on all FLUID leaders
    fe = pd.read_csv(f"{args.fluid_event_dir}/events_through.csv")
    ff = pd.read_csv(f"{args.fluid_event_dir}/features_through.csv")
    fs = np.load(f"{args.fluid_event_dir}/sequences_through.npz")
    fseq = {0.0: fs["lead+0.0"]}
    Ftr, Str = at(ff, fseq, 0.0, np.arange(len(fe)))
    ytr = fe["label"].values.astype(int)
    assert len(fe) == 88 and (Ftr["ldr_id"] < 0).all(), "expected the 88 FLUID FI leaders"
    models = [(RuleA, {}), (RuleB, {"decel": 3.0}), (LogRegDVA, {}), (LogRegDV, {}),
              (LSTMSeq, {"n_seeds": args.n_seeds}), (TransformerSeq, {"n_seeds": args.n_seeds})]
    fitted = [M(**kw).fit(Ftr, Str, ytr) for M, kw in models]

    # ---- 3. score the selected SinD leaders
    sel = np.where(ev["selected"] & ev["leader"])[0]
    et, Ft, St = ev.iloc[sel].reset_index(drop=True), F.iloc[sel].reset_index(drop=True), S[sel]
    P = {m.name: (m.proba(Ft, St), m.score(Ft, St)) for m in fitted}
    pd.DataFrame({"group": et["group"], "track_id": et["track_id"], "f_yellow": et["f_yellow"], "label": et["label"],
                  "d": Ft["d"], "v": Ft["v"], "a": Ft["a"], "t_r": Ft["t_r"],
                  **{n: p for n, (p, _) in P.items()}}).to_csv(f"{args.event_dir}/preds_leaders.csv", index=False)
    rows = []
    dz = (Ft["tt"].values >= DZ_LO) & (Ft["tt"].values <= DZ_HI)
    scopes = [("pooled", np.ones(len(et), bool))] + [(g, (et["group"] == g).values) for g in et["group"].unique()]
    for scope, m in scopes:
        b = bench_eval.bootstrap(et, P, m, args.n_boot, ref=RuleA.name)
        rows.append(b.assign(scope=scope, subset="all", in_table=scope == "pooled" or m.sum() >= MIN_TABLE))
    b = bench_eval.bootstrap(et, P, dz, args.n_boot, ref=RuleA.name)
    rows.append(b.assign(scope="pooled", subset="dz", in_table=True))
    met = pd.concat(rows, ignore_index=True)
    met.to_csv(f"{args.out}/metrics.csv", index=False)
    json.dump(dict(train="FLUID FI leaders", n_train=88, n_train_go=int(ytr.sum()), n_test=len(et),
                   n_test_go=int(et.label.sum()), test_per_recording=et.groupby("group")["label"].agg(n="size", go="sum").to_dict("index"),
                   t_r_test=sorted(Ft["t_r"].round(2).unique().tolist()), dz_n=int(dz.sum()), dz_go=int(et.label[dz].sum()),
                   min_table=MIN_TABLE, red_max=RED_MAX, min_cross=MIN_CROSS),
              open(f"{args.out}/counts.json", "w"), indent=1, default=int)
    print(met[met.metric != "brier"].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
