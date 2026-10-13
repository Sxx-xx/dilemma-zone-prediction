"""SinD v2.0 full release (6 intersections, 70 recordings with signal CSVs): screening, leader events, boundary, LOIO.

extract : per recording (spawn pool) -> screening rows (red-crossing rate of through vehicles with the inferred light
          map; per-approach visibility and approach speed as scripts/fluid_visibility.py) and all yellow-onset events
          with onset features (dzp.events / dzp.features, light map inferred from red crossings as for the sample).
          Tables under --event-dir (gitignored): events_all.csv, features_all.csv, sequences_all.npz; selection:
          recording red rate <= RED_MAX with >= MIN_CROSS through crossings, approach visibility >= DZ_HI * v_app.
          Leader tables (through, no unpassed vehicle ahead, selected approaches): events_through.csv, ...
eval    : (1) driver boundary t_50 (d/v logistic, d/v <= 12 s) with cycle-bootstrap CI, pooled and per intersection,
              next to the recorded yellow; (2) leave-one-intersection-out: rule A recorded t_r, rule A calibrated
              (t_r* from the training intersections), LR(d,v), LR(d,v,a), MLP; macro-F1, pairwise CIs; (3) rule A with
              the V2X-Seq boundary 2.67 s zero-shot. Outputs under --out (results/sind/full).

  .venv/bin/python scripts/sind_full.py --tasks extract --workers 6
  .venv/bin/python scripts/sind_full.py --tasks eval
"""
import argparse
import json
import os
import sys
import traceback
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from dzp import bench_eval, events, features, sind
from dzp.bench_models import LogRegDV, LogRegDVA, MLPDVA, RuleA
from dzp.geometry import build_approaches, project, turn_angle
from run_benchmark import at, movement

RED_MAX, MIN_CROSS = 0.08, 30
DZ_LO, DZ_HI = 2.5, 5.5
V2X_T50 = 2.67
_G = {}


def _init(root):
    os.nice(19)
    _G["root"], _G["off"], _G["maps"] = root, sind.full_offsets(root), {}


def screen_rows(rec, aps, cr, lmap):
    """Per-approach red-crossing counts, visibility (99th pct upstream distance) and free approach speed."""
    t = rec.tracks[rec.tracks["agent_type"].isin(sind.MOTOR_VEHICLES)]
    ta = turn_angle(t)
    thr = set(ta.index[np.abs(ta) < 30])
    c0 = cr[~cr["right_turn"] & cr["track_id"].isin(thr)]
    fx = t["x"].values + 0.5 * t["length"].values * np.cos(t["heading_rad"].values)
    fy = t["y"].values + 0.5 * t["length"].values * np.sin(t["heading_rad"].values)
    speed = np.hypot(t["vx"].values, t["vy"].values)
    rows = []
    for a in aps:
        c = c0[c0["sid"] == a.sid]
        red = known = 0
        if a.sid in lmap and len(c):
            st = rec.light_state_at(lmap[a.sid], c["frame"].values)
            known, red = int((st >= 0).sum()), int(((st == sind.RED) & (st >= 0)).sum())
        d, lat = project(a, fx, fy)
        vn = t["vx"].values * a.n[0] + t["vy"].values * a.n[1]
        on = t["track_id"].isin(c["track_id"]).values & (np.abs(lat) <= a.half_len + 1.5) & (d > 0)
        dd = d[on & (vn > 1)]
        up = pd.DataFrame({"tid": t["track_id"].values[on], "v": speed[on]}).groupby("tid")["v"].min()
        free = c[c["track_id"].isin(set(up.index[up >= 2.0]))]["speed"].values
        vis = float(np.quantile(dd, 0.99)) if len(dd) else np.nan
        va = float(free.mean()) if len(free) else np.nan
        rows.append(dict(recording=rec.name, intersection=rec.intersection, sid=a.sid, approach=a.name(), light=lmap.get(a.sid),
                         through_crossings=len(c), cross_known=known, cross_red=red, free_n=len(free), visible_m=vis, v_app_ms=va,
                         need_m=DZ_HI * va, vis_pass=bool(np.isfinite(va) and vis >= DZ_HI * va and len(c) >= MIN_CROSS)))
    return rows


def one(task):
    city, rec_name = task
    try:
        rec = sind.load_full_recording(_G["root"], city, rec_name, _G["off"], _G["maps"])
        aps, cr = events.approach_crossings(rec)
        lmap = events.infer_light_map(rec, cr)
        srows = screen_rows(rec, aps, cr, lmap)
        ev, _, stats = events.extract_events(rec)
        out = dict(recording=rec.name, intersection=rec.intersection, dur_min=(rec.tracks["timestamp_ms"].max() - rec.tracks["timestamp_ms"].min()) / 60000,
                   n_tracks=int(rec.tracks["track_id"].nunique()), n_lights=len(rec.light_cols), notes="; ".join(rec.notes),
                   light_map=json.dumps({int(k): v for k, v in lmap.items()}), n_events=len(ev))
        out.update({f"n_{k}": int(v) for k, v in stats.items() if k != "light_map" and np.isscalar(v)})
        feat = seq = None
        if len(ev):
            ev["group"], ev["movement"] = rec.name, movement(rec, ev)
            feat, seqs = features.build(rec, ev, aps, [0.0], dt=sind.DT)
            seq = seqs[0.0]
        return dict(rec=out, screen=srows, ev=ev, feat=feat, seq=seq)
    except Exception:
        return dict(error=f"{city}/{rec_name}: {traceback.format_exc()}")


def extract(args):
    recs = sind.full_recordings(args.root)
    print(f"{len(recs)} recordings with tracks + light CSV", flush=True)
    R, S, E, F, Q, errs = [], [], [], [], [], []
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.root,)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, recs)):
            if "error" in r:
                errs.append(r["error"])
                print(r["error"].splitlines()[-1], flush=True)
                continue
            R.append(r["rec"])
            S += r["screen"]
            if r["ev"] is not None and len(r["ev"]):
                off = sum(len(e) for e in E)
                f = r["feat"].copy()
                f["ev"] += off
                E.append(r["ev"])
                F.append(f)
                Q.append(r["seq"])
            print(f"{i + 1}/{len(recs)} {r['rec']['recording']} events {r['rec']['n_events']}", flush=True)
    os.makedirs(args.event_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    rec_df, sc = pd.DataFrame(R), pd.DataFrame(S)
    g = sc.groupby("recording").agg(cross_known=("cross_known", "sum"), cross_red=("cross_red", "sum"))
    rec_df = rec_df.merge(g, left_on="recording", right_index=True, how="left")
    rec_df["red_rate"] = rec_df["cross_red"] / rec_df["cross_known"].replace(0, np.nan)
    rec_df["sync_pass"] = (rec_df["red_rate"] <= RED_MAX) & (rec_df["cross_known"] >= MIN_CROSS)
    sc = sc.merge(rec_df[["recording", "sync_pass"]], on="recording")
    sc["selected"] = sc["sync_pass"] & sc["vis_pass"]
    rec_df.to_csv(f"{args.out}/screening_recordings.csv", index=False)
    sc.to_csv(f"{args.out}/screening_approaches.csv", index=False)
    open(f"{args.out}/extract_errors.txt", "w").write("\n".join(errs))
    ev = pd.concat(E, ignore_index=True)
    feat = pd.concat(F, ignore_index=True)
    seq = np.concatenate(Q)
    ev.to_csv(f"{args.event_dir}/events_all.csv", index=False)
    feat.to_csv(f"{args.event_dir}/features_all.csv", index=False)
    np.savez_compressed(f"{args.event_dir}/sequences_all.npz", **{"lead+0.0": seq})
    # leaders on selected (recording, approach)
    F0, S0 = at(feat, {0.0: seq}, 0.0, np.arange(len(ev)))
    ok_ap = set(map(tuple, sc.loc[sc["selected"], ["recording", "approach"]].values))
    sel = np.array([(g, a) in ok_ap for g, a in zip(ev["group"], ev["approach"])])
    lead = (F0["ldr_id"].values < 0) & (ev["movement"].values == "through")
    ev["leader"], ev["selected"] = lead, sel
    keep = np.where(lead & sel)[0]
    evl = ev.iloc[keep].reset_index(drop=True)
    evl["cycle"] = evl["group"].astype(str) + ":" + evl["f_yellow"].round().astype(int).astype(str)
    fl = feat[feat["ev"].isin(keep)].copy()
    fl["ev"] = fl["ev"].map({o: i for i, o in enumerate(keep)})
    evl.to_csv(f"{args.event_dir}/events_through.csv", index=False)
    fl.to_csv(f"{args.event_dir}/features_through.csv", index=False)
    np.savez_compressed(f"{args.event_dir}/sequences_through.npz", **{"lead+0.0": seq[keep]})
    summ = ev.assign(through=ev["movement"] == "through").groupby("group").agg(
        events=("label", "size"), through=("through", "sum"), leaders=("leader", "sum"),
        selected_leaders=("selected", lambda s: int((s & lead[s.index]).sum())), go_selected=("label", lambda s: int(s[sel[s.index] & lead[s.index]].sum())))
    rec_df = rec_df.merge(summ, left_on="recording", right_index=True, how="left")
    rec_df.to_csv(f"{args.out}/screening_recordings.csv", index=False)
    print(rec_df[["recording", "intersection", "dur_min", "cross_known", "red_rate", "sync_pass", "events", "through", "leaders", "selected_leaders", "go_selected", "notes"]].round(3).to_string(), flush=True)
    print(sc[["recording", "approach", "light", "through_crossings", "cross_red", "visible_m", "v_app_ms", "need_m", "vis_pass", "selected"]].round(1).to_string(), flush=True)
    print(f"leaders selected {len(evl)} go {int(evl.label.sum())} by intersection:\n{evl.groupby('intersection').label.agg(['size', 'sum'])}", flush=True)


# ----------------------------------------------------------------------------------------------- eval
def _t50(x, y):
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y)) < 2:
        return np.nan
    lr = LogisticRegression(max_iter=1000).fit(x[:, None], y)
    b1, b0 = lr.coef_[0, 0], lr.intercept_[0]
    return -b0 / b1 if b1 < 0 else np.nan


def _boot_t50(x, y, cyc, n_boot, seed=0):
    uc = np.unique(cyc)
    mem = {c: np.where(cyc == c)[0] for c in uc}
    rng = np.random.default_rng(seed)
    vals = np.array([_t50(*(lambda ix: (x[ix], y[ix]))(np.concatenate([mem[c] for c in rng.choice(uc, len(uc))]))) for _ in range(n_boot)], float)
    return np.nanpercentile(vals, [2.5, 97.5]), int(np.isnan(vals).sum())


def pairwise(ev, P, mask, n_boot, seed=0):
    y = ev["label"].values.astype(int)
    cyc = bench_eval.cycles(ev)
    ucyc = np.unique(cyc[mask])
    members = {c: np.where((cyc == c) & mask)[0] for c in ucyc}
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([members[c] for c in rng.choice(ucyc, len(ucyc))]) for _ in range(n_boot)]
    f1 = {n: np.array([bench_eval._macro_f1(y[ix], (P[n][0][ix] >= 0.5).astype(int)) for ix in draws]) for n in P}
    names, rows = list(P), []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = f1[a] - f1[b]
            rows.append(dict(model_a=a, model_b=b, diff=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5))))
    return pd.DataFrame(rows)


def evaluate(args):
    ev = pd.read_csv(f"{args.event_dir}/events_through.csv")
    feat = pd.read_csv(f"{args.event_dir}/features_through.csv")
    seq = np.load(f"{args.event_dir}/sequences_through.npz")["lead+0.0"]
    F, S = at(feat, {0.0: seq}, 0.0, np.arange(len(ev)))
    ok = F["t_r"].notna().values
    ev, F, S = ev[ok].reset_index(drop=True), F[ok].reset_index(drop=True), S[ok]
    ev["group_rec"] = ev["group"]
    n_sites_ok = int((ev.groupby("intersection")["label"].size() >= args.min_site).sum())
    by = args.loio_by if args.loio_by != "auto" else ("intersection" if n_sites_ok >= 2 else "recording")
    ev["group"] = ev["intersection"] if by == "intersection" else ev["group_rec"]   # fold unit; cycles keep the recording
    print(f"LOIO unit: {by} (intersections with >= {args.min_site} selected leaders: {n_sites_ok})", flush=True)
    y, tt, tr, d = ev["label"].values.astype(int), F["tt"].values, F["t_r"].values, F["d"].values
    band = (tt >= DZ_LO) & (tt <= DZ_HI)
    cyc = bench_eval.cycles(ev)
    print(f"leaders {len(ev)} go {int(y.sum())} band {int(band.sum())}; per intersection:\n{ev.groupby('group').label.agg(['size', 'sum'])}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    # (1) boundary
    keep = tt <= args.tt_max
    rows = []
    groups = [("pooled", np.ones(len(y), bool))] + [(s, ev["group"].values == s) for s in sorted(ev["group"].unique())]
    for name, m in groups:
        idx = np.where(m & keep)[0]
        if len(idx) < 10 or len(np.unique(y[idx])) < 2:
            rows.append(dict(site=name, n=len(idx), n_go=int(y[idx].sum()), tt50=np.nan, lo=np.nan, hi=np.nan, yellow_median=float(np.median(tr[idx])) if len(idx) else np.nan))
            continue
        (lo, hi), nn = _boot_t50(tt[idx], y[idx], cyc[idx], args.n_boot)
        rows.append(dict(site=name, n=len(idx), n_go=int(y[idx].sum()), tt50=_t50(tt[idx], y[idx]), lo=lo, hi=hi, boot_undefined=nn,
                         yellow_median=float(np.median(tr[idx])), yellow_min=float(tr[idx].min()), yellow_max=float(tr[idx].max()),
                         band_n=int((m & band).sum()), band_go_rate=float(y[m & band].mean()) if (m & band).any() else np.nan))
    bd = pd.DataFrame(rows)
    bd.to_csv(f"{args.out}/boundary.csv", index=False)
    print(bd.round(3).to_string(), flush=True)
    # (2) LOIO by intersection
    sites = ev["group"].value_counts()
    test_sites = sorted(sites.index[sites >= args.min_site])
    folds = [(s, np.where(ev["group"].values != s)[0], np.where(ev["group"].values == s)[0]) for s in test_sites]
    tested = np.zeros(len(y), bool)
    P = {RuleA.name: ((tr - tt > 0).astype(float), tr - tt), "ruleA_calibrated": np.full(len(y), np.nan),
         "ruleA_v2x_2.67": ((V2X_T50 - tt > 0).astype(float), V2X_T50 - tt)}
    for M in (LogRegDV, LogRegDVA, MLPDVA):
        P[M.name] = np.full(len(y), np.nan)
    params = []
    for site, trn, te in folds:
        tested[te] = True
        k = trn[tt[trn] <= args.tt_max]
        t50 = _t50(tt[k], y[k])
        P["ruleA_calibrated"][te] = (tt[te] < t50).astype(float)
        for M in (LogRegDV, LogRegDVA, MLPDVA):
            P[M.name][te] = M().fit(F.iloc[trn], S[trn], y[trn]).proba(F.iloc[te], S[te])
        params.append(dict(test_site=site, n_test=len(te), n_go_test=int(y[te].sum()), n_train=len(trn), t_r_star=t50, yellow_median_test=float(np.median(tr[te]))))
        print(f"  fold {site}: t_r* {t50:.2f}", flush=True)
    for k in ("ruleA_calibrated", LogRegDV.name, LogRegDVA.name, MLPDVA.name):
        P[k] = (P[k], P[k])
    pd.DataFrame(params).to_csv(f"{args.out}/loio_fold_params.csv", index=False)
    pd.DataFrame({k: v[0] for k, v in P.items()}).assign(recording=ev["group_rec"], intersection=ev["group"], track_id=ev["track_id"]).to_csv(f"{args.out}/preds_loio.csv", index=False)
    rows, pw = [], []
    subsets = [("all", tested), ("dz", tested & band), ("v<10", tested & (F["v"].values < 10)), ("v>=10", tested & (F["v"].values >= 10))]
    for sub, mask in subsets:
        if mask.sum() < 10:
            continue
        for ref in (RuleA.name, "ruleA_calibrated", LogRegDVA.name):
            rows.append(bench_eval.bootstrap(ev, P, mask, args.n_boot, ref=ref).assign(subset=sub))
        pw.append(pairwise(ev, P, mask, args.n_boot).assign(subset=sub))
        for site, _, te in folds:
            m = np.zeros(len(y), bool)
            m[te] = True
            if sub == "all":
                rows.append(bench_eval.bootstrap(ev, P, m, args.n_boot, ref=RuleA.name).assign(subset=f"site:{site}"))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(f"{args.out}/loio_metrics.csv", index=False)
    pd.concat(pw, ignore_index=True).to_csv(f"{args.out}/loio_pairwise.csv", index=False)
    q = df[(df.metric == "macro_f1") & (df.ref == RuleA.name)]
    print(q[["subset", "model", "value", "ci_lo", "ci_hi", "diff_vs_ref", "diff_lo", "diff_hi", "n", "n_go"]].round(3).to_string(), flush=True)
    json.dump(dict(n=len(ev), n_go=int(y.sum()), n_band=int(band.sum()), loio_unit=by, test_sites=test_sites, n_tested=int(tested.sum()),
                   per_site=ev.groupby("group")["label"].agg(n="size", go="sum").to_dict("index"),
                   per_intersection=ev.groupby("intersection")["label"].agg(n="size", go="sum").to_dict("index"),
                   yellow_s=ev["yellow_s"].round(1).value_counts().to_dict()), open(f"{args.out}/counts.json", "w"), indent=1, default=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/sind/full/SinD_v2.0_CSV_Dataset")
    ap.add_argument("--event-dir", default="data/raw/sind/results/full")
    ap.add_argument("--out", default="results/sind/full")
    ap.add_argument("--tasks", default="extract,eval")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-site", type=int, default=30)
    ap.add_argument("--tt-max", type=float, default=12.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--loio-by", default="auto", choices=["auto", "intersection", "recording"])
    args = ap.parse_args()
    for t in args.tasks.split(","):
        {"extract": extract, "eval": evaluate}[t](args)


if __name__ == "__main__":
    main()
