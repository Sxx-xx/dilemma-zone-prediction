"""Yellow-onset stop/go benchmark, dataset-agnostic: rules A/B, LR, MLP, LSTM, Transformer.

Example (FLUID FI, the pilot):
  .venv/bin/python scripts/run_benchmark.py --dataset fluid --root data/raw/fluid/FI/derived_data --site FI \
      --approaches W_approach,E_approach --light-map W_approach=light_E_s,E_approach=light_E_s \
      --out results/fluid_fi --event-dir data/raw/fluid/results/benchmark_fi
SinD (when the full data arrives; group = recording, or --group intersection for LOIO):
  .venv/bin/python scripts/run_benchmark.py --dataset sind --root data/raw/sind --out results/sind_bench \
      --event-dir data/raw/sind/results/benchmark --group intersection

--split leader|follower evaluates only platoon leaders / followers at yellow onset (default all).
Outputs: --out (aggregates + figures, committable): metrics.csv, leadtime.csv, offsets.csv, counts.json, fig_*.png
         --event-dir (per-event rows, keep out of git): events_through.csv, events_turning.csv, features_*.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dzp import bench_eval, events, features, sind
from dzp.bench_models import model_list
from dzp.geometry import build_approaches, turn_angle

OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


# ----------------------------------------------------------------------------------------------- adapters
def load_recordings(args, offset_s):
    """[(recording, group)] with the signal shifted by offset_s (signal time + offset)."""
    if args.dataset == "fluid":
        from dzp import fluid
        vids = [v for v in fluid.videos(args.root) if v not in set(filter(None, args.exclude.split(",")))]
        return [(fluid.load_recording(args.root, args.site, v, shift_s=offset_s), v) for v in vids]
    if args.dataset == "sind":
        out = []
        for rec in sind.load_all(args.root):
            rec.lights = rec.lights.assign(frame=rec.lights["frame"] + offset_s / sind.DT)
            out.append((rec, rec.intersection if args.group == "intersection" else rec.name))
        return out
    if args.dataset == "v2xseq":
        from dzp import v2xseq
        cache = os.path.join(args.root, "..", "..", "results", "cache")
        lanes = v2xseq.lane_index(args.root, cache)
        segs = [s for s in v2xseq.segments(args.root, cache)
                if (not args.site or s.inter == args.site) and s.name not in set(filter(None, args.exclude.split(",")))]
        return [(v2xseq.load_recording(args.root, s, lanes, shift_s=offset_s),
                 s.inter if args.group == "intersection" else s.name) for s in segs]
    raise ValueError(args.dataset)


def movement(rec, ev):
    """'through' / 'left' / 'right' per event: dataset entry/exit directions if present, else net turn angle."""
    t = rec.tracks
    if "entry_direction" in t.columns:
        ent = t.groupby("track_id")[["entry_direction", "exit_direction"]].first().reindex(ev["track_id"])
        e, x = ent["entry_direction"].values, ent["exit_direction"].values
        left = {"N": "E", "E": "S", "S": "W", "W": "N"}           # entering from N (heading S), leaving E = left
        return np.where(x == pd.Series(e).map(OPP).values, "through",
                        np.where(x == pd.Series(e).map(left).values, "left", "right"))
    ta = turn_angle(t).reindex(ev["track_id"]).values
    return np.where(np.abs(ta) < 30, "through", np.where(ta > 0, "left", "right"))


def extract(args, offset_s):
    lmap_arg = dict(kv.split("=") for kv in filter(None, args.light_map.split(",")))
    keep = set(filter(None, args.approaches.split(",")))
    evs, feats, seqs = [], [], {L: [] for L in args.leads}
    for rec, group in load_recordings(args, offset_s):
        aps = build_approaches(rec.map.stop_lines())
        override = None
        if lmap_arg:
            override = {a.sid: lmap_arg[a.name()] for a in aps if a.name() in lmap_arg}
        elif rec.light_map:
            override = rec.light_map
        if override is not None:
            infer, events.infer_light_map = events.infer_light_map, (lambda _r, _c: override)
        try:
            ev, _, _ = events.extract_events(rec)
        finally:
            if override is not None:
                events.infer_light_map = infer
        if ev.empty:
            continue
        if keep:
            ev = ev[ev["approach"].isin(keep)].reset_index(drop=True)
        ev["group"], ev["movement"] = group, movement(rec, ev)
        f, s = features.build(rec, ev, aps, args.leads, dt=sind.DT)
        f["ev"] += sum(len(e) for e in evs)
        evs.append(ev)
        feats.append(f)
        for L in args.leads:
            seqs[L].append(s[L])
    ev = pd.concat(evs, ignore_index=True)
    feat = pd.concat(feats, ignore_index=True)
    seqs = {L: np.concatenate(v) for L, v in seqs.items()}
    return ev, feat, seqs


def through(ev_all, feat_all, seq_all, split="all"):
    """Through events (renumbered 0..N-1), optionally only platoon leaders / followers at yellow onset
    (features.build: leader = no unpassed motor vehicle ahead in the same lane)."""
    keep = ev_all["movement"].values == "through"
    if split != "all":
        f0 = feat_all[feat_all["lead_s"] == 0.0].set_index("ev")["ldr_id"].reindex(np.arange(len(ev_all))).values
        keep &= (f0 < 0) if split == "leader" else (f0 >= 0)
    sel = np.where(keep)[0]
    feat = feat_all[feat_all["ev"].isin(sel)].copy()
    feat["ev"] = feat["ev"].map({o: i for i, o in enumerate(sel)})
    return sel, ev_all.iloc[sel].reset_index(drop=True), feat, {L: s[sel] for L, s in seq_all.items()}


def at(feat, seqs, L, idx):
    F = feat[feat["lead_s"] == L].set_index("ev").loc[idx].reset_index(drop=True)
    return F, seqs[L][idx]


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fluid", "sind", "v2xseq"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--site", default="")
    ap.add_argument("--exclude", default="", help="comma list of recordings/videos to drop")
    ap.add_argument("--approaches", default="", help="comma list of approach names to keep (empty = all)")
    ap.add_argument("--light-map", default="", help="approach=light,... (empty = infer from data)")
    ap.add_argument("--group", default="recording", choices=["recording", "intersection"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--event-dir", required=True)
    ap.add_argument("--offsets", default="-1,0,1", help="signal offsets (s) for the sensitivity check")
    ap.add_argument("--leads", default="-2,-1,0,0.5,1", help="prediction times relative to yellow onset (s)")
    ap.add_argument("--decel", type=float, default=3.0, help="rule B threshold (m/s^2)")
    ap.add_argument("--dz", default="2.5,5.5", help="dilemma-zone band of d/v at onset (s)")
    ap.add_argument("--schemes", default="group,strat")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--no-transformer", action="store_true")
    ap.add_argument("--split", default="all", choices=["all", "leader", "follower"],
                    help="evaluate all through events, or only platoon leaders / followers at yellow onset")
    args = ap.parse_args()
    args.leads = [float(x) for x in args.leads.split(",")]
    if 0.0 not in args.leads:
        args.leads.append(0.0)
    offsets = [float(x) for x in args.offsets.split(",")]
    dz_lo, dz_hi = map(float, args.dz.split(","))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.event_dir, exist_ok=True)
    models = model_list(args.decel, not args.no_transformer, args.n_seeds)
    ref = models[0][0].name

    # ---- events at offset 0; through movements only for the evaluation
    ev_all, feat_all, seq_all = extract(args, 0.0)
    turning = np.where(ev_all["movement"].values != "through")[0]
    ev_all.iloc[turning].to_csv(f"{args.event_dir}/events_turning.csv", index=False)
    _, ev, feat, seqs = through(ev_all, feat_all, seq_all, args.split)
    ev.to_csv(f"{args.event_dir}/events_through.csv", index=False)
    feat.to_csv(f"{args.event_dir}/features_through.csv", index=False)
    np.savez_compressed(f"{args.event_dir}/sequences_through.npz", **{f"lead{L:+.1f}": s for L, s in seqs.items()})
    idx = np.arange(len(ev))
    F0, S0 = at(feat, seqs, 0.0, idx)
    dz = (F0["tt"].values >= dz_lo) & (F0["tt"].values <= dz_hi)
    y = ev["label"].values
    counts = dict(split=args.split, events_all=len(ev_all), through=len(ev), through_stop=int((y == 0).sum()), through_go=int(y.sum()),
                  turning=int(len(turning)), groups=int(ev["group"].nunique()), cycles=int(len(np.unique(bench_eval.cycles(ev)))),
                  dz_band=[dz_lo, dz_hi], dz_n=int(dz.sum()), dz_go=int(y[dz].sum()),
                  per_group=ev.groupby("group")["label"].agg(n="size", go="sum").to_dict("index"),
                  lead_padded={f"{L:+.1f}": int((feat[feat.lead_s == L]["pad"] > 0).sum()) for L in args.leads},
                  decel=args.decel, t_r_at_onset=sorted(F0["t_r"].round(2).unique().tolist()))
    json.dump(counts, open(f"{args.out}/counts.json", "w"), indent=1, default=int)
    print(json.dumps(counts, indent=1, default=int), flush=True)

    # ---- main table: both schemes, all / dilemma-zone band
    rows = []
    for scheme in args.schemes.split(","):
        P = bench_eval.oof(models, F0, S0, ev, scheme, reps=args.reps)
        for subset, m in (("all", None), ("dz", dz)):
            b = bench_eval.bootstrap(ev, P, m, args.n_boot, ref=ref)
            rows.append(b.assign(scheme=scheme, subset=subset, offset_s=0.0, lead_s=0.0))
        print(f"[{scheme}] done", flush=True)
    met = pd.concat(rows, ignore_index=True)
    met.to_csv(f"{args.out}/metrics.csv", index=False)

    # ---- lead-time curve (group scheme)
    lt = []
    for L in args.leads:
        F, S = at(feat, seqs, L, idx)
        P = bench_eval.oof(models, F, S, ev, "group")
        b = bench_eval.bootstrap(ev, P, None, args.n_boot)
        lt.append(b[b.metric == "macro_f1"].assign(lead_s=L))
        print(f"[lead {L:+.1f}] done", flush=True)
    lt = pd.concat(lt, ignore_index=True)
    lt.to_csv(f"{args.out}/leadtime.csv", index=False)

    # ---- signal-offset sensitivity (group scheme, onset)
    off = []
    for o in offsets:
        if o == 0.0:
            e_o, F_o, S_o = ev, F0, S0
        else:
            ea, fa, sa = extract(args, o)
            _, e_o, fa, sa = through(ea, fa, sa, args.split)
            F_o, S_o = at(fa, sa, 0.0, np.arange(len(e_o)))
        P = bench_eval.oof(models, F_o, S_o, e_o, "group")
        b = bench_eval.bootstrap(e_o, P, None, args.n_boot, ref=ref)
        b = b.assign(offset_s=o, n_events=len(e_o), n_go=int(e_o["label"].sum()))
        for mname in ("macro_f1", "auroc"):
            r = b[b.metric == mname].copy()
            r["rank"] = r["value"].rank(ascending=False, method="min")
            off.append(r)
        print(f"[offset {o:+.1f}] n={len(e_o)}", flush=True)
    off = pd.concat(off, ignore_index=True)
    off.to_csv(f"{args.out}/offsets.csv", index=False)

    figures(args, met, lt, off, ev, F0, dz_lo, dz_hi)
    print(met[(met.metric == "macro_f1")].round(3).to_string())


def figures(args, met, lt, off, ev, F0, dz_lo, dz_hi):
    names = list(dict.fromkeys(met["model"]))
    # (1) bars: macro-F1 and AUROC with CI, first scheme, all vs dz
    sch = args.schemes.split(",")[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, mname in zip(axes, ("macro_f1", "auroc")):
        for k, (subset, c) in enumerate((("all", "tab:blue"), ("dz", "tab:orange"))):
            r = met[(met.scheme == sch) & (met.subset == subset) & (met.metric == mname)].set_index("model").loc[names]
            x = np.arange(len(names)) + (k - 0.5) * 0.38
            ax.bar(x, r["value"], 0.36, color=c, label=f"{subset} (n={int(r['n'].iloc[0])})",
                   yerr=[r["value"] - r["ci_lo"], r["ci_hi"] - r["value"]], capsize=3)
        ax.set_xticks(np.arange(len(names)), names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{mname} ({'leave-one-' + args.group + '-out' if sch == 'group' else 'repeated stratified 5-fold'}, 95% CI)", fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig1_metrics.png", dpi=130)
    plt.close(fig)
    # (2) (d, v) plane
    fig, ax = plt.subplots(figsize=(6.5, 5))
    y = ev["label"].values
    vv = np.linspace(0.1, max(F0["v"].max(), 1) * 1.05, 100)
    ax.fill_between(vv, dz_lo * vv, dz_hi * vv, color="0.85", zorder=0, label=f"d/v in [{dz_lo:g}, {dz_hi:g}] s")
    ax.scatter(F0["v"][y == 0], F0["d"][y == 0], s=14, c="tab:red", label="stop", alpha=0.7)
    ax.scatter(F0["v"][y == 1], F0["d"][y == 1], s=14, c="tab:green", label="go", alpha=0.7)
    tr = float(np.median(F0["t_r"]))
    ax.plot(vv, vv * tr, "k-", label=f"rule A: d = v·t_r (t_r={tr:.1f} s)")
    ax.plot(vv, vv ** 2 / (2 * args.decel), "k--", label=f"rule B: d = v²/(2·{args.decel:g})")
    ax.set_xlabel("v at yellow onset (m/s)")
    ax.set_ylabel("d to stop line (m)")
    ax.set_ylim(0, max(F0["d"].max() * 1.05, 5))
    ax.legend(fontsize=8)
    ax.set_title("rules predict go below their boundary (closer / faster)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig2_dv_plane.png", dpi=130)
    plt.close(fig)
    # (3) lead-time curve
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n in names:
        r = lt[lt.model == n].sort_values("lead_s")
        ax.errorbar(r["lead_s"], r["value"], yerr=[r["value"] - r["ci_lo"], r["ci_hi"] - r["value"]], marker="o", capsize=2, label=n)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("prediction time relative to yellow onset (s)")
    ax.set_ylabel("macro-F1")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig3_leadtime.png", dpi=130)
    plt.close(fig)
    # (4) rank by offset
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mname in zip(axes, ("macro_f1", "auroc")):
        for n in names:
            r = off[(off.model == n) & (off.metric == mname)].sort_values("offset_s")
            ax.plot(r["offset_s"], r["value"], marker="o", label=n)
        ax.set_xlabel("signal offset (s)")
        ax.set_ylabel(mname)
        ax.set_xticks(sorted(off["offset_s"].unique()))
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig4_offsets.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
