"""Paper materials (F): FLUID driver boundary, cross-dataset rule-A transfer, four 300-dpi figures + table CSVs.

results/figures/
  fig_a_go_probability_vs_tt.png   V2X-Seq leaders: P(go) vs d/v (binned go rates, logistic fit, 50 % boundary + CI)
  fig_b_tr_sensitivity.png         rule A macro-F1 vs fixed t_r (all / band), calibrated t_r* and recorded yellow marked
  fig_c_learning_curve.png         LOIO learning curve, 4 models, 10 draws (2.5-97.5 % band)
  fig_d_tt50_per_site.png          per-site d/v boundary with CI (15 sites) and the pooled value
  data/                            CSV behind every figure + the D main table and fold t_r*
  fluid_boundary.csv               FLUID leaders: d/v boundary (recording bootstrap) vs the 3 s yellow
  cross_dataset_ruleA.csv          rule A with recorded / own-boundary / other-dataset-boundary t_r, both directions

  .venv/bin/python scripts/paper_materials.py
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dzp import bench_eval
from v2xseq_eval import DZ_HI, DZ_LO, load_tables
from v2xseq_followup import load_fluid

# fixed categorical order (dataviz palette), each with its own line style / marker so B/W print still separates them
STYLE = {"logreg_dva": ("#2a78d6", "-", "o", "LR (d, v, a)"), "logreg_dv": ("#eb6834", "--", "s", "LR (d, v)"),
         "lstm_dva": ("#1baf7a", "-.", "^", "LSTM (d, v, a)"), "transformer_dva": ("#6250d6", ":", "D", "Transformer (d, v, a)")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 100})


def t50(x, y):
    if len(np.unique(y)) < 2:
        return np.nan, (np.nan, np.nan)
    lr = LogisticRegression(max_iter=1000).fit(x[:, None], y)
    b1, b0 = lr.coef_[0, 0], lr.intercept_[0]
    return (-b0 / b1 if b1 < 0 else np.nan), (b0, b1)


def boot_t50(x, y, clusters, n_boot, seed=0):
    uc = np.unique(clusters)
    mem = {c: np.where(clusters == c)[0] for c in uc}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ix = np.concatenate([mem[c] for c in rng.choice(uc, len(uc))])
        vals.append(t50(x[ix], y[ix])[0])
    vals = np.array(vals, float)
    return np.nanpercentile(vals, [2.5, 97.5]), int(np.isnan(vals).sum())


def rule_rows(ev, tt, t_r_rec, variants, n_boot, label):
    P = {}
    for name, t in variants.items():
        s = (t_r_rec if t is None else t) - tt
        P[name] = ((s > 0).astype(float), s)
    rows = []
    for sub, mask in [("all", np.ones(len(tt), bool)), ("dz", (tt >= DZ_LO) & (tt <= DZ_HI))]:
        b = bench_eval.bootstrap(ev, P, mask, n_boot, ref="recorded")
        rows.append(b.assign(subset=sub, dataset=label))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-dir", default="data/raw/v2xseq/results/leaders_v4")
    ap.add_argument("--fluid-event-dir", default="data/raw/fluid/results/benchmark_fi_leader")
    ap.add_argument("--eval", default="results/v2xseq/eval/v4")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tt-max", type=float, default=12.0)
    args = ap.parse_args()
    out, dat = args.out, f"{args.out}/data"
    os.makedirs(dat, exist_ok=True)
    ev, F, S = load_tables(args.event_dir)
    fe, FF, FS = load_fluid(args.fluid_event_dir)
    y, tt, trr = ev["label"].values.astype(int), F["tt"].values, F["t_r"].values
    fy, ftt, ftr = fe["label"].values.astype(int), FF["tt"].values, FF["t_r"].values

    # ---- F1: FLUID boundary (recording bootstrap) and the V2X-Seq pooled one (cycle bootstrap) for reference
    kf = ftt <= args.tt_max
    f50, (fb0, fb1) = t50(ftt[kf], fy[kf])
    (flo, fhi), fnan = boot_t50(ftt[kf], fy[kf], fe["group"].values[kf], args.n_boot)
    kv = tt <= args.tt_max
    v50, (vb0, vb1) = t50(tt[kv], y[kv])
    (vlo, vhi), vnan = boot_t50(tt[kv], y[kv], bench_eval.cycles(ev)[kv], args.n_boot)
    fb = pd.DataFrame([dict(dataset="fluid_fi_leaders", n=int(kf.sum()), n_go=int(fy[kf].sum()), clusters="recording (8 videos)", tt50=f50, lo=flo, hi=fhi,
                            boot_undefined=fnan, yellow_s=float(np.median(ftr)), beta0=fb0, beta1=fb1),
                       dict(dataset="v2xseq_leaders", n=int(kv.sum()), n_go=int(y[kv].sum()), clusters="yellow cycle", tt50=v50, lo=vlo, hi=vhi,
                            boot_undefined=vnan, yellow_s=float(np.median(trr)), beta0=vb0, beta1=vb1)])
    fb.to_csv(f"{out}/fluid_boundary.csv", index=False)
    print(fb.round(3).to_string(), flush=True)

    # ---- F2: cross-dataset rule A (recorded t_r / own boundary / the other dataset's boundary)
    cross = pd.concat([rule_rows(fe, ftt, ftr, {"recorded": None, "own_t50": f50, "v2xseq_t50": v50}, args.n_boot, "fluid_fi_leaders"),
                       rule_rows(ev, tt, trr, {"recorded": None, "own_t50": v50, "fluid_t50": f50}, args.n_boot, "v2xseq_leaders")], ignore_index=True)
    cross["t_r_used"] = cross["model"].map({"recorded": "recorded", "own_t50": f"own", "v2xseq_t50": f"{v50:.2f}", "fluid_t50": f"{f50:.2f}"})
    cross.to_csv(f"{out}/cross_dataset_ruleA.csv", index=False)
    q = cross[cross.metric.isin(["macro_f1", "auroc"])]
    print(q[["dataset", "subset", "model", "metric", "value", "ci_lo", "ci_hi", "diff_vs_ref", "diff_lo", "diff_hi", "n"]].round(3).to_string(), flush=True)

    # ---- fig (a): V2X-Seq P(go) vs d/v
    edges = np.arange(0, 12.01, 0.5)
    binned = pd.DataFrame({"bin": pd.cut(tt[kv], edges), "y": y[kv]}).groupby("bin", observed=True)["y"].agg(["size", "mean"]).reset_index()
    binned["tt_mid"] = [b.mid for b in binned["bin"]]
    grid = np.linspace(0, 12, 400)
    curve = 1 / (1 + np.exp(-(vb0 + vb1 * grid)))
    pd.DataFrame({"tt": grid, "p_go": curve}).to_csv(f"{dat}/fig_a_logistic_curve.csv", index=False)
    binned.drop(columns="bin").rename(columns={"size": "n", "mean": "go_rate"}).to_csv(f"{dat}/fig_a_binned_go_rate.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    rng = np.random.default_rng(0)
    ax.scatter(tt[kv], y[kv] + rng.uniform(-0.03, 0.03, kv.sum()), s=4, c=INK, alpha=0.12, lw=0, label=f"events (n={int(kv.sum())}, jittered)")
    ax.axvspan(vlo, vhi, color=GRID, alpha=0.9, lw=0)
    ax.axvline(v50, color=INK2, lw=1, ls="--", label=f"P(go) = 0.5 at d/v = {v50:.2f} s [{vlo:.2f}, {vhi:.2f}]")
    ax.axvline(float(np.median(trr)), color=INK2, lw=1, ls=":", label=f"recorded yellow {np.median(trr):.1f} s")
    ax.plot(grid, curve, color="#2a78d6", lw=2, label="logistic fit")
    ok = binned["size"] >= 10
    ax.scatter(binned["tt_mid"][ok], binned["mean"][ok], s=np.clip(binned["size"][ok] / 4, 10, 80), facecolors="white", edgecolors="#eb6834", lw=1.4, zorder=3,
               label="go rate per 0.5 s bin (area ~ n, n >= 10)")
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("time to stop line at yellow onset d/v (s)")
    ax.set_ylabel("P(go)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_a_go_probability_vs_tt.png", dpi=300)
    plt.close(fig)

    # ---- fig (b): fixed t_r sensitivity
    cur = pd.read_csv(f"{args.eval}/followup/ruleA_tr_curve.csv")
    cur.to_csv(f"{dat}/fig_b_tr_sensitivity.csv", index=False)
    fp = pd.read_csv(f"{args.eval}/calibrated_rule/fold_params.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(cur["t_r"], cur["macro_f1_all"], color="#2a78d6", lw=2, marker="o", ms=3.5, label="all leaders (n=1,867)")
    ax.plot(cur["t_r"], cur["macro_f1_dz"], color="#eb6834", lw=2, ls="--", marker="s", ms=3.5, label="dilemma-zone band, d/v 2.5-5.5 s (n=509)")
    ax.axvline(fp["t_r_star"].mean(), color=INK2, lw=1, ls="--", label=f"calibrated t_r* (LOIO mean {fp['t_r_star'].mean():.2f} s)")
    ax.axvline(4.0, color=INK2, lw=1, ls=":", label="recorded yellow 4.0 s")
    ax.set_xlabel("rule A threshold t_r (s): go if d/v < t_r")
    ax.set_ylabel("macro-F1")
    ax.set_xlim(1.5, 6)
    ax.set_ylim(0.15, 0.9)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_b_tr_sensitivity.png", dpi=300)
    plt.close(fig)

    # ---- fig (c): learning curve
    lc = pd.read_csv(f"{args.eval}/curve.csv")
    lc.to_csv(f"{dat}/fig_c_learning_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, (m, (col, ls, mk, lab)) in enumerate(STYLE.items()):
        r = lc[lc.model == m].sort_values("frac")
        x = r["frac"] * 100 + (i - 1.5) * 0.8
        ax.errorbar(x, r["mean"], yerr=[r["mean"] - r["lo"], r["hi"] - r["mean"]], color=col, ls=ls, marker=mk, ms=4, lw=1.6, capsize=2.5, label=lab)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_xlabel("training yellow cycles kept per LOIO fold (%)")
    ax.set_ylabel("pooled macro-F1 (15 test sites, n=1,726)")
    ax.set_ylim(0.80, 0.86)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=7, loc="lower right", title="bars: 2.5-97.5 % of 10 draws", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_c_learning_curve.png", dpi=300)
    plt.close(fig)

    # ---- fig (d): per-site boundary
    ps = pd.read_csv(f"{args.eval}/followup/tt50_per_site.csv")
    ps["site"] = ps["site"].str.replace("yizhuang", "").str.replace("_po", "")
    ps.to_csv(f"{dat}/fig_d_tt50_per_site.csv", index=False)
    pooled, sites = ps[ps.site == "pooled"].iloc[0], ps[ps.site != "pooled"].sort_values("tt50")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    xs = np.arange(len(sites))
    ax.axhspan(pooled["tt50_lo"], pooled["tt50_hi"], color=GRID, alpha=0.9, lw=0)
    ax.axhline(pooled["tt50"], color="#2a78d6", lw=1.5, ls="-", label=f"pooled {pooled['tt50']:.2f} s [{pooled['tt50_lo']:.2f}, {pooled['tt50_hi']:.2f}]")
    ax.axhline(4.0, color=INK2, lw=1, ls=":", label="recorded yellow 4.0 s")
    ax.errorbar(xs, sites["tt50"], yerr=[sites["tt50"] - sites["tt50_lo"], sites["tt50_hi"] - sites["tt50"]], fmt="o", color=INK, ms=4, lw=1.2, capsize=2.5,
                label="site boundary (95 % cycle-bootstrap CI)")
    for x, (_, r) in zip(xs, sites.iterrows()):
        ax.annotate(f"n={int(r['n'])}", (x, r["tt50_lo"]), textcoords="offset points", xytext=(0, -9), ha="center", fontsize=5.5, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(sites["site"], rotation=60, fontsize=7)
    ax.set_ylabel("d/v at P(go) = 0.5 (s)")
    ax.set_xlabel("intersection (sites with >= 30 leader events)")
    ax.set_ylim(1.0, 7.2)
    ax.grid(color=GRID, lw=0.6, axis="y")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_d_tt50_per_site.png", dpi=300)
    plt.close(fig)

    # ---- table CSVs
    shutil.copy(f"{args.eval}/calibrated_rule/metrics.csv", f"{dat}/table_D_metrics.csv")
    shutil.copy(f"{args.eval}/calibrated_rule/pairwise_macro_f1.csv", f"{dat}/table_D_pairwise.csv")
    shutil.copy(f"{args.eval}/calibrated_rule/fold_params.csv", f"{dat}/table_D_fold_tr_star.csv")
    print("figures written to", out, flush=True)


if __name__ == "__main__":
    main()
