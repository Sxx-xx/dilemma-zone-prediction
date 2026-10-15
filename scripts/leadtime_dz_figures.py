"""Figures for the dilemma-zone lead-time analysis (scripts/leadtime_dz.py outputs), 300 dpi, B/W-distinguishable.
  fig_l_a_leadtime_f1.png     macro-F1 vs lead time per model, one panel per dataset (CI bars)
  fig_l_b_uncertain_share.png share of band events with 0.2 < P(go) < 0.8 vs lead time (band LR, MLP), per dataset
  fig_l_c_calibration.png     reliability curves at +0 s and +1 s (band LR, MLP), per dataset
  results/figures/data/fig_l_*.csv

  .venv/bin/python scripts/leadtime_dz_figures.py --runs v2xseq=results/v2xseq/leadtime_dz womd=results/womd/leadtime_dz fluid=results/fluid_fi/leadtime_dz
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STYLE = {"band_logreg_dva": ("#2a78d6", "-", "o", "band-only LR (d, v, a)"), "mlp_dva": ("#eb6834", "--", "s", "MLP (d, v, a)"),
         "lstm_dva": ("#1baf7a", "-.", "^", "LSTM (d, v, a)"), "ruleA_calibrated": ("#52514e", ":", "D", "rule A, calibrated t50(L)")}
LABEL = {"v2xseq": "V2X-Seq (23-fold LOIO)", "womd": "WOMD (group 10-fold)", "fluid": "FLUID FI (8-fold, n=26)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4df"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=dir")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    runs = dict(r.split("=", 1) for r in args.runs)
    os.makedirs(f"{args.out}/data", exist_ok=True)
    lt = {k: pd.read_csv(f"{d}/leadtime.csv") for k, d in runs.items()}
    cf = {k: pd.read_csv(f"{d}/confidence.csv") for k, d in runs.items()}
    cal = {k: pd.read_csv(f"{d}/calibration.csv") for k, d in runs.items()}
    pd.concat([v.assign(dataset=k) for k, v in lt.items()]).to_csv(f"{args.out}/data/fig_l_a_leadtime_f1.csv", index=False)
    pd.concat([v.assign(dataset=k) for k, v in cf.items()]).to_csv(f"{args.out}/data/fig_l_b_confidence.csv", index=False)
    pd.concat([v.assign(dataset=k) for k, v in cal.items()]).to_csv(f"{args.out}/data/fig_l_c_calibration.csv", index=False)
    names = list(runs)
    # (a)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.4), sharey=True)
    axes = [axes] if len(names) == 1 else list(axes)
    for ax, k in zip(axes, names):
        q = lt[k][lt[k].metric == "macro_f1"]
        for i, (m, (col, ls, mk, lab)) in enumerate(STYLE.items()):
            r = q[q.model == m].sort_values("lead_s")
            if r.empty:
                continue
            x = r["lead_s"] + (i - 1.5) * 0.03
            ax.errorbar(x, r["value"], yerr=[r["value"] - r["ci_lo"], r["ci_hi"] - r["value"]], color=col, ls=ls, marker=mk, ms=4, lw=1.5, capsize=2, label=lab)
        ax.axhline(0.8, color=INK2, lw=0.8, ls=(0, (1, 2)))
        ax.set_title(LABEL.get(k, k), fontsize=9)
        ax.set_xlabel("prediction time after yellow onset (s)")
        ax.set_xticks([0, 0.5, 1, 1.5, 2])
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("band macro-F1 (d/v 2.5-5.5 s at onset)")
    axes[0].set_ylim(0.3, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"{args.out}/fig_l_a_leadtime_f1.png", dpi=300)
    plt.close(fig)
    # (b)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    dsty = {"v2xseq": "-", "womd": "--", "fluid": ":"}
    for k in names:
        for m in ("band_logreg_dva", "mlp_dva"):
            r = cf[k][cf[k].model == m].sort_values("lead_s")
            ax.plot(r["lead_s"], r["uncertain_share"], color=STYLE[m][0], ls=dsty.get(k, "-"), marker=STYLE[m][2], ms=4, lw=1.5, label=f"{STYLE[m][3]}, {LABEL.get(k, k).split(' (')[0]}")
    ax.set_xlabel("prediction time after yellow onset (s)")
    ax.set_ylabel("share of band events with 0.2 < P(go) < 0.8")
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_l_b_uncertain_share.png", dpi=300)
    plt.close(fig)
    # (c) calibration: datasets with enough band events (FLUID n=26 has too few per bin)
    cnames = [k for k in names if k != "fluid"]
    fig, axes = plt.subplots(1, len(cnames), figsize=(3.6 * len(cnames), 3.6), sharey=True)
    axes = [axes] if len(cnames) == 1 else list(axes)
    for ax, k in zip(axes, cnames):
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (1, 2)))
        for m in ("band_logreg_dva", "mlp_dva"):
            for L, ls, mk in ((0.0, "-", "o"), (1.0, "--", "s")):
                r = cal[k][(cal[k].model == m) & (cal[k].lead_s == L) & (cal[k].n >= 5)].sort_values("bin")
                ax.plot(r["p_mean"], r["go_rate"], color=STYLE[m][0], ls=ls, marker=mk, ms=4, lw=1.4, label=f"{STYLE[m][3].split(' (')[0]} +{L:.0f} s")
        ax.set_title(LABEL.get(k, k), fontsize=9)
        ax.set_xlabel("predicted P(go)")
        ax.set_xlim(0, 1)
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("observed go rate (bins with n >= 5)")
    axes[0].set_ylim(0, 1)
    axes[-1].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_l_c_calibration.png", dpi=300)
    plt.close(fig)
    print("figures written", flush=True)


if __name__ == "__main__":
    main()
