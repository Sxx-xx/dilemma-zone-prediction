"""N figure: mean commit time vs macro-F1 of the staged rule (per theta) and the commit-all baselines.

  .venv/bin/python scripts/staged_dz_figure.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = "results/figures"
DS = [("V2X-Seq (band, n = 509)", "results/v2xseq/staged/staged.csv"), ("WOMD (band, n = 1,055)", "results/womd/staged/staged.csv")]
STYLE = {"mlp_dva": dict(color="black", marker="o", ls="-", mfc="black", label="MLP (d, v, a)"),
         "band_logreg_dva": dict(color="0.45", marker="s", ls="--", mfc="white", label="Band-only LR (d, v, a)")}
BASE = {"all@0.0s": ("v", "commit all at 0 s"), "all@1.0s": ("D", "commit all at 1.0 s"), "all@2.0s": ("^", "commit all at 2.0 s")}


def main():
    os.makedirs(f"{OUT}/data", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    tab = []
    for ax, (title, path) in zip(axes, DS):
        r = pd.read_csv(path)
        tab.append(r.assign(dataset=title.split(" (")[0]))
        for m, st in STYLE.items():
            q = r[(r.model == m) & r.policy.str.startswith("theta=") & (r.policy != "theta=train-selected")]
            q = q.assign(th=q.policy.str[6:].astype(float)).sort_values("th")
            ax.plot(q.t_mean, q.macro_f1, color=st["color"], ls=st["ls"], marker=st["marker"], mfc=st["mfc"], mec=st["color"], ms=6, lw=1.2, label=st["label"])
            for t in q.itertuples():
                ax.annotate(f"{t.th:g}", (t.t_mean, t.macro_f1), textcoords="offset points", xytext=(-3, 6) if m == "mlp_dva" else (4, -11), ha="right" if m == "mlp_dva" else "left", fontsize=7, color=st["color"])
            s = r[(r.model == m) & (r.policy == "theta=train-selected")].iloc[0]
            ax.plot(s.t_mean, s.macro_f1, marker="*", ms=13, color=st["color"], mfc=st["mfc"] if m != "mlp_dva" else "black", mec=st["color"], ls="none",
                    label="train-selected θ" if m == "mlp_dva" else None)
            for b, (mk, lab) in BASE.items():
                s = r[(r.model == m) & (r.policy == b)].iloc[0]
                ax.plot(s.t_mean, s.macro_f1, marker=mk, ms=7, color=st["color"], mfc="white", mec=st["color"], mew=1.3, ls="none",
                        label=lab if m == "mlp_dva" else None)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Mean commit time after yellow onset (s)")
        ax.set_xlim(-0.1, 2.1)
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("Macro-F1 (band events)")
    axes[0].text(0.02, 0.97, "numbers = θ (commit when P ≤ θ or P ≥ 1 − θ)", transform=axes[0].transAxes, fontsize=7, va="top")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(f"{OUT}/fig_n_staged_tradeoff.png", dpi=300)
    pd.concat(tab).to_csv(f"{OUT}/data/fig_n_staged_tradeoff.csv", index=False)


if __name__ == "__main__":
    main()
