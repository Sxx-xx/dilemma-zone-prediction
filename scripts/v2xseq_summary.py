"""Tables from scripts/v2xseq_eval.py outputs: external test, LOIO (pooled / per site / speed split),
sensitivity across runs (v4 = calibrated lines, v3 = uncalibrated, v4_excl = ambiguous go removed), learning curve.

  .venv/bin/python scripts/v2xseq_summary.py --eval results/v2xseq/eval
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ORDER = ["ruleA_d/v<t_r", "ruleB_v2/2d>b", "logreg_dva", "logreg_dv", "mlp_dva", "lstm_xyva", "transformer_xyva", "lstm_dva", "transformer_dva"]


def fmt(r):
    return f"{r.value:.3f} ({r.diff_vs_ref:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}])"


def table(df, metric="macro_f1", cols=("subset",), scope=None):
    d = df[df.metric == metric]
    if scope is not None:
        d = d[d.scope == scope]
    out = {}
    for key, g in d.groupby(list(cols)):
        key = key[0] if len(cols) == 1 else key
        out[key] = {r.model: fmt(r) for r in g.itertuples()}
    T = pd.DataFrame(out).reindex(ORDER)
    T.index.name = metric
    return T


def sig(r):
    return "+" if r.diff_lo > 0 else ("-" if r.diff_hi < 0 else "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    args = ap.parse_args()
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 40)
    runs = {k: f"{args.eval}/{k}" for k in ("v4", "v3", "v4_excl95") if os.path.exists(f"{args.eval}/{k}/loio.csv")}
    for k, p in runs.items():
        c = json.load(open(f"{p}/counts.json"))
        print(f"\n===== {k}: n={c['n']} go={c['n_go']} subsets={c['subsets']}")
        if os.path.exists(f"{p}/external.csv"):
            ex = pd.read_csv(f"{p}/external.csv")
            print("\n[external, FLUID-trained] macro-F1 (Δ vs rule A [95% CI])")
            print(table(ex).to_string())
            print("\n[external] AUROC")
            print(table(ex, "auroc").to_string())
        lo = pd.read_csv(f"{p}/loio.csv")
        lc = json.load(open(f"{p}/loio_counts.json"))
        print(f"\n[LOIO] test sites {len(lc['test_sites'])}, tested events {lc['n_tested']} (go {lc['n_tested_go']})")
        print(table(lo, scope="pooled").to_string())
        print("\n[LOIO] AUROC")
        print(table(lo, "auroc", scope="pooled").to_string())
        ps = lo[(lo.metric == "macro_f1") & (lo.scope != "pooled")]
        T = ps.pivot(index="scope", columns="model", values="diff_vs_ref")[[m for m in ORDER if m != ORDER[0]]].round(3)
        T.insert(0, "ruleA", ps[ps.model == ORDER[0]].set_index("scope")["value"].round(3))
        T.insert(0, "n", ps[ps.model == ORDER[0]].set_index("scope")["n"])
        S = ps.pivot(index="scope", columns="model", values="diff_lo")[[m for m in ORDER if m != ORDER[0]]]
        H = ps.pivot(index="scope", columns="model", values="diff_hi")[[m for m in ORDER if m != ORDER[0]]]
        print("\n[LOIO per site] rule A macro-F1 and Δ per model (* = CI excludes 0)")
        T2 = T.copy()
        for m in [m for m in ORDER if m != ORDER[0]]:
            T2[m] = [f"{v:+.2f}{'*' if (l > 0 or h < 0) else ''}" for v, l, h in zip(T[m], S[m], H[m])]
        print(T2.sort_values("n", ascending=False).to_string())
    if len(runs) > 1:
        print("\n===== sensitivity: sign of Δ macro-F1 vs rule A (+ / - significant, 0 n.s.) by run")
        rows = []
        for k, p in runs.items():
            for name, f in (("external", "external.csv"), ("loio", "loio.csv")):
                if not os.path.exists(f"{p}/{f}"):
                    continue
                d = pd.read_csv(f"{p}/{f}")
                d = d[(d.metric == "macro_f1") & (d.subset == "all")]
                if "scope" in d:
                    d = d[d.scope == "pooled"]
                for r in d.itertuples():
                    rows.append(dict(run=k, test=name, model=r.model, d=f"{r.diff_vs_ref:+.3f}{sig(r)}"))
        R = pd.DataFrame(rows).pivot(index=["test", "model"], columns="run", values="d")
        print(R.reindex([(t, m) for t in ("external", "loio") for m in ORDER]).dropna(how="all").to_string())
    if os.path.exists(f"{args.eval}/v4/curve.csv"):
        cv = pd.read_csv(f"{args.eval}/v4/curve.csv")
        print("\n===== LOIO learning curve (v4): macro-F1 mean (2.5-97.5 % over draws)")
        cv["s"] = [f"{m:.3f} ({l:.3f}-{h:.3f})" for m, l, h in zip(cv["mean"], cv["lo"], cv["hi"])]
        print(cv.pivot(index="model", columns="frac", values="s").to_string())


if __name__ == "__main__":
    main()
