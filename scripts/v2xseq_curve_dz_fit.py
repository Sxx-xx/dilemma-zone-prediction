"""Band learning curve (task curve_dz of v2xseq_eval.py): table per model, slope 75 -> 100 %, power-law fit
f(n) = a - b * n^(-c) on the four mean values (n = training fraction), extrapolated to 2x and 4x the training cycles.
Also checks that the all-event re-run reproduces curve.csv. Output: curve_dz_fit.csv (+ markdown on stdout).

  .venv/bin/python scripts/v2xseq_curve_dz_fit.py --eval results/v2xseq/eval/v4
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

NAME = {"logreg_dva": "LR (d, v, a) 전체 학습", "band_logreg_dva": "띠 전용 LR (d, v, a)", "mlp_dva": "MLP (d, v, a)",
        "lstm_dva": "LSTM (d, v, a)", "transformer_dva": "Transformer (d, v, a)"}


def power(n, a, b, c):
    return a - b * n ** (-c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="results/v2xseq/eval/v4")
    args = ap.parse_args()
    lc = pd.read_csv(f"{args.eval}/curve_dz.csv")
    old = pd.read_csv(f"{args.eval}/curve.csv")
    rows = []
    for sub in ("dz", "all"):
        for m in NAME:
            q = lc[(lc.model == m) & (lc.subset == sub)].set_index("frac").sort_index()
            if q.empty:
                continue
            n = q.index.values.astype(float)
            mu = q["mean"].values
            r = dict(subset=sub, model=m, **{f"m{int(f * 100)}": q.loc[f, "mean"] for f in n}, **{f"lo{int(f * 100)}": q.loc[f, "lo"] for f in n},
                     **{f"hi{int(f * 100)}": q.loc[f, "hi"] for f in n}, slope_75_100=q.loc[1.0, "mean"] - q.loc[0.75, "mean"],
                     n_train_100=q.loc[1.0, "n_train"])
            try:
                (a, b, c), _ = curve_fit(power, n, mu, p0=[mu[-1] + 0.02, 0.05, 1.0], bounds=([0, 0, 0.05], [1.5, 1.0, 5.0]), maxfev=20000)
                r.update(fit_a=a, fit_b=b, fit_c=c, fit_rmse=float(np.sqrt(np.mean((power(n, a, b, c) - mu) ** 2))),
                         pred_2x=power(2.0, a, b, c), pred_4x=power(4.0, a, b, c))
            except Exception as e:  # noqa: BLE001
                r.update(fit_a=np.nan, fit_b=np.nan, fit_c=np.nan, fit_rmse=np.nan, pred_2x=np.nan, pred_4x=np.nan, fit_error=str(e)[:60])
            if sub == "all" and m in set(old.model):
                o = old[old.model == m].set_index("frac")["mean"]
                r["max_abs_diff_vs_curve_csv"] = float(np.max(np.abs(o.reindex(q.index).values - mu)))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.eval}/curve_dz_fit.csv", index=False)
    for sub in ("dz", "all"):
        d = df[df.subset == sub]
        print(f"\n### {sub}")
        print("| 모델 | 25% | 50% | 75% | 100% | 기울기 75→100 | 거듭제곱 맞춤 a − b·n^−c (RMSE) | 2배 예측 | 4배 예측 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in d.itertuples():
            cells = [f"{getattr(r, f'm{k}'):.3f} ({getattr(r, f'lo{k}'):.3f}–{getattr(r, f'hi{k}'):.3f})" for k in (25, 50, 75)] + [f"{r.m100:.3f}"]
            fit = f"a={r.fit_a:.3f}, b={r.fit_b:.3f}, c={r.fit_c:.2f} ({r.fit_rmse:.4f})" if np.isfinite(r.fit_a) else "—"
            print(f"| {NAME[r.model]} | " + " | ".join(cells) + f" | {r.slope_75_100:+.3f} | {fit} | {r.pred_2x:.3f} | {r.pred_4x:.3f} |")
        if sub == "all":
            print(d[["model", "max_abs_diff_vs_curve_csv"]].to_string())


if __name__ == "__main__":
    main()
