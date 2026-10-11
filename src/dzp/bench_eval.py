"""Benchmark evaluation: out-of-fold predictions under a split scheme, metrics, cluster bootstrap CIs.

Schemes
  group : leave-one-group-out, group = ev["group"] (FLUID: video; SinD: recording or intersection -> LOIO)
  strat : StratifiedGroupKFold(n_folds) over yellow cycles, repeated with `reps` seeds; per-event probabilities
          are averaged over the repeats before scoring.
Metrics (go = positive for AUROC / Brier): macro-F1 at P(go) >= 0.5, AUROC of the model score, Brier of P(go).
Bootstrap: resample yellow cycles with replacement (events of one cycle share the signal and interact).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def cycles(ev: pd.DataFrame) -> np.ndarray:
    return (ev["group"].astype(str) + ":" + ev["f_yellow"].round().astype(int).astype(str)).values


def splits(ev: pd.DataFrame, scheme: str, n_folds: int = 5, seed: int = 0):
    if scheme == "group":
        g = ev["group"].values
        for u in sorted(np.unique(g)):
            yield np.where(g != u)[0], np.where(g == u)[0]
    elif scheme == "strat":
        cv = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        yield from cv.split(ev, ev["label"].values, groups=cycles(ev))
    else:
        raise ValueError(scheme)


def oof(models, F: pd.DataFrame, S: np.ndarray, ev: pd.DataFrame, scheme: str, reps: int = 1, n_folds: int = 5):
    """Out-of-fold P(go) and score per model; {name: (proba, score)} (averaged over repeats for 'strat')."""
    y = ev["label"].values.astype(int)
    out = {}
    for M, kw in models:
        P, Sc = [], []
        for r in range(reps if scheme == "strat" else 1):
            p, s = np.full(len(y), np.nan), np.full(len(y), np.nan)
            for tr, te in splits(ev, scheme, n_folds, seed=r):
                if len(np.unique(y[tr])) < 2:
                    continue
                m = M(**kw).fit(F.iloc[tr], S[tr], y[tr])
                p[te], s[te] = m.proba(F.iloc[te], S[te]), m.score(F.iloc[te], S[te])
            P.append(p)
            Sc.append(s)
        out[M.name] = (np.nanmean(P, 0), np.nanmean(Sc, 0))
    return out


def _macro_f1(y, yhat):
    f = []
    for c in (0, 1):
        tp = np.sum((yhat == c) & (y == c))
        den = np.sum(yhat == c) + np.sum(y == c)
        f.append(2 * tp / den if den else 0.0)
    return float(np.mean(f))


def metrics(y, p, s) -> dict:
    ok = np.isfinite(p)
    y, p, s = y[ok], p[ok], s[ok]
    auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    return dict(macro_f1=_macro_f1(y, (p >= 0.5).astype(int)), auroc=auc, brier=float(np.mean((p - y) ** 2)), n=int(len(y)),
                n_go=int(y.sum()))


def bootstrap(ev, preds: dict, mask=None, n_boot: int = 1000, seed: int = 0, ref: str | None = None) -> pd.DataFrame:
    """Point estimate + 95% percentile CI per model and metric; also the paired difference to `ref`."""
    y = ev["label"].values.astype(int)
    mask = np.ones(len(y), bool) if mask is None else np.asarray(mask)
    cyc = cycles(ev)
    ucyc = np.unique(cyc[mask])
    members = {c: np.where((cyc == c) & mask)[0] for c in ucyc}
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([members[c] for c in rng.choice(ucyc, len(ucyc))]) for _ in range(n_boot)]
    idx0 = np.where(mask)[0]
    rows = []
    boot = {}
    for name, (p, s) in preds.items():
        pt = metrics(y[idx0], p[idx0], s[idx0])
        bs = pd.DataFrame([metrics(y[ix], p[ix], s[ix]) for ix in draws])
        boot[name] = bs
        for mname in ("macro_f1", "auroc", "brier"):
            lo, hi = np.nanpercentile(bs[mname], [2.5, 97.5])
            rows.append(dict(model=name, metric=mname, value=pt[mname], ci_lo=lo, ci_hi=hi, n=pt["n"], n_go=pt["n_go"]))
    df = pd.DataFrame(rows)
    if ref is not None and ref in boot:
        d = []
        for name, bs in boot.items():
            for mname in ("macro_f1", "auroc", "brier"):
                diff = bs[mname] - boot[ref][mname]
                lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                d.append(dict(model=name, metric=mname, diff_vs_ref=float(np.nanmean(diff)), diff_lo=lo, diff_hi=hi))
        df = df.merge(pd.DataFrame(d), on=["model", "metric"])
        df["ref"] = ref
    return df
