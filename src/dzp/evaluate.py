"""Evaluation: metrics at tau = 0 / 0.5 / 1 s after yellow onset, two split schemes.

  * within-site: GroupKFold over yellow cycles (vehicles from the same cycle never straddle
    train/test -- they share the same signal event and often interact).
  * LOIO: leave-one-intersection-out (train on all other intersections, test on the held-out one).

Positive class for precision/recall/F1 = STOP (label 0): missing a human driver's stop is the
rear-end-risk error for a following AV. Macro-F1 is reported as the symmetric summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from .models import all_models

TAUS = (0.0, 0.5, 1.0)


def metrics(y, p) -> dict:
    ys, ps = 1 - np.asarray(y), 1 - np.asarray(p)  # stop = positive
    return dict(
        acc=accuracy_score(y, p),
        prec_stop=precision_score(ys, ps, zero_division=0),
        rec_stop=recall_score(ys, ps, zero_division=0),
        f1_stop=f1_score(ys, ps, zero_division=0),
        macro_f1=f1_score(y, p, average="macro", zero_division=0),
        n=len(y),
    )


def _macro_f1_fast(y: np.ndarray, p: np.ndarray) -> float:
    """numpy macro-F1 for binary labels (same value as sklearn, zero_division=0)."""
    f = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c))
        den = np.sum(p == c) + np.sum(y == c)
        f.append(2 * tp / den if den else 0.0)
    return float(np.mean(f))


def slice_at(seq: np.ndarray, onset_idx: int, tau: float, dt: float) -> np.ndarray:
    """Sequence truncated so that its last step is onset + tau."""
    return seq[:, : onset_idx + int(round(tau / dt)) + 1]


def _splits(ev: pd.DataFrame, scheme: str, n_folds: int):
    if scheme == "loio":
        for site in sorted(ev["intersection"].unique()):
            te = (ev["intersection"] == site).values
            yield site, np.where(~te)[0], np.where(te)[0]
    else:
        groups = ev["intersection"] + ":" + ev["cycle_id"].astype(str)
        k = min(n_folds, groups.nunique())
        for i, (tr, te) in enumerate(GroupKFold(k).split(ev, groups=groups)):
            yield f"fold{i}", tr, te


def run(ev: pd.DataFrame, seq: np.ndarray, onset_idx: int, dt: float, scheme: str = "within",
        n_folds: int = 5, models=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (metrics, preds). metrics: per-(model, tau) on the pooled out-of-fold predictions,
    plus per-held-out-site rows for LOIO. preds: long table of out-of-fold predictions."""
    y = ev["label"].values.astype(int)
    rows, preds = [], []
    for tau in TAUS:
        s = slice_at(seq, onset_idx, tau, dt)
        for M in models or all_models():
            pred = np.full(len(y), -1)
            for split, tr, te in _splits(ev, scheme, n_folds):
                if len(np.unique(y[tr])) < 2:
                    continue  # cannot train a classifier on one class
                m = M().fit(s[tr], y[tr])
                pred[te] = m.predict(s[te])
                if scheme == "loio":
                    rows.append(dict(model=M.name, tau=tau, split=split, **metrics(y[te], pred[te])))
            ok = pred >= 0
            rows.append(dict(model=M.name, tau=tau, split="pooled", **metrics(y[ok], pred[ok])))
            preds.append(pd.DataFrame(dict(row=np.arange(len(y)), model=M.name, tau=tau, y=y, pred=pred)))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def bootstrap_vs_baseline(ev: pd.DataFrame, preds: pd.DataFrame, baseline: str = "rule_d/v<t_r",
                          n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Paired cluster bootstrap (resampling yellow cycles) of macro-F1(model) - macro-F1(baseline)."""
    rng = np.random.default_rng(seed)
    cyc = (ev["intersection"] + ":" + ev["cycle_id"].astype(str)).values
    ucyc = np.unique(cyc)
    members = {c: np.where(cyc == c)[0] for c in ucyc}
    out = []
    for tau, g in preds.groupby("tau"):
        P = {m: gg.sort_values("row") for m, gg in g.groupby("model")}
        base = P[baseline]
        for m, pm in P.items():
            if m == baseline:
                continue
            diffs = []
            for _ in range(n_boot):
                idx = np.concatenate([members[c] for c in rng.choice(ucyc, len(ucyc))])
                ok = (pm["pred"].values[idx] >= 0) & (base["pred"].values[idx] >= 0)
                yb = pm["y"].values[idx][ok]
                diffs.append(_macro_f1_fast(yb, pm["pred"].values[idx][ok]) - _macro_f1_fast(yb, base["pred"].values[idx][ok]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            out.append(dict(tau=tau, model=m, baseline=baseline, metric="macro_f1",
                            mean_diff=float(np.mean(diffs)), ci_lo=lo, ci_hi=hi, p_le_0=float(np.mean(np.array(diffs) <= 0))))
    return pd.DataFrame(out)
