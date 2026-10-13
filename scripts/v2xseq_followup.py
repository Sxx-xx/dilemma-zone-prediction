"""Follow-ups on the V2X-Seq leader evaluation (v4 tables, no tuning). Outputs under --out (results/v2xseq/eval/v4/followup).

A. acceleration: why LR(d,v) - LR(d,v,a) is -0.121 on FLUID but -0.009 on V2X-Seq
   accel_stats.csv       onset-a spread, frame-to-frame changes, and the spread of a re-estimated as the least-squares
                         slope of the 2 s speed window over the last 0.2 / 0.5 / 1.0 s (causal), both datasets
   accel_smooth.csv      LR(d,v,a_w) vs LR(d,v), paired cycle bootstrap: V2X-Seq LOIO (sites >= 30 events) and
                         FLUID leave-one-video-out; subsets all / dz / v0 < 10 / v0 >= 10
B. dilemma zone (d/v 2.5-5.5 s at onset) only
   dz_only_loio.csv      models trained AND tested inside the band (LOIO, test sites >= --dz-min-site band events),
                         paired differences vs LR(d,v,a); subsets all / v<10 / v>=10
   fig_dz_boundaries.png P(go) = 0.5 boundaries of LR(d,v,a), LR(d,v), MLP(d,v,a), LSTM(d,v,a) fitted on all v4
                         leaders, drawn on the (d, v) plane at a = 0 (LSTM: constant-speed 2 s window), band shaded
   dz_speed_split.csv    trained-on-all LOIO models tested inside the band, by speed (from ../preds_loio.csv)   [B2]
   dz_inband_vs_all.csv  same band test events: trained-on-all models vs the in-band-trained LR(d,v,a)          [B2]
D. calibrated rule A, fair LOIO (23 folds, every site tested) -> ../calibrated_rule/
   fold_params.csv       per fold: t_r* (logistic d/v boundary of the training sites), d_max* (band go-50 % distance)
   metrics.csv           calibrated rule A (+ distance cap, E), band-only LR(d,v,a)/(d,v), full-training LR/MLP/LSTM;
                         cycle bootstrap, diffs vs calibrated rule A and vs LR(d,v,a); subsets all / dz / v<10 / v>=10
   pairwise_macro_f1.csv all pairwise macro-F1 differences with CIs; tr_fixed_sensitivity.csv (from ruleA_tr_curve.csv)
C. rule A inside the band
   ruleA_tr_sweep.csv    rule A with t_r fixed at 2.5 / 3.0 / 3.5 / 4.0 s vs the recorded t_r (paired bootstrap)
   ruleA_tr_curve.csv    band macro-F1 for t_r = 1.5 ... 6.0 s (point estimates)
   tt50_per_site.csv     logistic P(go | d/v) per test site and pooled: d/v at P = 0.5 with cycle-bootstrap CI,
                         next to the site's median recorded t_r

  nice -n 19 .venv/bin/python scripts/v2xseq_followup.py --tasks A,C      # cheap, no torch
  nice -n 19 .venv/bin/python scripts/v2xseq_followup.py --tasks B,B2
  nice -n 19 .venv/bin/python scripts/v2xseq_followup.py --tasks D --threads 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from dzp import bench_eval
from dzp.bench_models import LogRegDV, LogRegDVA, LSTMSeq, MLPDVA, RuleA
from run_benchmark import at
from seq_dva import LSTMDVA
from v2xseq_eval import DZ_HI, DZ_LO, V_SPLIT, load_tables, loio_folds, subsets

DT = 0.1
WINDOWS = (0.2, 0.5, 1.0)


# ---------------------------------------------------------------- helpers
def load_fluid(event_dir):
    fe = pd.read_csv(f"{event_dir}/events_through.csv")
    ff = pd.read_csv(f"{event_dir}/features_through.csv")
    F, S = at(ff, {0.0: np.load(f"{event_dir}/sequences_through.npz")["lead+0.0"]}, 0.0, np.arange(len(fe)))
    fe["segment"] = fe["group"]
    return fe, F, S


def slope_a(S, w, dt=DT):
    """Least-squares slope of the speed channel over the last round(w/dt)+1 steps of the window (ends at onset)."""
    k = int(round(w / dt))
    v = S[:, -(k + 1):, 2].astype(float)
    t = np.arange(k + 1) * dt
    t = t - t.mean()
    return (v * t).sum(1) / (t ** 2).sum()


def oof(models, F, S, y, folds, seeds=3):
    """Out-of-fold P(go) per model over the given (name, train idx, test idx) folds."""
    P = {}
    for M, kw in models:
        p = np.full(len(y), np.nan)
        for _, tr, te in folds:
            if len(np.unique(y[tr])) < 2:
                continue
            p[te] = M(**kw).fit(F.iloc[tr], S[tr], y[tr]).proba(F.iloc[te], S[te])
        P[M.name] = (p, p)
        print(f"  {M.name} done", flush=True)
    return P


class _Smooth(LogRegDVA):
    """LR(d, v, a_w): a replaced by the causal slope of the speed window."""
    w = 0.5

    def _F(self, F, S):
        return F.assign(a=slope_a(S, self.w))

    def fit(self, F, S, y):
        return super().fit(self._F(F, S), S, y)

    def proba(self, F, S):
        return super().proba(self._F(F, S), S)

    score = proba


def smooth_cls(w):
    return type(f"LogRegDVA_w{w}", (_Smooth,), {"w": w, "name": f"logreg_dva_w{w}"})


# ---------------------------------------------------------------- A
def task_a(args, ev, F, S, out):
    fe, FF, FS = load_fluid(args.fluid_event_dir)
    rows = []
    for name, (E, X, Z) in {"v2xseq": (ev, F, S), "fluid": (fe, FF, FS)}.items():
        a_seq, v = Z[:, :, 3].astype(float), Z[:, :, 2].astype(float)
        r = dict(dataset=name, n=len(X), pad_any=int((X["pad"] > 0).sum()), a_onset_std=X["a"].std(), a_onset_mean=X["a"].mean(),
                 a_seq_std=a_seq.std(), da_frame_abs_mean=np.abs(np.diff(a_seq, axis=1)).mean(),
                 dv_frame_abs_mean=np.abs(np.diff(v, axis=1)).mean(), d2v_frame_std=np.diff(v, 2, axis=1).std(),
                 v_onset_mean=X["v"].mean())
        for w in WINDOWS:
            aw = slope_a(Z, w)
            r[f"a_w{w}_std"] = aw.std()
            r[f"corr_a_w{w}_vs_a"] = np.corrcoef(aw, X["a"])[0, 1]
            r[f"a_w{w}_minus_a_std"] = (aw - X["a"]).std()
        rows.append(r)
    pd.DataFrame(rows).to_csv(f"{out}/accel_stats.csv", index=False)
    print(pd.DataFrame(rows).round(3).T.to_string(), flush=True)

    models = [(LogRegDV, {}), (LogRegDVA, {})] + [(smooth_cls(w), {}) for w in WINDOWS]
    res = []
    for name, (E, X, Z) in {"v2xseq": (ev, F, S), "fluid": (fe, FF, FS)}.items():
        y = E["label"].values.astype(int)
        folds = loio_folds(E, args.min_site if name == "v2xseq" else 1)
        tested = np.zeros(len(y), bool)
        for _, _, te in folds:
            tested[te] = True
        print(f"[A] {name}: {len(folds)} folds, {int(tested.sum())} tested", flush=True)
        P = oof(models, X, Z, y, folds)
        for sub, mask in subsets(E, X).items():
            if (mask & tested).sum() < 10:
                continue
            b = bench_eval.bootstrap(E, P, mask & tested, args.n_boot, ref=LogRegDV.name)
            res.append(b.assign(dataset=name, subset=sub))
    df = pd.concat(res, ignore_index=True)
    df.to_csv(f"{out}/accel_smooth.csv", index=False)
    q = df[df.metric == "macro_f1"]
    print(q[["dataset", "subset", "model", "value", "diff_vs_ref", "diff_lo", "diff_hi", "n"]].round(3).to_string(), flush=True)


# ---------------------------------------------------------------- B
def task_b(args, ev, F, S, out):
    tt = F["tt"].values
    band = (tt >= DZ_LO) & (tt <= DZ_HI)
    E, X, Z = ev[band].reset_index(drop=True), F[band].reset_index(drop=True), S[band]
    y = E["label"].values.astype(int)
    folds = loio_folds(E, args.dz_min_site)
    tested = np.zeros(len(y), bool)
    for _, _, te in folds:
        tested[te] = True
    print(f"[B] band events {len(E)} (go {int(y.sum())}), test sites {len(folds)}, tested {int(tested.sum())}", flush=True)
    models = [(RuleA, {}), (LogRegDVA, {}), (LogRegDV, {}), (MLPDVA, {}), (LSTMSeq, {"n_seeds": args.n_seeds}), (LSTMDVA, {"n_seeds": args.n_seeds})]
    P = oof(models, X, Z, y, folds)
    P[RuleA.name] = ((RuleA().score(X, Z) > 0).astype(float), RuleA().score(X, Z))
    res = []
    for sub, mask in subsets(E, X).items():
        if sub == "dz":
            continue
        for ref in (LogRegDVA.name, RuleA.name):
            b = bench_eval.bootstrap(E, P, mask & tested, args.n_boot, ref=ref)
            res.append(b.assign(subset=sub))
    df = pd.concat(res, ignore_index=True)
    df.to_csv(f"{out}/dz_only_loio.csv", index=False)
    q = df[(df.metric == "macro_f1")]
    print(q[["subset", "ref", "model", "value", "ci_lo", "ci_hi", "diff_vs_ref", "diff_lo", "diff_hi", "n", "n_go"]].round(3).to_string(), flush=True)
    pd.DataFrame({k: v[0] for k, v in P.items()}).assign(segment=E["segment"], track_id=E["track_id"], site=E["group"]).to_csv(f"{out}/preds_dz_only.csv", index=False)
    boundary_figure(args, ev, F, S, out)


def boundary_figure(args, ev, F, S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = ev["label"].values.astype(int)
    fits = [(LogRegDVA, {}, "#2a78d6", "LR (d, v, a)"), (LogRegDV, {}, "#eb6834", "LR (d, v)"),
            (MLPDVA, {}, "#1baf7a", "MLP (d, v, a)"), (LSTMDVA, {"n_seeds": args.n_seeds}, "#6250d6", "LSTM (d, v, a)")]
    dg, vg = np.meshgrid(np.linspace(0.5, 120, 240), np.linspace(0.5, 25, 200))
    G = pd.DataFrame({"d": dg.ravel(), "v": vg.ravel(), "a": 0.0})
    G["tt"], G["t_r"] = G["d"] / G["v"], 4.0
    # constant-speed window ending at (d, v): x = -(d + v * (19 - i) dt), lat 0, v, a 0
    steps = (np.arange(20)[::-1]) * DT
    GS = np.zeros((len(G), 20, 4), np.float32)
    GS[:, :, 0] = -(G["d"].values[:, None] + G["v"].values[:, None] * steps[None, :])
    GS[:, :, 2] = G["v"].values[:, None]
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=160)
    ax.set_facecolor("#fcfcfb")
    vv = np.linspace(0, 25, 50)
    ax.fill_between(vv, DZ_LO * vv, DZ_HI * vv, color="#e6e4df", alpha=0.6, lw=0, label="dilemma-zone band (d/v 2.5–5.5 s)")
    ax.plot(vv, 4.0 * vv, color="#52514e", lw=1.2, ls="--", label="rule A, t_r = 4.0 s (d = 4 v)")
    go, st = y == 1, y == 0
    ax.scatter(F["v"][st], F["d"][st], s=7, c="#0b0b0b", alpha=0.35, lw=0, label=f"stop (n={int(st.sum())})")
    ax.scatter(F["v"][go], F["d"][go], s=9, facecolors="none", edgecolors="#52514e", alpha=0.45, lw=0.5, label=f"go (n={int(go.sum())})")
    rows = []
    for M, kw, col, lab in fits:
        m = M(**kw).fit(F, S, y)
        p = m.proba(G, GS).reshape(dg.shape)
        cs = ax.contour(vg, dg, p, levels=[0.5], colors=[col], linewidths=2)
        ax.plot([], [], color=col, lw=2, label=f"{lab}: P(go) = 0.5")
        for seg in cs.allsegs[0]:
            rows += [dict(model=M.name, v=a, d=b) for a, b in seg]
        print(f"  boundary {M.name} drawn", flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/dz_boundaries.csv", index=False)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 120)
    ax.set_xlabel("speed at yellow onset v (m/s)")
    ax.set_ylabel("distance to stop line d (m)")
    ax.set_title("V2X-Seq leaders (n=%d): P(go)=0.5 boundaries at a = 0, fitted on all events" % len(y), fontsize=10)
    ax.grid(color="#e6e4df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_dz_boundaries.png")
    plt.close(fig)


def task_b2(args, ev, F, S, out):
    """Band comparisons from saved out-of-fold predictions (no refits):
    dz_speed_split.csv   trained-on-all LOIO models (../preds_loio.csv) inside the band, by speed, vs LR(d,v,a)
    dz_inband_vs_all.csv on the band events tested by task B: trained-on-all models vs the in-band-trained LR(d,v,a)"""
    key = ["segment", "track_id"]
    pa = pd.read_csv(f"{out}/../preds_loio.csv").drop(columns=["site"])
    pb = pd.read_csv(f"{out}/preds_dz_only.csv").drop(columns=["site"])
    names = [c for c in pa.columns if c not in key]
    E = ev[key].merge(pa, on=key, how="left")
    assert len(E) == len(ev)
    tt = F["tt"].values
    band = (tt >= DZ_LO) & (tt <= DZ_HI)
    P = {m: (E[m].values, E[m].values) for m in names}
    tested = np.isfinite(E[LogRegDVA.name].values)
    rows = []
    for sub, m in [("dz", band), ("dz&v<10", band & (F["v"].values < V_SPLIT)), ("dz&v>=10", band & (F["v"].values >= V_SPLIT))]:
        rows.append(bench_eval.bootstrap(ev, P, m & tested, args.n_boot, ref=LogRegDVA.name).assign(subset=sub))
    d1 = pd.concat(rows, ignore_index=True)
    d1.to_csv(f"{out}/dz_speed_split.csv", index=False)
    print(d1[d1.metric == "macro_f1"][["subset", "model", "value", "diff_vs_ref", "diff_lo", "diff_hi", "n", "n_go"]].round(3).to_string(), flush=True)
    bn = [c for c in pb.columns if c not in key]
    Eb = ev[key].merge(pb.rename(columns={c: f"inband_{c}" for c in bn}), on=key, how="left")
    ok = np.isfinite(Eb[f"inband_{LogRegDVA.name}"].values) & tested
    P2 = dict(P)
    for c in bn:
        if c != RuleA.name:
            P2[f"inband_{c}"] = (Eb[f"inband_{c}"].values, Eb[f"inband_{c}"].values)
    rows = []
    for sub, m in [("dz", ok), ("dz&v<10", ok & (F["v"].values < V_SPLIT)), ("dz&v>=10", ok & (F["v"].values >= V_SPLIT))]:
        rows.append(bench_eval.bootstrap(ev, P2, m, args.n_boot, ref=f"inband_{LogRegDVA.name}").assign(subset=sub))
    d2 = pd.concat(rows, ignore_index=True)
    d2.to_csv(f"{out}/dz_inband_vs_all.csv", index=False)
    print(d2[d2.metric == "macro_f1"][["subset", "model", "value", "diff_vs_ref", "diff_lo", "diff_hi", "n", "n_go"]].round(3).to_string(), flush=True)


# ---------------------------------------------------------------- D
def _t50(x, y):
    """d/v (or d) at P(go) = 0.5 from a univariate logistic fit; NaN when the slope is not negative."""
    if len(np.unique(y)) < 2:
        return np.nan
    lr = LogisticRegression(max_iter=1000).fit(x[:, None], y)
    b1, b0 = lr.coef_[0, 0], lr.intercept_[0]
    return -b0 / b1 if b1 < 0 else np.nan


def pairwise(ev, P, mask, n_boot, seed=0):
    """All pairwise macro-F1 differences with cycle-bootstrap 95 % CIs (same draws as bench_eval.bootstrap)."""
    y = ev["label"].values.astype(int)
    cyc = bench_eval.cycles(ev)
    ucyc = np.unique(cyc[mask])
    members = {c: np.where((cyc == c) & mask)[0] for c in ucyc}
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([members[c] for c in rng.choice(ucyc, len(ucyc))]) for _ in range(n_boot)]
    names = list(P)
    f1 = {}
    for n in names:
        p = P[n][0]
        f1[n] = np.array([bench_eval._macro_f1(y[ix], (p[ix] >= 0.5).astype(int)) for ix in draws])
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = f1[a] - f1[b]
            rows.append(dict(model_a=a, model_b=b, diff=float(d.mean()), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5))))
    return pd.DataFrame(rows)


def task_d(args, ev, F, S, out):
    """Calibrated rule A under full LOIO (every site tested, 23 folds): t_r* from the training sites' leaders
    (univariate logistic on d/v, d/v <= --tt-max), band-only LR(d,v,a)/(d,v), full-training LR/MLP/LSTM, and
    the distance-capped rule (E): go iff d/v < t_r* and d < d_max*, d_max* = band go-rate-50 % distance."""
    out = f"{out}/../calibrated_rule"
    os.makedirs(out, exist_ok=True)
    y = ev["label"].values.astype(int)
    tt, d, tr = F["tt"].values, F["d"].values, F["t_r"].values
    band = (tt >= DZ_LO) & (tt <= DZ_HI)
    folds = loio_folds(ev, 1)
    print(f"[D] {len(folds)} folds, band events {int(band.sum())}", flush=True)
    names = {"ruleA_calibrated": None, "ruleA_calibrated_dcap": None, "ruleA_calibrated_dcap50": None, "band_logreg_dva": None, "band_logreg_dv": None}
    P = {k: np.full(len(y), np.nan) for k in names}
    params = []
    for site, trn, te in folds:
        keep = trn[tt[trn] <= args.tt_max]
        t50 = _t50(tt[keep], y[keep])
        btr = trn[band[trn]]
        d50 = _t50(d[btr], y[btr])
        P["ruleA_calibrated"][te] = (tt[te] < t50).astype(float)
        # E: the cap only exists when the band go rate crosses 50 % at some positive distance; otherwise no cap
        cap = d50 if (np.isfinite(d50) and d50 > 0) else np.inf
        P["ruleA_calibrated_dcap"][te] = ((tt[te] < t50) & (d[te] < cap)).astype(float)
        P["ruleA_calibrated_dcap50"][te] = ((tt[te] < t50) & (d[te] < args.d_cap)).astype(float)
        bte = te[band[te]]
        if len(bte) and len(np.unique(y[btr])) == 2:
            P["band_logreg_dva"][bte] = LogRegDVA().fit(F.iloc[btr], S[btr], y[btr]).proba(F.iloc[bte], S[bte])
            P["band_logreg_dv"][bte] = LogRegDV().fit(F.iloc[btr], S[btr], y[btr]).proba(F.iloc[bte], S[bte])
        params.append(dict(test_site=site, n_test=len(te), n_test_band=len(bte), n_train=len(trn), n_train_band=len(btr),
                           t_r_star=t50, d_max_star=d50, d_cap_used=cap, t_r_recorded_median=float(np.median(tr[te])),
                           band_go_rate_train=float(y[btr].mean())))
    pr = pd.DataFrame(params)
    pr.to_csv(f"{out}/fold_params.csv", index=False)
    print(pr.round(2).to_string(), flush=True)
    print(f"t_r* mean {pr.t_r_star.mean():.3f} range {pr.t_r_star.min():.3f}-{pr.t_r_star.max():.3f}; d_max* mean {pr.d_max_star.mean():.1f} range {pr.d_max_star.min():.1f}-{pr.d_max_star.max():.1f}", flush=True)
    full = [(RuleA, {}), (LogRegDVA, {}), (LogRegDV, {}), (MLPDVA, {}), (LSTMSeq, {"n_seeds": args.n_seeds}), (LSTMDVA, {"n_seeds": args.n_seeds})]
    cache = f"{out}/preds_loio23.csv"
    if args.reuse_preds and os.path.exists(cache):
        c = ev[["segment", "track_id"]].merge(pd.read_csv(cache), on=["segment", "track_id"], how="left")
        assert len(c) == len(ev)
        Q = {M.name: (c[M.name].values, c[M.name].values) for M, _ in full if M.name in c}
        print(f"  reusing full-model predictions from {cache}: {list(Q)}", flush=True)
    else:
        Q = oof(full, F, S, y, folds)
    Q[RuleA.name] = ((tr - tt > 0).astype(float), tr - tt)
    for k, v in P.items():
        Q[k] = (v, v)
    # composite: full LR(d,v,a) outside the band, band-only LR(d,v,a) inside (gives the band model an all-event score)
    comp = np.where(band, P["band_logreg_dva"], Q[LogRegDVA.name][0])
    Q["composite_band_lr_dva"] = (comp, comp)
    pd.DataFrame({k: v[0] for k, v in Q.items()}).assign(segment=ev["segment"], track_id=ev["track_id"], site=ev["group"]).to_csv(f"{out}/preds_loio23.csv", index=False)
    rows, pw = [], []
    for sub, mask in [("all", np.ones(len(y), bool)), ("dz", band), ("v<10", F["v"].values < V_SPLIT), ("v>=10", F["v"].values >= V_SPLIT)]:
        Qs = {k: v for k, v in Q.items() if np.isfinite(v[0][mask]).all()}
        for ref in ("ruleA_calibrated", LogRegDVA.name):
            rows.append(bench_eval.bootstrap(ev, Qs, mask, args.n_boot, ref=ref).assign(subset=sub))
        pw.append(pairwise(ev, Qs, mask, args.n_boot).assign(subset=sub))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(f"{out}/metrics.csv", index=False)
    pd.concat(pw, ignore_index=True).to_csv(f"{out}/pairwise_macro_f1.csv", index=False)
    q = df[(df.metric == "macro_f1") & (df.ref == "ruleA_calibrated")]
    print(q[["subset", "model", "value", "ci_lo", "ci_hi", "diff_vs_ref", "diff_lo", "diff_hi", "n", "n_go"]].round(3).to_string(), flush=True)
    cur = pd.read_csv(f"{args.out}/ruleA_tr_curve.csv")
    cur[(cur.t_r >= 2.0) & (cur.t_r <= 4.0)].to_csv(f"{out}/tr_fixed_sensitivity.csv", index=False)


# ---------------------------------------------------------------- C
def task_c(args, ev, F, S, out):
    y = ev["label"].values.astype(int)
    tt, tr = F["tt"].values, F["t_r"].values
    band = (tt >= DZ_LO) & (tt <= DZ_HI)
    P = {RuleA.name: ((tr - tt > 0).astype(float), tr - tt)}
    for t in (2.5, 3.0, 3.5, 4.0):
        P[f"ruleA_tr{t}"] = ((t - tt > 0).astype(float), t - tt)
    res = []
    for sub, mask in [("all", np.ones(len(y), bool)), ("dz", band)]:
        b = bench_eval.bootstrap(ev, P, mask, args.n_boot, ref=RuleA.name)
        res.append(b.assign(subset=sub))
    df = pd.concat(res, ignore_index=True)
    df.to_csv(f"{out}/ruleA_tr_sweep.csv", index=False)
    print(df[df.metric == "macro_f1"][["subset", "model", "value", "ci_lo", "ci_hi", "diff_vs_ref", "diff_lo", "diff_hi"]].round(3).to_string(), flush=True)
    grid = np.arange(1.5, 6.01, 0.25)
    cur = pd.DataFrame([dict(t_r=t, macro_f1_dz=bench_eval._macro_f1(y[band], (tt[band] < t).astype(int)),
                             macro_f1_all=bench_eval._macro_f1(y, (tt < t).astype(int)), pred_go_dz=float((tt[band] < t).mean())) for t in grid])
    cur.to_csv(f"{out}/ruleA_tr_curve.csv", index=False)
    print(cur.round(3).to_string(), flush=True)

    # tt50: logistic P(go | d/v), events with d/v <= --tt-max
    keep = tt <= args.tt_max
    cyc = bench_eval.cycles(ev)
    rng = np.random.default_rng(0)

    def tt50(idx):
        yy, xx = y[idx], tt[idx]
        if len(np.unique(yy)) < 2:
            return np.nan
        lr = LogisticRegression(max_iter=1000).fit(xx[:, None], yy)
        b1, b0 = lr.coef_[0, 0], lr.intercept_[0]
        return -b0 / b1 if b1 < 0 else np.nan

    def boot(idx):
        uc = np.unique(cyc[idx])
        mem = {c: idx[cyc[idx] == c] for c in uc}
        vals = [tt50(np.concatenate([mem[c] for c in rng.choice(uc, len(uc))])) for _ in range(args.n_boot)]
        return np.nanpercentile(vals, [2.5, 97.5])

    rows = []
    sites = ev["group"].value_counts()
    groups = [("pooled", np.ones(len(y), bool))] + [(s, ev["group"].values == s) for s in sites.index if sites[s] >= args.min_site]
    for name, m in groups:
        idx = np.where(m & keep)[0]
        lo, hi = boot(idx)
        rows.append(dict(site=name, n=len(idx), n_go=int(y[idx].sum()), tt50=tt50(idx), tt50_lo=lo, tt50_hi=hi,
                         t_r_median=float(np.median(tr[idx])), go_rate_band=float(y[m & band].mean()) if (m & band).any() else np.nan,
                         n_band=int((m & band).sum())))
        print(f"  tt50 {name}: {rows[-1]['tt50']:.2f} [{lo:.2f}, {hi:.2f}] t_r {rows[-1]['t_r_median']:.1f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{out}/tt50_per_site.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-dir", default="data/raw/v2xseq/results/leaders_v4")
    ap.add_argument("--fluid-event-dir", default="data/raw/fluid/results/benchmark_fi_leader")
    ap.add_argument("--out", default="results/v2xseq/eval/v4/followup")
    ap.add_argument("--tasks", default="A,B,C")
    ap.add_argument("--min-site", type=int, default=30)
    ap.add_argument("--dz-min-site", type=int, default=20)
    ap.add_argument("--tt-max", type=float, default=12.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--d-cap", type=float, default=50.0, help="D/E: fixed distance cap (m) for the sensitivity variant")
    ap.add_argument("--reuse-preds", action="store_true", help="D: reuse full-model LOIO predictions saved by an earlier run")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    ev, F, S = load_tables(args.event_dir)
    print(f"leaders {len(ev)} go {int(ev.label.sum())}", flush=True)
    for t in args.tasks.split(","):
        print(f"=== task {t}", flush=True)
        {"A": task_a, "B": task_b, "B2": task_b2, "C": task_c, "D": task_d}[t](args, ev, F, S, args.out)


if __name__ == "__main__":
    main()
