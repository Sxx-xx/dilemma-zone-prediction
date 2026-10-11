"""Leader / follower view of the yellow-onset events (same dataset arguments as run_benchmark.py).

At yellow onset each through event is a platoon *leader* (no unpassed motor vehicle ahead in its lane) or a
*follower* (features.build ldr_* columns). This script writes, to --out:
  roles.csv          role x stop/go counts (+ per group)
  lead_features.csv  followers only, leave-one-group-out: rule A, LR(d,v,a), LR(d,v) vs LR/MLP with leader features
                     (gap, dv, a_lead) and with the leader's own outcome ("[oracle]": assumes the leader's behaviour
                     is known); bootstrap differences vs rule A and vs LR(d,v,a)
  risk_pairs.csv     (leader outcome, follower outcome) pairs; leader stops + follower goes = rear-end risk pattern
  all_trained_by_role.csv  models trained on all through events (leave-one-group-out), scored on leaders / followers
  risk_recall.csv    per model, share of follower 'go' predicted as go on those risk pairs (and on all follower go)
  fig_risk_recall.png
The per-split model comparison itself is run_benchmark.py --split leader|follower.

  .venv/bin/python scripts/leader_follower.py --dataset fluid --root data/raw/fluid/FI/derived_data --site FI \
      --approaches W_approach,E_approach --light-map W_approach=light_E_s,E_approach=light_E_s \
      --out results/fluid_fi/leader_follower
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dzp import bench_eval
from dzp.bench_models import (LogRegDV, LogRegDVA, LogRegLead, LogRegLeadOracle, MLPLead, MLPLeadOracle, RuleA,
                              model_list)
from run_benchmark import at, extract, through


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fluid", "sind"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--site", default="")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--approaches", default="")
    ap.add_argument("--light-map", default="")
    ap.add_argument("--group", default="recording", choices=["recording", "intersection"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--decel", type=float, default=3.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--no-transformer", action="store_true")
    args = ap.parse_args()
    args.leads = [0.0]
    os.makedirs(args.out, exist_ok=True)

    ev_all, feat_all, seq_all = extract(args, 0.0)
    _, ev, feat, seqs = through(ev_all, feat_all, seq_all)
    F, S = at(feat, seqs, 0.0, np.arange(len(ev)))
    y = ev["label"].values.astype(int)
    fol = F["ldr_id"].values >= 0
    role = np.where(fol, "follower", "leader")

    # ---- 1. roles
    r = pd.DataFrame(dict(group=ev["group"], role=role, label=np.where(y == 1, "go", "stop")))
    tab = pd.crosstab([r["group"], r["role"]], r["label"]).reset_index()
    tot = pd.crosstab(r["role"], r["label"]).reset_index().assign(group="ALL")
    roles = pd.concat([tot, tab], ignore_index=True).fillna(0)
    roles["n"] = roles["go"] + roles["stop"]
    roles["go_rate"] = roles["go"] / roles["n"]
    roles[["group", "role", "n", "stop", "go", "go_rate"]].to_csv(f"{args.out}/roles.csv", index=False)
    print(roles.to_string(), flush=True)
    # consistency: leader outcome from its own trajectory vs its event label when the leader is itself an event
    lab = {(g, fy, t): l for g, fy, t, l in zip(ev_all["group"], ev_all["f_yellow"], ev_all["track_id"], ev_all["label"])}
    own = np.array([lab.get((g, fy, t), np.nan) for g, fy, t in zip(ev["group"], ev["f_yellow"], F["ldr_id"])])
    both = fol & np.isfinite(own)
    print(f"leader outcome vs leader's own event label: {int((own[both] == F['ldr_label'].values[both]).sum())}"
          f"/{int(both.sum())} agree", flush=True)

    # ---- 3. followers: leader features
    fi = np.where(fol & np.isfinite(F["ldr_label"].values))[0]
    ev_f, F_f, S_f = ev.iloc[fi].reset_index(drop=True), F.iloc[fi].reset_index(drop=True), S[fi]
    lm = [(RuleA, {}), (LogRegDVA, {}), (LogRegDV, {}), (LogRegLead, {}), (MLPLead, {}), (LogRegLeadOracle, {}),
          (MLPLeadOracle, {})]
    P_f = bench_eval.oof(lm, F_f, S_f, ev_f, "group")
    b1 = bench_eval.bootstrap(ev_f, P_f, None, args.n_boot, ref=RuleA.name)
    b2 = bench_eval.bootstrap(ev_f, P_f, None, args.n_boot, ref=LogRegDVA.name)
    b = b1.merge(b2[["model", "metric", "diff_vs_ref", "diff_lo", "diff_hi"]], on=["model", "metric"],
                 suffixes=("_ruleA", "_lr_dva"))
    b["oracle"] = b["model"].str.contains("oracle")
    b.drop(columns="ref").to_csv(f"{args.out}/lead_features.csv", index=False)
    print(b[b.metric != "brier"].round(3).to_string(), flush=True)

    # ---- 4. risk pairs: (leader outcome, follower outcome)
    lo = F["ldr_label"].values
    pairs = pd.crosstab(pd.Series(np.where(lo[fi] == 1, "leader_go", "leader_stop"), name="leader"),
                        pd.Series(np.where(y[fi] == 1, "follower_go", "follower_stop"), name="follower"))
    pairs = pairs.stack().rename("n").reset_index()
    pairs["share_of_pairs"] = pairs["n"] / len(fi)
    pairs.to_csv(f"{args.out}/risk_pairs.csv", index=False)
    print(pairs.to_string(), flush=True)
    # recall of follower 'go': models trained on all through events (main benchmark set) + follower-only models
    P_all = bench_eval.oof(model_list(args.decel, not args.no_transformer, args.n_seeds), F, S, ev, "group")
    risk = fol & (lo == 0) & (y == 1)
    fgo = fol & (y == 1)
    rows = []
    for train, P, pos in (("all_through", P_all, np.arange(len(ev))), ("followers_only", P_f, fi)):
        full = np.full(len(ev), np.nan)
        for name, (p, _) in P.items():
            full[:] = np.nan
            full[pos] = p
            for subset, m in (("risk_pairs", risk), ("all_follower_go", fgo)):
                hit = full[m] >= 0.5
                rows.append(dict(train=train, model=name, subset=subset, n=int(m.sum()), predicted_go=int(hit.sum()),
                                 recall=hit.mean() if m.any() else np.nan))
    rec = pd.DataFrame(rows)
    # models trained on all through events, scored per role (separates the role effect from the small training set)
    ref = model_list()[0][0].name
    br = pd.concat([bench_eval.bootstrap(ev, P_all, role == rl, args.n_boot, ref=ref).assign(role=rl)
                    for rl in ("leader", "follower")], ignore_index=True)
    br.to_csv(f"{args.out}/all_trained_by_role.csv", index=False)
    print(br[br.metric == "macro_f1"].round(3).to_string(), flush=True)
    rec.to_csv(f"{args.out}/risk_recall.csv", index=False)
    print(rec.round(3).to_string(), flush=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    rr = rec[rec.subset == "risk_pairs"]
    lbl = [f"{m}\n({t})" for t, m in zip(rr["train"], rr["model"])]
    ax.bar(np.arange(len(rr)), rr["recall"], color=np.where(rr["train"] == "all_through", "tab:blue", "tab:orange"))
    ax.set_xticks(np.arange(len(rr)), lbl, rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall of follower 'go'")
    ax.set_title(f"risk pairs: leader stops, follower goes (n={int(risk.sum())})", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{args.out}/fig_risk_recall.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
