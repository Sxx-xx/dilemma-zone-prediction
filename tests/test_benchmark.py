import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from dzp import bench_eval
from dzp.bench_models import RuleA, RuleB


def _F(d, v, t_r=3.0):
    d, v = np.asarray(d, float), np.asarray(v, float)
    return pd.DataFrame(dict(d=d, v=v, a=0.0, t_r=t_r, tt=d / np.maximum(v, 0.1),
                             dreq=np.where(d > 0, v ** 2 / (2 * np.maximum(d, 0.1)), np.inf)))


def test_rule_a_constant_speed_reach():
    F = _F([20, 40, -1], [10, 10, 5])           # 2 s < 3 s -> go; 4 s -> stop; past the line -> go
    assert RuleA().proba(F, None).tolist() == [1, 0, 1]


def test_rule_b_required_deceleration():
    F = _F([10, 30], [10, 10])                   # 100/20 = 5 > 3 -> go; 100/60 = 1.7 -> stop
    assert RuleB(decel=3.0).proba(F, None).tolist() == [1, 0]
    assert RuleB(decel=6.0).proba(F, None).tolist() == [0, 0]


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 50), rng.integers(0, 2, 50)
    assert abs(bench_eval._macro_f1(y, p) - f1_score(y, p, average="macro")) < 1e-12


def test_group_split_leaves_each_group_out_once():
    ev = pd.DataFrame(dict(group=list("aabbbc"), label=[0, 1, 0, 1, 1, 0], f_yellow=[1, 1, 2, 2, 3, 4]))
    te = [set(t) for _, t in bench_eval.splits(ev, "group")]
    assert te == [{0, 1}, {2, 3, 4}, {5}]


def _track(tid, x0, v, n=60, length=4.0):
    xs = x0 + v * 0.1 * np.arange(n) if np.ndim(v) == 0 else x0 + np.cumsum(np.r_[0, v[:-1]]) * 0.1
    vv = np.full(n, v, float) if np.ndim(v) == 0 else np.asarray(v, float)
    return pd.DataFrame(dict(track_id=tid, frame_id=np.arange(n), x=xs - length / 2, y=0.0, vx=vv, vy=0.0, heading_rad=0.0,
                             a_lon=0.0, length=length, agent_type="car"))


def test_leader_follower_definition():
    from types import SimpleNamespace

    from dzp import features
    from dzp.geometry import Approach

    ap = Approach(0, np.zeros(2), np.array([0.0, 1.0]), np.array([1.0, 0.0]), 5.0, 0.0)   # travel +x, line at x=0
    stop = np.r_[np.linspace(5, 0, 30), np.zeros(30)]
    tr = pd.concat([_track(1, -30, 6.0), _track(2, -15, stop), _track(3, 5, 6.0), _track(4, -8, 6.0)])
    ev = pd.DataFrame(dict(track_id=[1, 4], sid=0, k0=0, f_red=30))
    f, _ = features.build(SimpleNamespace(tracks=tr), ev, [ap], dt=0.1)
    # fronts at d = 30 (1), 15 (2, stops before the line), 8 (4), -5 (3): the nearest ahead of 1 is 2; gap = 30 - (15 + 4)
    assert f.loc[0, "ldr_id"] == 2 and f.loc[0, "ldr_label"] == 0 and abs(f.loc[0, "ldr_gap"] - 11) < 1e-6
    # track 4: the only vehicle ahead (3) is past the stop line -> platoon leader
    assert f.loc[1, "ldr_id"] == -1
