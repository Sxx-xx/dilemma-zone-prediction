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
