import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dzp.events import label_event
from dzp.models import RuleDV

DT = 0.1


def _traj(v0, decel, n=300, d0=40.0):
    v = np.maximum(v0 - decel * DT * np.arange(n), 0.0)
    d = d0 - np.concatenate([[0], np.cumsum(v[:-1] * DT)])
    return np.arange(n), d, v


def test_go_when_crossing_during_yellow():
    fr, d, v = _traj(15, 0)  # 40 m at 15 m/s -> crosses at ~2.7 s
    t_c = fr[np.argmax(d <= 0)]
    assert label_event(fr, d, v, 0, t_c, f_red=30, f_green=900, dt=DT) == (1, "yellow")


def test_stop_when_halting_upstream():
    fr, d, v = _traj(10, 3.0)  # stops after ~16.7 m
    assert label_event(fr, d, v, 0, np.nan, f_red=30, f_green=900, dt=DT) == (0, "halt")


def test_late_crossing_without_halt_is_unresolved():
    fr, d, v = _traj(2.5, 0, n=400)  # creeps across ~16 s after onset
    t_c = fr[np.argmax(d <= 0)]
    assert label_event(fr, d, v, 0, t_c, f_red=30, f_green=900, dt=DT) is None


def test_rule_dv():
    # channels (d, v, a, t_red); last step is the decision time
    seq = np.array([[[30.0, 15.0, 0.0, 3.0]], [[60.0, 15.0, 0.0, 3.0]], [[-1.0, 5.0, 0.0, 2.0]]])
    assert RuleDV().predict(seq).tolist() == [1, 0, 1]
