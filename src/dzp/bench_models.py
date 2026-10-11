"""Benchmark predictors with one interface (1 = go, 0 = stop):

    m.fit(F, S, y);  m.score(F, S) -> real-valued go score;  m.proba(F, S) -> P(go) in [0, 1]

F: DataFrame of instantaneous features at the prediction time (dzp.features: d, v, a, t_r, ...).
S: (N, T, 4) sequences (x, y, v, a) ending at the prediction time.
Rules are not trained and have no calibrated probability: proba = hard 0/1 decision (so their Brier score is the
error rate), score = their own margin (used for AUROC). No hyper-parameter tuning anywhere.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(4)
DVA = ["d", "v", "a"]


class RuleA:
    """Go iff the stop line is reached before red at constant speed: d / v < t_r."""

    name = "ruleA_d/v<t_r"

    def __init__(self, **_):
        pass

    def fit(self, F, S, y):
        return self

    def score(self, F, S):
        return F["t_r"].values - F["tt"].values

    def proba(self, F, S):
        return ((F["d"].values <= 0) | (self.score(F, S) > 0)).astype(float)


class RuleB:
    """Type I dilemma-zone rule: go iff stopping needs more than `decel` m/s^2 (v^2 / 2d > decel)."""

    name = "ruleB_v2/2d>b"

    def __init__(self, decel: float = 3.0, **_):
        self.decel = decel

    def fit(self, F, S, y):
        return self

    def score(self, F, S):
        return np.clip(F["dreq"].values, 0, 50) - self.decel

    def proba(self, F, S):
        return ((F["d"].values <= 0) | (self.score(F, S) > 0)).astype(float)


class _Tab:
    def __init__(self, est):
        self.est = est

    def fit(self, F, S, y):
        self.est.fit(F[DVA].values, y)
        return self

    def proba(self, F, S):
        return self.est.predict_proba(F[DVA].values)[:, 1]

    score = proba


class LogRegDVA(_Tab):
    name = "logreg_dva"

    def __init__(self, **_):
        super().__init__(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)))


class MLPDVA(_Tab):
    name = "mlp_dva"

    def __init__(self, seed: int = 0, **_):
        super().__init__(make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2,
                                                                       max_iter=3000, random_state=seed)))


class _LSTMNet(torch.nn.Module):
    def __init__(self, c_in, h=32):
        super().__init__()
        self.rnn = torch.nn.LSTM(c_in, h, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(h, 1))

    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        return self.head(hn[-1]).squeeze(-1)


class _TransformerNet(torch.nn.Module):
    def __init__(self, c_in, d_model=32, n_head=2, n_layers=1, T=20):
        super().__init__()
        self.inp = torch.nn.Linear(c_in, d_model)
        self.pos = torch.nn.Parameter(torch.zeros(1, T, d_model))
        layer = torch.nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, n_layers)
        self.head = torch.nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.enc(self.inp(x) + self.pos[:, : x.shape[1]])
        return self.head(h[:, -1]).squeeze(-1)          # read out at the prediction time step


class _Seq:
    """Seed ensemble of a small sequence net on the (x, y, v, a) window; class-weighted BCE."""

    net_cls = None

    def __init__(self, n_seeds: int = 3, epochs: int = 200, lr: float = 3e-3, **_):
        self.n_seeds, self.epochs, self.lr = n_seeds, epochs, lr

    def _x(self, S):
        return torch.from_numpy(((S - self.mu) / self.sd).astype(np.float32))

    def fit(self, F, S, y):
        flat = S.reshape(-1, S.shape[-1])
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-6
        X, Y = self._x(S), torch.tensor(y, dtype=torch.float32)
        pos = float(Y.mean())
        pw = torch.tensor((1 - pos) / max(pos, 1e-3))
        self.nets = []
        for s in range(self.n_seeds):
            torch.manual_seed(s)
            net = self.net_cls(S.shape[-1])
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
            lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
            g = torch.Generator().manual_seed(s)
            for _ in range(self.epochs):
                net.train()
                for idx in torch.randperm(len(X), generator=g).split(64):
                    opt.zero_grad()
                    lossf(net(X[idx]), Y[idx]).backward()
                    opt.step()
            net.eval()
            self.nets.append(net)
        return self

    def proba(self, F, S):
        with torch.no_grad():
            X = self._x(S)
            return np.mean([torch.sigmoid(n(X)).numpy() for n in self.nets], 0)

    score = proba


class LSTMSeq(_Seq):
    name = "lstm_xyva"
    net_cls = _LSTMNet


class TransformerSeq(_Seq):
    name = "transformer_xyva"
    net_cls = _TransformerNet


def model_list(decel: float = 3.0, transformer: bool = True, n_seeds: int = 3):
    ms = [(RuleA, {}), (RuleB, {"decel": decel}), (LogRegDVA, {}), (MLPDVA, {}), (LSTMSeq, {"n_seeds": n_seeds})]
    if transformer:
        ms.append((TransformerSeq, {"n_seeds": n_seeds}))
    return ms
