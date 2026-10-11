"""The four compared predictors. All share one interface:

    m.fit(seq_train, y_train); m.predict(seq_test) -> 0/1   (1 = go, 0 = stop)

`seq` has shape (N, T, 4) with channels (d, v, a, t_red) and its LAST time step is the
decision time (yellow onset + tau). Instantaneous models only look at the last step.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D, V, A, T_RED = 0, 1, 2, 3
torch.set_num_threads(4)  # tiny models: more threads only add overhead (and thrash when runs are parallel)


class RuleDV:
    """Prior C-ITS rule: go iff the vehicle reaches the stop line before the light turns red,
    assuming constant speed: d / v < t_r. No training."""

    name = "rule_d/v<t_r"

    def fit(self, seq, y):
        return self

    def predict(self, seq):
        last = seq[:, -1]
        d, v, tr = last[:, D], last[:, V], last[:, T_RED]
        tt = np.where(v > 0.1, d / np.maximum(v, 0.1), np.inf)
        return ((d <= 0) | (tt < tr)).astype(int)


def instant_features(seq: np.ndarray) -> np.ndarray:
    last = seq[:, -1]
    d, v, a, tr = last[:, D], last[:, V], last[:, A], last[:, T_RED]
    # d/v - t_r is the rule's own margin; giving it to the learners keeps the comparison fair
    margin = np.clip(d / np.maximum(v, 0.1), 0, 20) - tr
    return np.stack([d, v, a, tr, margin], 1)


class _Instant:
    def __init__(self, est):
        self.est = est

    def fit(self, seq, y):
        self.est.fit(instant_features(seq), y)
        return self

    def predict(self, seq):
        return self.est.predict(instant_features(seq))


class LogReg(_Instant):
    name = "logreg"

    def __init__(self):
        super().__init__(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)))


class MLP(_Instant):
    name = "mlp"

    def __init__(self, seed: int = 0):
        super().__init__(make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=3000, random_state=seed)))


class _Net(torch.nn.Module):
    def __init__(self, c_in: int, h: int):
        super().__init__()
        self.lstm = torch.nn.LSTM(c_in, h, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(h, 1))

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.head(hn[-1]).squeeze(-1)


class LSTM:
    """LSTM over the last `window` steps of (d, v, a, t_red). Small ensemble of seeds because
    the datasets are tiny and single runs are noisy."""

    name = "lstm"

    def __init__(self, window: int = 20, hidden: int = 32, epochs: int = 200, n_seeds: int = 5, lr: float = 3e-3):
        self.window, self.hidden, self.epochs, self.n_seeds, self.lr = window, hidden, epochs, n_seeds, lr

    def _x(self, seq):
        x = seq[:, -self.window:].astype(np.float32)
        return torch.from_numpy(((x - self.mu) / self.sd).astype(np.float32))

    def fit(self, seq, y):
        x = seq[:, -self.window:].reshape(-1, seq.shape[-1])
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        X, Y = self._x(seq), torch.tensor(y, dtype=torch.float32)
        pos = float(Y.mean())
        pw = torch.tensor((1 - pos) / max(pos, 1e-3))
        self.nets = []
        for s in range(self.n_seeds):
            torch.manual_seed(s)
            net = _Net(seq.shape[-1], self.hidden)
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

    def predict_proba(self, seq):
        with torch.no_grad():
            X = self._x(seq)
            return np.mean([torch.sigmoid(n(X)).numpy() for n in self.nets], 0)

    def predict(self, seq):
        return (self.predict_proba(seq) >= 0.5).astype(int)


def all_models():
    return [RuleDV, LogReg, MLP, LSTM]
