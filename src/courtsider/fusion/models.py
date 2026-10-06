"""Turn per-segment text probabilities into calibrated, time-stamped event beliefs.

Three options, all strictly causal (each output uses only segments that have already ended):
  * raw       — the segment classifier's probability, isotonic-calibrated
  * hmm       — per-event two-state (quiet / event-just-happened) HMM forward filter whose
                emissions are Gaussian on the classifier logit; interpretable, 6 parameters/event
  * gru       — a small GRU over the segment sequence (probabilities + inter-segment gaps)
Fusion models are fit on the validation split; everything is reported on the test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

from courtsider.data.taxonomy import EVENT_TYPES

P_COLS = [f"p_{e}" for e in EVENT_TYPES]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Calibrator:
    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        self.iso = [
            IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[:, k], y[:, k]) for k in range(p.shape[1])
        ]
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.column_stack([iso.predict(p[:, k]) for k, iso in enumerate(self.iso)])


class HMMFilter:
    """Two hidden states per event; Gaussian emissions on logit(p); forward filtering per game."""

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> HMMFilter:
        x = _logit(df[P_COLS].to_numpy())
        self.params = []
        for k in range(len(EVENT_TYPES)):
            yk, xk = y[:, k].astype(bool), x[:, k]
            mu = (xk[~yk].mean(), xk[yk].mean() if yk.any() else xk.mean() + 3)
            sd = (xk[~yk].std() + 1e-3, xk[yk].std() + 1e-3 if yk.any() else 1.0)
            # transitions estimated from label sequences within games
            a01 = a10 = 1e-3
            n0 = n1 = 1.0
            for _, idx in df.groupby("game_id", sort=False).indices.items():
                s = yk[idx]
                a01 += (~s[:-1] & s[1:]).sum()
                a10 += (s[:-1] & ~s[1:]).sum()
                n0 += (~s[:-1]).sum()
                n1 += s[:-1].sum()
            self.params.append(dict(mu=mu, sd=sd, p01=a01 / n0, p10=a10 / n1))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = _logit(df[P_COLS].to_numpy())
        out = np.zeros_like(x)
        for _, idx in df.groupby("game_id", sort=False).indices.items():
            for k, prm in enumerate(self.params):
                T = np.array([[1 - prm["p01"], prm["p01"]], [prm["p10"], 1 - prm["p10"]]])
                belief = np.array([1 - prm["p01"], prm["p01"]])
                for j, i in enumerate(idx):
                    if j:
                        belief = belief @ T
                    lik = np.array(
                        [np.exp(-0.5 * ((x[i, k] - prm["mu"][s]) / prm["sd"][s]) ** 2) / prm["sd"][s] for s in (0, 1)]
                    )
                    belief = belief * lik
                    belief /= belief.sum() + 1e-300
                    out[i, k] = belief[1]
        return out


class _GRU(torch.nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden: int = 64):
        super().__init__()
        self.gru = torch.nn.GRU(n_in, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, n_out)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h)


class GRUFusion:
    def _features(self, df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for _, idx in df.groupby("game_id", sort=False).indices.items():
            g = df.iloc[idx]
            gap = np.diff(g.g_end.to_numpy(), prepend=g.g_end.iloc[0])
            feats = np.column_stack(
                [_logit(g[P_COLS].to_numpy()) / 4, np.clip(gap, 0, 30) / 10, (g.g_end - g.g_start).to_numpy() / 5]
            )
            out.append((idx, feats.astype(np.float32)))
        return out

    def fit(self, df: pd.DataFrame, y: np.ndarray, epochs: int = 30, seed: int = 0) -> GRUFusion:
        torch.manual_seed(seed)
        seqs = self._features(df)
        self.net = _GRU(seqs[0][1].shape[1], y.shape[1])
        loss_fn = torch.nn.BCEWithLogitsLoss()  # unweighted so outputs stay calibrated probabilities
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        for _ in range(epochs):
            for i in np.random.default_rng(seed).permutation(len(seqs)):
                idx, f = seqs[i]
                logits = self.net(torch.from_numpy(f)[None])[0]
                loss = loss_fn(logits, torch.tensor(y[idx], dtype=torch.float32))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(df), len(EVENT_TYPES)))
        for idx, f in self._features(df):
            out[idx] = torch.sigmoid(self.net(torch.from_numpy(f)[None])[0]).numpy()
        return out
