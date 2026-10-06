"""Dixon–Coles (1997) bivariate-Poisson model with exponential time decay."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def tau(x: np.ndarray, y: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """Low-score dependence correction."""
    t = np.ones_like(lh, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


@dataclass
class DixonColes:
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.25
    rho: float = -0.05
    base: float = 0.1

    def rates(self, home: str, away: str) -> tuple[float, float]:
        a, d = self.attack, self.defence
        lh = np.exp(self.base + self.home_adv + a.get(home, 0.0) + d.get(away, 0.0))
        la = np.exp(self.base + a.get(away, 0.0) + d.get(home, 0.0))
        return float(lh), float(la)

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        lh, la = self.rates(home, away)
        g = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
        x, y = np.meshgrid(g, g, indexing="ij")
        m *= tau(x, y, np.full_like(m, lh), np.full_like(m, la), self.rho)
        return m / m.sum()

    def outcome_probs(self, home: str, away: str) -> np.ndarray:
        m = self.score_matrix(home, away)
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


def fit(matches: pd.DataFrame, as_of: pd.Timestamp, xi: float = 0.0019, l2: float = 0.05) -> DixonColes:
    """Fit on matches strictly before `as_of`; weight = exp(-xi * days_ago). xi≈0.0019 is the D&C value."""
    m = matches[matches.date < as_of]
    teams = sorted(set(m.home) | set(m.away))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = m.home.map(idx).to_numpy()
    ai = m.away.map(idx).to_numpy()
    x = m.home_goals.to_numpy(dtype=int)
    y = m.away_goals.to_numpy(dtype=int)
    w = np.exp(-xi * (as_of - m.date).dt.days.to_numpy())

    def nll(p: np.ndarray) -> float:
        att, dfn = p[:n], p[n : 2 * n]
        base, home, rho = p[2 * n], p[2 * n + 1], p[2 * n + 2]
        lh = np.exp(base + home + att[hi] + dfn[ai])
        la = np.exp(base + att[ai] + dfn[hi])
        t = np.clip(tau(x, y, lh, la, rho), 1e-10, None)
        ll = np.log(t) + poisson.logpmf(x, lh) + poisson.logpmf(y, la)
        return -(w * ll).sum() + l2 * (att @ att + dfn @ dfn)

    p0 = np.concatenate([np.zeros(2 * n), [0.1, 0.25, -0.05]])
    bounds = [(None, None)] * (2 * n + 2) + [(-0.3, 0.3)]
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
    p = res.x
    return DixonColes(
        attack={t: p[i] for t, i in idx.items()},
        defence={t: p[n + i] for t, i in idx.items()},
        base=p[2 * n],
        home_adv=p[2 * n + 1],
        rho=p[2 * n + 2],
    )


class ModelBook:
    """Caches one fit per (league, month) so every game is priced only with data before it."""

    def __init__(self, results: pd.DataFrame, lookback_days: int = 730):
        self.results = results
        self.lookback = pd.Timedelta(days=lookback_days)
        self._fit = lru_cache(maxsize=None)(self._fit_uncached)

    def _fit_uncached(self, league: str, month_start: pd.Timestamp) -> DixonColes:
        r = self.results
        r = r[(r.league == league) & (r.date >= month_start - self.lookback)]
        return fit(r, month_start)

    def model_for(self, league: str, date: pd.Timestamp) -> DixonColes:
        return self._fit(league, pd.Timestamp(date.year, date.month, 1))
