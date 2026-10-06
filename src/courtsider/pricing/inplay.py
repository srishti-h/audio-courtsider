"""In-play pricing: remaining-goal Poisson model conditioned on the live match state.

Remaining expected goals = full-match rate x fraction of goals still to come, where the
fraction comes from the empirical goal-time distribution in SoccerNet. Red cards scale rates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import poisson

CONTRACTS = ("HOME", "DRAW", "AWAY", "OVER25", "BTTS")
PAYOUT = 100
PENALTY_CONVERSION = 0.76
RED_SELF = 0.72  # rate multiplier for the side down a player
RED_OPP = 1.22  # and for its opponent
_G = np.arange(11)


@dataclass(frozen=True)
class MatchState:
    minute: float = 0.0
    home_goals: int = 0
    away_goals: int = 0
    home_reds: int = 0
    away_reds: int = 0
    pending_penalty: str | None = None  # "home"/"away" when a penalty is awarded but not yet taken

    def with_goal(self, team: str) -> MatchState:
        if team == "home":
            return replace(self, home_goals=self.home_goals + 1, pending_penalty=None)
        return replace(self, away_goals=self.away_goals + 1, pending_penalty=None)

    def with_red(self, team: str) -> MatchState:
        if team == "home":
            return replace(self, home_reds=self.home_reds + 1)
        return replace(self, away_reds=self.away_reds + 1)


class GoalClock:
    """Fraction of a match's goals expected after minute m (empirical survival function)."""

    def __init__(self, goal_minutes: np.ndarray | None = None):
        grid = np.arange(0, 96)
        if goal_minutes is None or len(goal_minutes) == 0:
            surv = 1 - grid / 95.0
        else:
            gm = np.clip(np.asarray(goal_minutes, float), 0, 95)
            surv = np.array([(gm > m).mean() for m in grid])
        self.grid, self.surv = grid, surv

    def remaining(self, minute: float) -> float:
        return float(np.interp(minute, self.grid, self.surv))


class InPlayPricer:
    def __init__(self, lam_home: float, lam_away: float, clock: GoalClock):
        self.lh, self.la, self.clock = lam_home, lam_away, clock

    def _rates(self, s: MatchState) -> tuple[float, float]:
        f = self.clock.remaining(s.minute)
        lh = self.lh * f * RED_SELF**s.home_reds * RED_OPP**s.away_reds
        la = self.la * f * RED_SELF**s.away_reds * RED_OPP**s.home_reds
        return lh, la

    def _final_score_matrix(self, s: MatchState) -> np.ndarray:
        lh, la = self._rates(s)
        m = np.outer(poisson.pmf(_G, lh), poisson.pmf(_G, la))
        return m / m.sum()

    def _prices_no_penalty(self, s: MatchState) -> dict[str, float]:
        m = self._final_score_matrix(s)
        fh = s.home_goals + _G[:, None]
        fa = s.away_goals + _G[None, :]
        p = {
            "HOME": m[fh > fa].sum(),
            "DRAW": m[fh == fa].sum(),
            "AWAY": m[fh < fa].sum(),
            "OVER25": m[(fh + fa) > 2.5].sum(),
            "BTTS": m[(fh > 0) & (fa > 0)].sum(),
        }
        return {k: PAYOUT * float(v) for k, v in p.items()}

    def prices(self, s: MatchState) -> dict[str, float]:
        if s.pending_penalty is None:
            return self._prices_no_penalty(s)
        scored = self._prices_no_penalty(s.with_goal(s.pending_penalty))
        missed = self._prices_no_penalty(replace(s, pending_penalty=None))
        return {k: PENALTY_CONVERSION * scored[k] + (1 - PENALTY_CONVERSION) * missed[k] for k in scored}

    def prices_if(self, s: MatchState, event: str, team: str | None) -> dict[str, float]:
        """Fair prices *if* `event` just happened: what an informed trader expects after the feed."""
        if team not in ("home", "away"):
            return self.prices(s)
        if event == "goal":
            return self.prices(s.with_goal(team))
        if event == "red_card":
            return self.prices(s.with_red(team))
        if event == "penalty":
            return self.prices(replace(s, pending_penalty=team))
        return self.prices(s)


def settle(contract: str, home_goals: int, away_goals: int) -> float:
    outcome = {
        "HOME": home_goals > away_goals,
        "DRAW": home_goals == away_goals,
        "AWAY": home_goals < away_goals,
        "OVER25": home_goals + away_goals > 2,
        "BTTS": home_goals > 0 and away_goals > 0,
    }[contract]
    return float(PAYOUT if outcome else 0)
