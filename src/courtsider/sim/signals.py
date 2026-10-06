"""Build per-game commentary signals and pricers for the market simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from courtsider.pricing.dixon_coles import ModelBook
from courtsider.pricing.inplay import GoalClock, InPlayPricer
from courtsider.sim.match import PRICE_EVENTS, Signal


def signals_for_game(beliefs: pd.DataFrame, latency: float) -> list[Signal]:
    """`beliefs`: one row per segment with calibrated p_<event> and team_home/team_away columns."""
    out = []
    has_team = "team_home" in beliefs
    for r in beliefs.itertuples():
        probs = {e: float(getattr(r, f"p_{e}")) for e in PRICE_EVENTS}
        team = {"home": float(r.team_home), "away": float(r.team_away)} if has_team else {"home": 0.5, "away": 0.5}
        out.append(Signal(float(r.g_end) + latency, probs, team, r.text))
    return out


class PricerFactory:
    def __init__(self, games: pd.DataFrame, results: pd.DataFrame, events: pd.DataFrame):
        self.games = games.set_index("game_id")
        self.book = ModelBook(results)
        train_goals = events[(events.event == "goal") & events.game_id.isin(games[games.split == "train"].game_id)]
        self.clock = GoalClock(train_goals.minute.to_numpy())

    def __call__(self, game_id: str) -> InPlayPricer | None:
        g = self.games.loc[game_id]
        if pd.isna(g.fd_home):
            return None
        model = self.book.model_for(g.league, g.date)
        return InPlayPricer(*model.rates(g.fd_home, g.fd_away), self.clock)


def final_score(games: pd.DataFrame, game_id: str) -> tuple[int, int]:
    g = games.set_index("game_id").loc[game_id]
    return int(g.home_goals), int(g.away_goals)


def null_signals(n: int = 0) -> list[Signal]:
    return [Signal(float(t), {e: 0.0 for e in PRICE_EVENTS}, {"home": 0.5, "away": 0.5}) for t in np.zeros(n)]
