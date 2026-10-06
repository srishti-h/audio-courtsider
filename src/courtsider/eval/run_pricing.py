"""How good is the market maker's pricing model?

python -m courtsider.eval.run_pricing

1. Pre-match: Dixon–Coles (fit only on earlier matches) vs Pinnacle closing odds (margin removed)
   on every 2014-17 match in the five leagues: log loss and Brier score of the 1X2 outcome.
2. In-play: on SoccerNet test matches, Brier score of P(home/draw/away) at fixed minutes given the
   labelled score and red cards, vs the pre-match model alone.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from courtsider.paths import PROCESSED, RESULTS
from courtsider.pricing.dixon_coles import ModelBook
from courtsider.pricing.inplay import GoalClock, InPlayPricer, MatchState


def _outcome(h: int, a: int) -> int:
    return 0 if h > a else (1 if h == a else 2)


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    onehot = np.eye(3)[y]
    return dict(
        log_loss=float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)))),
        brier=float(np.mean(((p - onehot) ** 2).sum(1))),
    )


def main() -> None:
    results = pd.read_parquet(PROCESSED / "results.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet")
    events = pd.read_parquet(PROCESSED / "events.parquet")
    book = ModelBook(results)

    ev = results[results.fd_season.isin(["1415", "1516", "1617"]) & results.mkt_h.notna()].copy()
    probs = np.array([book.model_for(r.league, r.date).outcome_probs(r.home, r.away) for r in ev.itertuples()])
    y = np.array([_outcome(h, a) for h, a in zip(ev.home_goals, ev.away_goals, strict=True)])
    mkt = ev[["mkt_h", "mkt_d", "mkt_a"]].to_numpy()
    base = np.tile(np.bincount(y, minlength=3) / len(y), (len(y), 1))
    pre = dict(
        n_matches=len(ev), dixon_coles=scores(probs, y), pinnacle_closing=scores(mkt, y), base_rates=scores(base, y)
    )

    train_goals = events[(events.event == "goal") & events.game_id.isin(games[games.split == "train"].game_id)]
    clock = GoalClock(train_goals.minute.to_numpy())
    test = games[(games.split == "test") & games.fd_home.notna()]
    minutes = (0, 15, 30, 45, 60, 75, 85)
    inplay_p = {m: [] for m in minutes}
    prematch_p, ys = [], []
    for g in test.itertuples():
        model = book.model_for(g.league, g.date)
        pricer = InPlayPricer(*model.rates(g.fd_home, g.fd_away), clock)
        ge = events[(events.game_id == g.game_id) & events.event.isin(["goal", "red_card"]) & events.team.notna()]
        ys.append(_outcome(g.home_goals, g.away_goals))
        prematch_p.append(model.outcome_probs(g.fd_home, g.fd_away))
        for m in minutes:
            past = ge[ge.minute <= m]
            s = MatchState(
                minute=m,
                home_goals=int(((past.event == "goal") & (past.team == "home")).sum()),
                away_goals=int(((past.event == "goal") & (past.team == "away")).sum()),
                home_reds=int(((past.event == "red_card") & (past.team == "home")).sum()),
                away_reds=int(((past.event == "red_card") & (past.team == "away")).sum()),
            )
            p = pricer.prices(s)
            inplay_p[m].append([p["HOME"] / 100, p["DRAW"] / 100, p["AWAY"] / 100])
    ys = np.array(ys)
    inplay = {f"minute_{m}": scores(np.array(v), ys)["brier"] for m, v in inplay_p.items()}
    report = dict(
        prematch=pre,
        inplay_brier=inplay,
        inplay_n_matches=len(ys),
        prematch_brier_on_same_matches=scores(np.array(prematch_p), ys)["brier"],
    )
    (RESULTS / "pricing.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
