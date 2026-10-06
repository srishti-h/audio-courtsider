import numpy as np
import pandas as pd

from courtsider.pricing.dixon_coles import fit
from courtsider.pricing.inplay import GoalClock, InPlayPricer, MatchState, settle
from courtsider.sim.match import MatchSim, Signal, SimConfig


def _pricer():
    return InPlayPricer(1.6, 1.1, GoalClock())


def test_prices_are_consistent_probabilities():
    p = _pricer().prices(MatchState(minute=30, home_goals=1))
    assert abs(p["HOME"] + p["DRAW"] + p["AWAY"] - 100) < 1e-6
    assert all(0 <= v <= 100 for v in p.values())


def test_goal_moves_prices_the_right_way_and_time_decay():
    pr = _pricer()
    s = MatchState(minute=60)
    base, after = pr.prices(s), pr.prices_if(s, "goal", "home")
    assert after["HOME"] > base["HOME"] and after["AWAY"] < base["AWAY"]
    late = pr.prices(MatchState(minute=88, home_goals=1))
    early = pr.prices(MatchState(minute=10, home_goals=1))
    assert late["HOME"] > early["HOME"]
    assert pr.prices(MatchState(minute=95, home_goals=1, away_goals=1))["BTTS"] == 100


def test_penalty_is_between_scored_and_missed():
    pr = _pricer()
    s = MatchState(minute=70)
    pen = pr.prices_if(s, "penalty", "away")["AWAY"]
    assert pr.prices(s)["AWAY"] < pen < pr.prices_if(s, "goal", "away")["AWAY"]


def test_settlement():
    assert settle("HOME", 2, 1) == 100 and settle("DRAW", 2, 1) == 0
    assert settle("OVER25", 2, 1) == 100 and settle("BTTS", 1, 0) == 0


def test_dixon_coles_recovers_strength_ordering():
    rng = np.random.default_rng(0)
    teams = ["strong", "mid", "weak"]
    strength = {"strong": 0.5, "mid": 0.0, "weak": -0.5}
    rows = []
    for d in range(400):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append(
            dict(
                date=pd.Timestamp("2015-01-01") + pd.Timedelta(days=d % 300),
                home=h,
                away=a,
                home_goals=rng.poisson(np.exp(0.3 + strength[h] - strength[a])),
                away_goals=rng.poisson(np.exp(strength[a] - strength[h])),
            )
        )
    m = fit(pd.DataFrame(rows), pd.Timestamp("2016-01-01"))
    assert m.attack["strong"] > m.attack["mid"] > m.attack["weak"]
    assert m.home_adv > 0


def _events():
    return pd.DataFrame(
        dict(
            game_id="g",
            half=[1, 1, 1],
            t=[0.0, 600.0, 2700.0],
            g=[0.0, 600.0, 2700.0],
            minute=[0.0, 10.0, 45.0],
            label=["Kick-off", "Goal", "Corner"],
            event=[None, "goal", "corner"],
            team=["home", "home", "away"],
            visible=True,
        )
    )


def test_courtsider_profits_from_an_early_correct_signal_and_loses_on_a_false_one():
    good = [Signal(600.5, {"goal": 0.95, "red_card": 0, "penalty": 0}, {"home": 0.95, "away": 0.05})]
    cfg = SimConfig(feed_delay=3.0, feed_jitter=0.0, noise_rate=1 / 30)
    out = MatchSim(_pricer(), _events(), good, cfg, final_score=(1, 0)).run()
    assert out["cs_entries"] == 1 and out["pnl"]["cs"] > 0
    assert out["mm_markout"]["cs"] < 0  # the market maker was adversely selected

    wrong = [Signal(1500.0, {"goal": 0.95, "red_card": 0, "penalty": 0}, {"home": 0.95, "away": 0.05})]
    out = MatchSim(_pricer(), _events(), wrong, cfg, final_score=(1, 0)).run()
    assert out["cs_entries"] == 1 and out["pnl"]["cs"] < 0


def test_audio_aware_market_maker_pulls_quotes():
    good = [Signal(600.5, {"goal": 0.95, "red_card": 0, "penalty": 0}, {"home": 0.95, "away": 0.05})]
    cfg = SimConfig(feed_delay=3.0, feed_jitter=0.0, mm_audio_guard=True, mm_guard_threshold=0.5)
    sim = MatchSim(_pricer(), _events(), good, cfg, final_score=(1, 0))
    out = sim.run()
    assert out["cs_volume"] == 0  # MM's own detector fired at the same time, so there was nothing to hit
