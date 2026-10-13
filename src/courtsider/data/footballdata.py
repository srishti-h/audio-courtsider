"""Historical results + closing odds from football-data.co.uk, matched to SoccerNet games."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz

from courtsider.paths import FOOTBALLDATA_DIR

LEAGUE_CODES = {
    "england_epl": "E0",
    "spain_laliga": "SP1",
    "italy_serie-a": "I1",
    "germany_bundesliga": "D1",
    "france_ligue-1": "F1",
}
SEASONS = ("1213", "1314", "1415", "1516", "1617")
URL = "https://football-data.co.uk/mmz4281/{season}/{code}.csv"


def download(out_dir: Path = FOOTBALLDATA_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        for code in LEAGUE_CODES.values():
            path = out_dir / f"{code}_{season}.csv"
            if path.exists() and path.stat().st_size > 0:
                continue
            r = requests.get(URL.format(season=season, code=code), timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            path.write_bytes(r.content)


def _implied(odds: pd.DataFrame) -> np.ndarray:
    inv = 1.0 / odds.to_numpy(dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)  # remove the bookmaker margin proportionally


def load(root: Path = FOOTBALLDATA_DIR) -> pd.DataFrame:
    code_to_league = {v: k for k, v in LEAGUE_CODES.items()}
    frames = []
    for path in sorted(root.glob("*.csv")):
        code, season = path.stem.split("_")
        df = pd.read_csv(path, encoding="latin1").dropna(subset=["HomeTeam", "FTHG"])
        df["league"] = code_to_league[code]
        df["fd_season"] = season
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    out = df[["league", "fd_season", "date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].rename(
        columns={"HomeTeam": "home", "AwayTeam": "away", "FTHG": "home_goals", "FTAG": "away_goals"}
    )
    # Pinnacle closing odds when present, else Bet365.
    for cols, tag in ((["PSCH", "PSCD", "PSCA"], "pin"), (["B365H", "B365D", "B365A"], "b365")):
        if all(c in df for c in cols):
            ok = df[cols].notna().all(axis=1)
            p = np.full((len(df), 3), np.nan)
            p[ok.to_numpy()] = _implied(df.loc[ok, cols])
            out[[f"{tag}_h", f"{tag}_d", f"{tag}_a"]] = p
    out["mkt_h"] = out["pin_h"].fillna(out["b365_h"])
    out["mkt_d"] = out["pin_d"].fillna(out["b365_d"])
    out["mkt_a"] = out["pin_a"].fillna(out["b365_a"])
    return out.drop(columns=[c for c in out if c.startswith(("pin_", "b365_"))])


def match_games(games: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Attach football-data team names + market probabilities to SoccerNet domestic games."""
    rows = []
    for g in games.itertuples():
        if g.league not in LEAGUE_CODES:
            continue
        cand = results[(results.league == g.league) & ((results.date - g.date).abs() <= pd.Timedelta(days=1))]
        if cand.empty:
            continue
        scores = cand.apply(lambda r, g=g: fuzz.WRatio(g.home, r.home) + fuzz.WRatio(g.away, r.away), axis=1)
        best = cand.loc[scores.idxmax()]
        if scores.max() < 140 or (best.home_goals, best.away_goals) != (g.home_goals, g.away_goals):
            continue
        rows.append(
            dict(
                game_id=g.game_id,
                fd_home=best.home,
                fd_away=best.away,
                mkt_h=best.mkt_h,
                mkt_d=best.mkt_d,
                mkt_a=best.mkt_a,
            )
        )
    return pd.DataFrame(rows)
