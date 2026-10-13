"""Load SoccerNet action-spotting labels (Labels-v2.json) into tidy tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from courtsider.data.taxonomy import SOCCERNET_TO_EVENT, global_time
from courtsider.paths import SOCCERNET_DIR


def download_labels(local_dir: Path = SOCCERNET_DIR) -> None:
    from SoccerNet.Downloader import SoccerNetDownloader

    SoccerNetDownloader(LocalDirectory=str(local_dir)).downloadGames(
        files=["Labels-v2.json"], split=["train", "valid", "test"]
    )


def game_splits() -> dict[str, str]:
    from SoccerNet.utils import getListGames

    return {g: s for s in ("train", "valid", "test") for g in getListGames(s)}


def _half_starts(ann: list[dict]) -> dict[int, float]:
    """Video time of each half's kick-off; videos sometimes start before the whistle."""
    starts = {1: 0.0, 2: 0.0}
    for a in ann:
        half = int(a["gameTime"].split(" - ")[0])
        t = int(a["position"]) / 1000.0
        if a["label"] == "Kick-off" and t < 300 and starts[half] == 0.0:
            starts[half] = t
    return starts


def load(local_dir: Path = SOCCERNET_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (games, events). Event times are seconds of half-video plus match minute."""
    splits = game_splits()
    games, events = [], []
    for path in sorted(local_dir.glob("**/Labels-v2.json")):
        d = json.loads(path.read_text())
        game_id = d["UrlLocal"].rstrip("/")
        league, season = game_id.split("/")[:2]
        hs, as_ = (int(x) for x in d["gameScore"].split(" - "))
        games.append(
            dict(
                game_id=game_id,
                league=league,
                season=season,
                date=pd.to_datetime(d["gameDate"].split(" - ")[0], dayfirst=True),
                home=d["gameHomeTeam"],
                away=d["gameAwayTeam"],
                home_goals=hs,
                away_goals=as_,
                split=splits.get(game_id, "unknown"),
            )
        )
        starts = _half_starts(d["annotations"])
        for a in d["annotations"]:
            half = int(a["gameTime"].split(" - ")[0])
            t = int(a["position"]) / 1000.0
            team = a["team"] if a["team"] in ("home", "away") else None
            events.append(
                dict(
                    game_id=game_id,
                    half=half,
                    t=t,
                    g=global_time(half, t),
                    minute=max(0.0, (t - starts[half]) / 60.0) + 45.0 * (half - 1),
                    label=a["label"],
                    event=SOCCERNET_TO_EVENT.get(a["label"]),
                    team=team,
                    visible=a["visibility"] == "visible",
                )
            )
    games_df = pd.DataFrame(games)
    events_df = pd.DataFrame(events).sort_values(["game_id", "g"]).reset_index(drop=True)
    return games_df, events_df
