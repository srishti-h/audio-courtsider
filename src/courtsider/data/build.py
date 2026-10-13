"""Build processed tables: `python -m courtsider.data.build [--download]`."""

from __future__ import annotations

import argparse
import subprocess

from courtsider.data import echoes, footballdata, soccernet
from courtsider.paths import ECHOES_DIR, PROCESSED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="fetch raw data first")
    args = ap.parse_args()

    if args.download:
        soccernet.download_labels()
        if not ECHOES_DIR.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", echoes.ECHOES_REPO, str(ECHOES_DIR.parent)], check=True
            )
        footballdata.download()

    games, events = soccernet.load()
    segments = echoes.load(games.game_id.tolist())
    results = footballdata.load()
    matched = footballdata.match_games(games, results)
    games = games.merge(matched, on="game_id", how="left")
    games["has_transcript"] = games.game_id.isin(set(segments.game_id))

    games.to_parquet(PROCESSED / "games.parquet")
    events.to_parquet(PROCESSED / "events.parquet")
    segments.to_parquet(PROCESSED / "segments.parquet")
    results.to_parquet(PROCESSED / "results.parquet")
    print(
        f"games={len(games)} with_transcript={games.has_transcript.sum()} "
        f"odds_matched={games.mkt_h.notna().sum()} events={len(events)} segments={len(segments)}"
    )


if __name__ == "__main__":
    main()
