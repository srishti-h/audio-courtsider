"""Recent (2025/26 onwards) match data from ESPN's public soccer API: python -m courtsider.data.espn

For every finished match in the five leagues it caches the match summary, which carries ESPN's
written play-by-play commentary ("Goal! Sevilla 1, Mallorca 0. Rubén Vargas ...") and key events with
wall-clock times. Two uses:
  * `commentary()`: written commentary lines labelled with our event types, extra text-model training data;
  * `key_events()`: goals/cards/penalties with wall-clock times, used to label recorded radio (data.radio).
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from courtsider.live.feeds import HEADERS
from courtsider.paths import DATA, PROCESSED

RAW = DATA / "raw" / "espn"
API = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/{what}"
LEAGUES = {
    "eng.1": "england_epl",
    "esp.1": "spain_laliga",
    "ita.1": "italy_serie-a",
    "ger.1": "germany_bundesliga",
    "fra.1": "france_ligue-1",
}
START = date(2025, 8, 1)  # first matchday of 2025/26 in the five leagues is mid-August


def _get(league: str, what: str, **params) -> dict:
    r = requests.get(API.format(league=league, what=what), params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def scoreboard(league: str, day: date) -> list[dict]:
    """Finished matches on `day`; cached once the day is over."""
    path = RAW / "scoreboard" / league / f"{day:%Y%m%d}.json"
    if path.exists():
        return json.loads(path.read_text())
    events = _get(league, "scoreboard", dates=f"{day:%Y%m%d}").get("events", [])
    out = [
        dict(id=e["id"], league=league, date=e["date"], name=e["name"])
        for e in events
        if e["status"]["type"].get("completed")
    ]
    if day < date.today() - timedelta(days=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
    return out


def summary(league: str, match_id: str) -> dict:
    path = RAW / "summary" / f"{match_id}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    d = _get(league, "summary", event=match_id)
    keep = {k: d.get(k) for k in ("header", "commentary", "keyEvents")}
    if (d.get("header") or {}).get("competitions", [{}])[0].get("status", {}).get("type", {}).get("completed"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(keep).encode()))
    return keep


def teams(s: dict) -> tuple[str, str]:
    comp = s["header"]["competitions"][0]["competitors"]
    by = {c["homeAway"]: c["team"]["displayName"] for c in comp}
    return by["home"], by["away"]


def events_of(play_type: str, text: str) -> set[str]:
    """Our event types for one ESPN play (type text like 'Goal - Header', 'Penalty - Scored', 'Shot Blocked')."""
    t, x = play_type.lower(), text.lower()
    ev = set()
    if t.startswith("goal") or t == "own goal" or t == "penalty - scored":
        ev.add("goal")
    if t.startswith("penalty") or x.startswith("penalty conceded"):
        ev.add("penalty")
    if t.startswith("shot") or t.startswith("goal") or t == "penalty - scored":
        ev.add("shot")
    if t == "corner awarded":
        ev.add("corner")
    if t == "yellow card":
        ev.add("yellow_card")
    if t == "red card" or "second yellow" in x:
        ev.add("red_card")
    if t == "offside":
        ev.add("offside")
    if t == "substitution":
        ev.add("substitution")
    return ev


def _side(play: dict, home: str, away: str) -> str:
    name = (play.get("team") or {}).get("displayName")
    if play.get("type", {}).get("text") == "Own Goal":  # ESPN's team field is ambiguous for own goals
        return "unknown"
    return "home" if name == home else "away" if name == away else "unknown"


def key_events(s: dict) -> list[dict]:
    """Price-moving events with epoch wall-clock times."""
    home, away = teams(s)
    out = []
    for k in s.get("keyEvents") or []:
        ev = events_of(k["type"]["text"], k.get("text", "")) & {"goal", "red_card", "penalty"}
        if not ev or not k.get("wallclock"):
            continue
        wall = datetime.fromisoformat(k["wallclock"].replace("Z", "+00:00")).timestamp()
        for e in ev:
            out.append(
                dict(
                    event=e,
                    wall=wall,
                    clock=float((k.get("clock") or {}).get("value") or 0),  # match-clock seconds
                    period=int((k.get("period") or {}).get("number") or 0),
                    team=_side(k, home, away),
                    home=home,
                    away=away,
                    text=k.get("text", ""),
                )
            )
    return out


def commentary() -> pd.DataFrame:
    """One row per written commentary line of every cached match, labelled with our event types."""
    from courtsider.data.taxonomy import EVENT_TYPES

    rows = []
    for path in sorted((RAW / "summary").glob("*.json.gz")):
        s = json.loads(gzip.decompress(path.read_bytes()))
        if not s.get("commentary"):
            continue
        home, away = teams(s)
        day = s["header"]["competitions"][0]["date"][:10]
        for c in s["commentary"]:
            play = c.get("play") or {}
            ev = events_of(play.get("type", {}).get("text", ""), c.get("text", ""))
            team = _side(play, home, away) if ev & {"goal", "red_card", "penalty"} else "unknown"
            rows.append(
                dict(
                    game_id=f"espn-{path.name.split('.')[0]}",
                    date=day,
                    home=home,
                    away=away,
                    seq=c.get("sequence", 0),
                    text=c.get("text", ""),
                    team=team,
                    **{e: int(e in ev) for e in EVENT_TYPES},
                )
            )
    return pd.DataFrame(rows).sort_values(["game_id", "seq"]).reset_index(drop=True)


def download(start: date = START, end: date | None = None, workers: int = 8) -> None:
    end = end or date.today()
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    with ThreadPoolExecutor(workers) as pool:
        boards = list(pool.map(lambda a: _safe(scoreboard, *a), [(lg, d) for lg in LEAGUES for d in days]))
        matches = [m for b in boards if b for m in b]
        print(f"{len(matches)} finished matches since {start}", flush=True)
        list(pool.map(lambda m: _safe(summary, m["league"], m["id"]), matches))


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:  # one bad day/match must not stop a long backfill; rerun to retry
        print(f"skip {fn.__name__}{args}: {e}", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=str(START))
    args = ap.parse_args()
    download(date.fromisoformat(args.start))
    df = commentary()
    df.to_parquet(PROCESSED / "espn_commentary.parquet")
    print(
        f"{df.game_id.nunique()} matches, {len(df)} lines; positives:",
        df[["goal", "red_card", "penalty"]].sum().to_dict(),
    )


if __name__ == "__main__":
    main()
