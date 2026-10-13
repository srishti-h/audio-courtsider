"""Live "official feed" sources, polled to timestamp when a goal becomes public.

* ESPN (default, no key): the public JSON behind espn.com's scoreboards. Undocumented, so treat it
  as best-effort and poll politely (5 s). Lineups double as a real team sheet for name biasing.
* football-data.org (optional): set FOOTBALL_DATA_TOKEN to use its free tier instead.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests

ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/all"
FOOTBALL_DATA = "https://api.football-data.org/v4"
HEADERS = {"User-Agent": "audio-courtsider (research; polls every 5 s)"}


@dataclass
class Snapshot:
    match_id: str
    home: str
    away: str
    score: tuple[int, int]
    clock: str
    state: str  # pre / in / post
    lineups: dict[str, list[str]] = field(default_factory=dict)
    red_cards: tuple[int, int] = (0, 0)


class EspnFeed:
    name = "espn"

    def matches(self) -> list[dict]:
        d = requests.get(f"{ESPN}/scoreboard", headers=HEADERS, timeout=10).json()
        out = []
        for e in d.get("events", []):
            c = e["competitions"][0]
            teams = {x["homeAway"]: x for x in c["competitors"]}
            out.append(
                dict(
                    id=e["id"],
                    name=e["name"],
                    state=e["status"]["type"]["state"],
                    clock=e["status"].get("displayClock", ""),
                    date=e["date"],
                    home=teams["home"]["team"]["displayName"],
                    away=teams["away"]["team"]["displayName"],
                    score=[int(teams["home"].get("score") or 0), int(teams["away"].get("score") or 0)],
                )
            )
        order = {"in": 0, "pre": 1, "post": 2}
        return sorted(out, key=lambda m: (order.get(m["state"], 3), m["date"]))

    def snapshot(self, match_id: str) -> Snapshot:
        d = requests.get(f"{ESPN}/summary", params={"event": match_id}, headers=HEADERS, timeout=10).json()
        comp = d["header"]["competitions"][0]
        teams = {x["homeAway"]: x for x in comp["competitors"]}
        ids = {x["team"]["id"]: x["homeAway"] for x in comp["competitors"]}
        lineups = {}
        for r in d.get("rosters", []):
            side = r.get("homeAway") or ids.get(r.get("team", {}).get("id"))
            lineups[side] = [p["athlete"]["displayName"] for p in r.get("roster", []) if p.get("athlete")]
        reds = {"home": 0, "away": 0}
        for k in d.get("keyEvents", []):
            if "Red Card" in (k.get("type", {}).get("text") or ""):
                side = ids.get((k.get("team") or {}).get("id"))
                if side:
                    reds[side] += 1
        status = comp.get("status", {})
        return Snapshot(
            match_id=match_id,
            home=teams["home"]["team"]["displayName"],
            away=teams["away"]["team"]["displayName"],
            score=(int(teams["home"].get("score") or 0), int(teams["away"].get("score") or 0)),
            clock=status.get("displayClock", ""),
            state=status.get("type", {}).get("state", ""),
            lineups=lineups,
            red_cards=(reds["home"], reds["away"]),
        )


class FootballDataFeed:
    name = "football-data"

    def __init__(self, token: str | None = None):
        self.headers = {"X-Auth-Token": token or os.environ["FOOTBALL_DATA_TOKEN"]}

    def matches(self) -> list[dict]:
        d = requests.get(f"{FOOTBALL_DATA}/matches", headers=self.headers, timeout=10).json()
        state = {"IN_PLAY": "in", "PAUSED": "in", "TIMED": "pre", "SCHEDULED": "pre", "FINISHED": "post"}
        return [
            dict(
                id=str(m["id"]),
                name=f"{m['awayTeam']['name']} at {m['homeTeam']['name']}",
                state=state.get(m["status"], "post"),
                clock=str(m.get("minute") or ""),
                date=m["utcDate"],
                home=m["homeTeam"]["name"],
                away=m["awayTeam"]["name"],
                score=[m["score"]["fullTime"]["home"] or 0, m["score"]["fullTime"]["away"] or 0],
            )
            for m in d.get("matches", [])
        ]

    def snapshot(self, match_id: str) -> Snapshot:
        m = requests.get(f"{FOOTBALL_DATA}/matches/{match_id}", headers=self.headers, timeout=10).json()
        return Snapshot(
            match_id,
            m["homeTeam"]["name"],
            m["awayTeam"]["name"],
            (m["score"]["fullTime"]["home"] or 0, m["score"]["fullTime"]["away"] or 0),
            str(m.get("minute") or ""),
            "in" if m["status"] in ("IN_PLAY", "PAUSED") else "post",
        )


def get_feed(name: str | None = None):
    name = name or ("football-data" if os.environ.get("FOOTBALL_DATA_TOKEN") else "espn")
    return FootballDataFeed() if name == "football-data" else EspnFeed()


def diff(prev: Snapshot | None, cur: Snapshot, wall: float | None = None) -> list[dict]:
    """Feed events implied by two consecutive snapshots: goals (score went up) and red cards."""
    if prev is None:
        return []
    wall = wall or time.time()
    out = []
    for i, side in enumerate(("home", "away")):
        for _ in range(cur.score[i] - prev.score[i]):
            out.append(dict(kind="feed", event="goal", team=side, score=list(cur.score), clock=cur.clock, wall=wall))
        for _ in range(cur.red_cards[i] - prev.red_cards[i]):
            out.append(
                dict(kind="feed", event="red_card", team=side, score=list(cur.score), clock=cur.clock, wall=wall)
            )
    return out


def roster_vocabulary(lineups: dict[str, list[str]], limit: int = 60) -> list[str]:
    """Surnames first (what commentators say), then full names; deduplicated."""
    seen, out = set(), []
    for names in lineups.values():
        for n in names:
            for v in (n.split()[-1], n):
                if v not in seen:
                    seen.add(v)
                    out.append(v)
    return out[:limit]
