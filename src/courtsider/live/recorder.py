"""Unattended live recorder: python -m courtsider.live.recorder [--station ser] [--once]

Every minute it checks ESPN for live matches in the watched competitions. While any are on, it
streams Spanish radio commentary (which covers every La Liga match, switching between grounds)
through the live pipeline with Whisper translating Spanish -> English, polls the feed for every
live match, and logs commentary text, detections and feed goals to data/live/*.jsonl. No audio is
stored. `python -m courtsider.live.report` turns the logs into lead times.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from datetime import datetime

import numpy as np
import requests

from courtsider.live.feeds import HEADERS, EspnFeed, diff, roster_vocabulary
from courtsider.live.pipeline import LivePipeline
from courtsider.live.report import LIVE_DIR

STATIONS = {
    "ser": "http://playerservices.streamtheworld.com/api/livestream-redirect/CADENASER.mp3",
    "ondacero": "https://atres-live.ondacero.es/live/ondacero/bitrate_1.m3u8",
}
# ESPN league slugs: La Liga, Copa del Rey, Champions League, Europa League
LEAGUES = ("esp.1", "esp.copa_del_rey", "uefa.champions", "uefa.europa")
SPANISH_ONLY_IN = ("uefa.champions", "uefa.europa", "esp.copa_del_rey")  # Spanish radio follows Spanish clubs there
CHECK_S, POLL_S, TICK_S = 60.0, 5.0, 0.5


def live_matches() -> list[dict]:
    """In-progress matches in the watched competitions that Spanish radio will be covering."""
    out = []
    for league in LEAGUES:
        try:
            d = requests.get(
                f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard", headers=HEADERS, timeout=10
            ).json()
        except Exception:
            continue
        for e in d.get("events", []):
            if e["status"]["type"]["state"] != "in":
                continue
            teams = e["competitions"][0]["competitors"]
            spanish = any(
                (t.get("team", {}).get("location") or "") in SPANISH_CLUBS or t["team"]["displayName"] in SPANISH_CLUBS
                for t in teams
            )
            if league in SPANISH_ONLY_IN and not spanish:
                continue
            out.append(dict(id=e["id"], league=league, name=e["name"]))
    return out


SPANISH_CLUBS = {
    "Real Madrid",
    "Barcelona",
    "Atletico Madrid",
    "Atlético Madrid",
    "Athletic Club",
    "Real Sociedad",
    "Villarreal",
    "Real Betis",
    "Sevilla",
    "Valencia",
    "Girona",
    "Celta Vigo",
    "Osasuna",
    "Getafe",
    "Rayo Vallecano",
    "Mallorca",
    "Alavés",
    "Espanyol",
    "Levante",
    "Elche",
    "Real Oviedo",
}


class Session:
    def __init__(self, station: str, matches: list[dict]):
        self.feed = EspnFeed()
        self.station = station
        self.snaps = {m["id"]: self.feed.snapshot(m["id"]) for m in matches}
        vocab = roster_vocabulary({f"{k}-{s}": v for k, sn in self.snaps.items() for s, v in sn.lineups.items()}, 120)
        first = next(iter(self.snaps.values()))
        self.pipe = LivePipeline(
            first.home, first.away, vocab, model="small", fast=True, language="es", task="translate"
        )
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        self.log = (LIVE_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_radio-{station}.jsonl").open("a")
        self.lock = threading.Lock()
        self.emit(
            [
                dict(
                    kind="session",
                    wall=time.time(),
                    started=datetime.now().isoformat(timespec="seconds"),
                    source="radio",
                    station=station,
                    feed="espn",
                    home=first.home,
                    away=first.away,
                    matches=[
                        dict(id=k, home=s.home, away=s.away, score=list(s.score), clock=s.clock)
                        for k, s in self.snaps.items()
                    ],
                    vocabulary=vocab[:120],
                )
            ]
        )

    def emit(self, events: list[dict]) -> None:
        with self.lock:
            for e in events:
                self.log.write(json.dumps(e, ensure_ascii=False) + "\n")
            self.log.flush()

    def watch_feed(self, stop: threading.Event) -> None:
        while not stop.wait(POLL_S):
            for mid, prev in list(self.snaps.items()):
                try:
                    cur = self.feed.snapshot(mid)
                except Exception:
                    continue
                events = diff(prev, cur)
                for e in events:
                    e.update(match_id=mid, home=cur.home, away=cur.away)
                self.emit(events)
                self.snaps[mid] = cur

    def run(self, stop: threading.Event) -> None:
        ff = subprocess.Popen(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-i",
                STATIONS[self.station],
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "f32le",
                "-",
            ],
            stdout=subprocess.PIPE,
        )
        threading.Thread(target=self.watch_feed, args=(stop,), daemon=True).start()
        buf: list[np.ndarray] = []
        cond = threading.Condition()

        def reader():  # keep pulling audio while a tick runs, so we never fall behind real time
            while not stop.is_set():
                raw = ff.stdout.read(int(16000 * TICK_S) * 4)
                if not raw:
                    stop.set()
                with cond:
                    buf.append(np.frombuffer(raw, dtype=np.float32))
                    cond.notify()

        threading.Thread(target=reader, daemon=True).start()
        try:
            while not stop.is_set():
                with cond:
                    cond.wait_for(lambda: buf or stop.is_set(), timeout=5)
                    chunks, buf[:] = list(buf), []
                if chunks:
                    self.pipe.feed(np.concatenate(chunks))  # everything that arrived during the last tick
                    self.emit([e for e in self.pipe.tick() if e["kind"] in ("segment", "detection", "status")])
        finally:
            ff.terminate()
            self.log.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="ser", choices=list(STATIONS))
    ap.add_argument("--once", action="store_true", help="record the current live window, then exit")
    args = ap.parse_args()
    print(f"watching {', '.join(LEAGUES)} on {args.station}", flush=True)
    while True:
        matches = live_matches()
        if matches:
            print(f"{datetime.now():%H:%M} live: {[m['name'] for m in matches]}", flush=True)
            stop = threading.Event()
            session = Session(args.station, matches)

            def until_quiet(stop=stop):
                quiet_since = None
                while not stop.wait(CHECK_S):
                    if live_matches():
                        quiet_since = None
                    elif quiet_since is None:
                        quiet_since = time.time()
                    elif time.time() - quiet_since > 300:  # five minutes after the last final whistle
                        stop.set()

            threading.Thread(target=until_quiet, daemon=True).start()
            session.run(stop)
            print(f"{datetime.now():%H:%M} session ended", flush=True)
            if args.once:
                return
        elif args.once:
            print("nothing live right now")
            return
        time.sleep(CHECK_S)


if __name__ == "__main__":
    main()
