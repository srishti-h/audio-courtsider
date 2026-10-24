"""Recent radio training data from Cadena SER's podcast archive: python -m courtsider.data.podcasts [--limit N]

SER publishes each Champions League / La Liga radio broadcast as "Los Partidos de la Jornada | A - B y
C - D | 1ª Parte / 2ª Parte" (about 50 min per half, back to Dec 2025). For each half this:
  1. downloads the episode (kept locally, never redistributed),
  2. transcribes it with Whisper (Spanish -> English) and stores log-mel features for the audio model,
  3. finds ESPN's key events for the matches named in the title (match-clock times within the half),
  4. estimates where the episode starts relative to kick-off: the offset that lines up the most ESPN
     goals, cards and substitutions with the commentary announcing them (data.radio's alignment),
  5. labels transcript segments and audio frames exactly like the recorded radio.
Episodes where fewer than 3 events line up are skipped. Alignment is good to
about +-5 s (a goal call may sit anywhere in the labelling window), so podcasts train the text model,
whose segment labels tolerate that, and not the audio model.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import timedelta
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests
from rapidfuzz import fuzz

from courtsider.data import espn, radio
from courtsider.live.recorder import LEAGUES
from courtsider.paths import DATA, PROCESSED, RESULTS
from courtsider.text.dataset import build_inputs, label_segments

FEED = "https://fapi-top.prisasd.com/podcast/playser/carrusel_deportivo/itunestfp/podcast.xml"
ROOT = DATA / "raw" / "podcasts"
OFFSETS = np.arange(-900.0, 900.5, 1.0)  # where kick-off falls in the episode, seconds
HALF_S = 45 * 60
UA = {"User-Agent": "Mozilla/5.0"}


def episodes() -> list[dict]:
    """Half-match broadcasts in the feed, newest first."""
    path = ROOT / "carrusel.xml"
    ROOT.mkdir(parents=True, exist_ok=True)
    path.write_bytes(requests.get(FEED, headers=UA, timeout=60).content)
    out = []
    for it in ET.parse(path).getroot().findall(".//item"):
        title = it.findtext("title") or ""
        m = re.match(r"Los Partidos de la Jornada \| (.+?) \| (\d)ª parte", title, re.I)
        if not m:
            continue
        day = parsedate_to_datetime(it.findtext("pubDate")).date()
        pairs = [p.strip() for p in re.split(r",| y ", m.group(1)) if " - " in p]
        slug = f"{day:%Y%m%d}_{re.sub(r'[^a-z0-9]+', '-', pairs[0].lower()).strip('-')}_{m.group(2)}"
        out.append(
            dict(
                slug=slug,
                title=title,
                day=day,
                half=int(m.group(2)),
                matches=[p.split(" - ") for p in pairs],
                url=it.find("enclosure").attrib["url"],
            )
        )
    return out


def download(ep: dict):
    """curl, not requests: the podcast CDN sometimes trickles a byte at a time, which a socket read timeout
    never catches (a download sat for 20+ minutes); curl aborts below 20 kB/s for 30 s, retries 3 times."""
    path = ROOT / "audio" / f"{ep['slug']}.mp3"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        subprocess.run(
            [
                "curl",
                "-sSLf",
                "--retry",
                "3",
                "--retry-all-errors",
                "--speed-limit",
                "20000",
                "--speed-time",
                "30",
                "--max-time",
                "900",
                "-A",
                UA["User-Agent"],
                "-o",
                str(tmp),
                ep["url"],
            ],
            check=True,
        )
        tmp.rename(path)
    return path


def decode(path) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def transcribe_and_features(ep: dict) -> pd.DataFrame:
    """Transcript segments (cached) and log-mel features (cached) for one episode."""
    from courtsider.acoustic.extract import log_mel

    tpath, mpath = ROOT / "transcripts" / f"{ep['slug']}.json", ROOT / "mel" / f"{ep['slug']}.npy"
    if not (tpath.exists() and mpath.exists()):
        import mlx.core as mx
        import mlx_whisper

        from courtsider.asr.streaming import MODELS as WHISPER

        mx.set_cache_limit(1 << 30)  # see asr.streaming: MLX's buffer cache is otherwise unbounded
        mp3 = download(ep)
        y = decode(mp3)
        res = mlx_whisper.transcribe(
            y, path_or_hf_repo=WHISPER["small"], language="es", task="translate", condition_on_previous_text=False
        )
        tpath.parent.mkdir(parents=True, exist_ok=True)
        tpath.write_text(
            json.dumps([dict(start=s["start"], end=s["end"], text=s["text"].strip()) for s in res["segments"]])
        )
        mpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(mpath, log_mel(y))
        mp3.unlink()  # transcript + features are all that is kept; the broadcast audio is not
    return pd.DataFrame(json.loads(tpath.read_text())).rename(columns={"start": "audio_start", "end": "audio_end"})


# Events commentators announce, used to find where kick-off falls (labels still come from all of them).
# Goals alone are often too few, so cards help. Substitutions don't: the translated commentary rarely
# says "comes on"/"replaces" (0 of 18 matched at the true offset in the pilot) and their chance hits
# outvoted the goals.
ANCHORS = {
    "goal": radio.GOAL_CALL,
    "yellow_card": re.compile(r"yellow card|\bbooked\b|\bcaution", re.I),
    "red_card": re.compile(r"red card|sent off|\bexpelled", re.I),
}
MIN_MATCHED = 3


def espn_events(ep: dict) -> pd.DataFrame:
    """ESPN key events (goals, cards, penalties, substitutions) of the matches named in the title,
    timed on this half's match clock."""
    rows = []
    for league in LEAGUES:
        for day in (ep["day"] - timedelta(1), ep["day"]):
            for m in espn._safe(espn.scoreboard, league, day) or []:
                s = espn._safe(espn.summary, league, m["id"])
                if not s or not s.get("header"):
                    continue
                home, away = espn.teams(s)
                if not any(fuzz.WRatio(a, home) + fuzz.WRatio(b, away) >= 150 for a, b in ep["matches"]):
                    continue
                for k in s.get("keyEvents") or []:
                    if int((k.get("period") or {}).get("number") or 0) != ep["half"]:
                        continue
                    clock = float((k.get("clock") or {}).get("value") or 0)
                    for e in espn.events_of(k["type"]["text"], k.get("text", "")) - {"shot"}:
                        team = espn._side(k, home, away) if e in ("goal", "red_card", "penalty") else "unknown"
                        rows.append(
                            dict(
                                event=e,
                                clock=clock,
                                t=clock - (ep["half"] - 1) * HALF_S,
                                match_id=m["id"],
                                team=team,
                                home=home,
                                away=away,
                            )
                        )
    return pd.DataFrame(rows).drop_duplicates(["match_id", "event", "clock"]) if rows else pd.DataFrame()


def align(segs: pd.DataFrame, ev: pd.DataFrame) -> tuple[float | None, int, int]:
    """Episode time of kick-off -> (offset or None if not confident, anchors matched, anchors)."""
    hits, n = np.zeros(len(OFFSETS), int), 0
    for event, pattern in ANCHORS.items():
        t = ev[ev.event == event].t.to_numpy() if len(ev) else np.zeros(0)
        if len(t):
            hits += radio.delay_hits(segs, lambda d, t=t: t + d, OFFSETS, pattern)
            n += len(t)
    off = radio.best_delay(hits, n, OFFSETS, default=None)
    matched = int(hits[np.abs(OFFSETS - off).argmin()]) if off is not None else int(hits.max(initial=0))
    return (off if matched >= MIN_MATCHED else None), matched, n


def build(limit: int | None = None, only: str | None = None) -> list[dict]:
    eps = [e for e in episodes() if not only or only.lower() in e["title"].lower()][:limit]
    report, seg_frames, ev_frames = [], [], []
    n_test = max(1, round(radio.TEST_FRACTION * len(eps)))
    for i, ep in enumerate(sorted(eps, key=lambda e: (e["day"], e["half"]))):
        try:
            segs = transcribe_and_features(ep)
        except (requests.RequestException, subprocess.CalledProcessError) as e:  # one bad episode must not end the run
            report.append(dict(episode=ep["slug"], error=str(e)[:200], kickoff_at_s=None))
            print(report[-1], flush=True)
            continue
        ev = espn_events(ep)
        off, matched, n = align(segs, ev) if len(ev) else (None, 0, 0)
        report.append(dict(episode=ep["slug"], anchors=n, matched=matched, kickoff_at_s=off))
        print(report[-1], flush=True)
        if off is None:
            continue
        split = "test" if i >= len(eps) - n_test else "train"
        header = tuple(ep["matches"][0])
        ev = ev.assign(
            game_id=ep["slug"],
            g=ev.t + off,
            split=split,
            team=[
                t if fuzz.WRatio(h, header[0]) + fuzz.WRatio(a, header[1]) >= 150 else "unknown"
                for t, h, a in zip(ev.team, ev.home, ev.away, strict=True)
            ],
        )
        seg_frames.append(
            pd.DataFrame(
                dict(
                    game_id=ep["slug"],
                    split=split,
                    half=1,
                    seg=np.arange(len(segs)),
                    start=segs.audio_start,
                    end=segs.audio_end,
                    g_start=segs.audio_start,
                    g_end=segs.audio_end,
                    text=segs.text,
                    home=header[0],
                    away=header[1],
                )
            )
        )
        ev_frames.append(ev)
    if seg_frames:
        segs, events = pd.concat(seg_frames, ignore_index=True), pd.concat(ev_frames, ignore_index=True)
        df = label_segments(segs, events)
        df["input"] = build_inputs(df, segs[["game_id", "home", "away"]].drop_duplicates("game_id"))
        df.to_parquet(PROCESSED / "podcast_segments.parquet")
        events.to_parquet(PROCESSED / "podcast_events.parquet")
    (RESULTS / "podcast_alignment.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="newest N half-broadcasts only")
    ap.add_argument("--only", default=None, help="only episodes whose title contains this")
    args = ap.parse_args()
    rep = build(args.limit, args.only)
    ok = [r for r in rep if r["kickoff_at_s"] is not None]
    print(f"{len(ok)}/{len(rep)} episodes aligned on {sum(r['matched'] for r in ok)} matched events")


if __name__ == "__main__":
    main()
