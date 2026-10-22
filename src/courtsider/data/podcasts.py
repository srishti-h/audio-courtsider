"""Recent radio training data from Cadena SER's podcast archive: python -m courtsider.data.podcasts [--limit N]

SER publishes each Champions League / La Liga radio broadcast as "Los Partidos de la Jornada | A - B y
C - D | 1ª Parte / 2ª Parte" (about 50 min per half, back to Dec 2025). For each half this:
  1. downloads the episode (kept locally, never redistributed),
  2. transcribes it with Whisper (Spanish -> English) and stores log-mel features for the audio model,
  3. finds ESPN's goals for the matches named in the title (match-clock times within the half),
  4. estimates where the episode starts relative to kick-off: the offset that lines up the most ESPN
     goals with a goal call in the transcript (data.radio's alignment, over a wider range),
  5. labels transcript segments and audio frames exactly like the recorded radio.
Episodes whose alignment is not confident (fewer than 2 goals matched) are skipped. Alignment is good to
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
    path = ROOT / "audio" / f"{ep['slug']}.mp3"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(ep["url"], headers=UA, timeout=120, stream=True) as r:
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
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
        y = decode(download(ep))
        res = mlx_whisper.transcribe(
            y, path_or_hf_repo=WHISPER["small"], language="es", task="translate", condition_on_previous_text=False
        )
        tpath.parent.mkdir(parents=True, exist_ok=True)
        tpath.write_text(
            json.dumps([dict(start=s["start"], end=s["end"], text=s["text"].strip()) for s in res["segments"]])
        )
        mpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(mpath, log_mel(y))
    return pd.DataFrame(json.loads(tpath.read_text())).rename(columns={"start": "audio_start", "end": "audio_end"})


def espn_goals(ep: dict) -> pd.DataFrame:
    """ESPN price-moving events of the matches named in the title, timed on this half's match clock."""
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
                for e in espn.key_events(s):
                    if e["period"] == ep["half"]:
                        rows.append(dict(e, match_id=m["id"], t=e["clock"] - (ep["half"] - 1) * HALF_S))
    return pd.DataFrame(rows).drop_duplicates(["match_id", "event", "clock"]) if rows else pd.DataFrame()


def align(segs: pd.DataFrame, ev: pd.DataFrame) -> tuple[float | None, int, int]:
    """Episode time of kick-off -> (offset or None if not confident, goals matched, goals)."""
    goals = ev[ev.event == "goal"].t.to_numpy() if len(ev) else np.zeros(0)
    hits = radio.delay_hits(segs, lambda d: goals + d, OFFSETS)
    off = radio.best_delay(hits, len(goals), OFFSETS, default=None)
    matched = int(hits[np.abs(OFFSETS - off).argmin()]) if off is not None else int(hits.max(initial=0))
    return off, matched, len(goals)


def build(limit: int | None = None, only: str | None = None) -> list[dict]:
    eps = [e for e in episodes() if not only or only.lower() in e["title"].lower()][:limit]
    report, seg_frames, ev_frames = [], [], []
    n_test = max(1, round(radio.TEST_FRACTION * len(eps)))
    for i, ep in enumerate(sorted(eps, key=lambda e: (e["day"], e["half"]))):
        segs = transcribe_and_features(ep)
        ev = espn_goals(ep)
        off, matched, n = align(segs, ev) if len(ev) else (None, 0, 0)
        report.append(dict(episode=ep["slug"], goals=n, goals_matched=matched, kickoff_at_s=off))
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
    print(f"{len(ok)}/{len(rep)} episodes aligned, {sum(r['goals_matched'] for r in ok)} goals matched")


if __name__ == "__main__":
    main()
