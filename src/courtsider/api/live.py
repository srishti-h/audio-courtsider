"""Live mode endpoints.

GET /api/live/matches            matches from the live-score feed (live first)
GET /api/live/sessions           lead-time summaries of logged live sessions
WS  /ws/live?match_id=...        browser sends 16 kHz mono float32 PCM frames; the server streams
                                 back words, scored segments, crowd/excitement levels, detections
                                 and feed events. Every event is logged to data/live/*.jsonl.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from courtsider.live.feeds import diff, get_feed, roster_vocabulary
from courtsider.live.report import LIVE_DIR, load, summarize

router = APIRouter()
POLL_S = 5.0
TICK_S = 1.0
_text_model = None


def _model():
    global _text_model
    if _text_model is None:
        from courtsider.paths import MODELS
        from courtsider.text.models import TransformerModel

        _text_model = TransformerModel.load(MODELS / "text_extractor")
    return _text_model


@router.get("/api/live/matches")
def live_matches() -> list[dict]:
    try:
        matches = get_feed().matches()[:60]
    except Exception as e:  # feed down or offline: live listening still works without it
        return [dict(error=str(e))]
    try:
        from courtsider.pricing.current import fair_1x2

        for m in matches:  # pre-match fair prices from the model fit on the latest seasons
            m["fair"] = fair_1x2(m["home"], m["away"])
    except Exception:  # no processed results yet (fresh clone): list matches without prices
        pass
    return matches


@router.get("/api/live/sessions")
def live_sessions() -> list[dict]:
    return [summarize(load(p)) for p in sorted(LIVE_DIR.glob("*.jsonl"), reverse=True)]


@router.websocket("/ws/live")
async def live(ws: WebSocket):
    from courtsider.live.pipeline import LivePipeline

    await ws.accept()
    q = ws.query_params
    feed = get_feed(q.get("feed"))
    match_id = q.get("match_id")
    home, away, vocab, snap = q.get("home", "Home"), q.get("away", "Away"), [], None
    if match_id:
        try:
            snap = await asyncio.to_thread(feed.snapshot, match_id)
            home, away, vocab = snap.home, snap.away, roster_vocabulary(snap.lineups)
        except Exception as e:
            await ws.send_json(dict(kind="error", message=f"feed unavailable: {e}"))
    pipe = await asyncio.to_thread(
        LivePipeline, home, away, vocab, q.get("model", "small"), float(q.get("threshold", 0.5)), _model()
    )
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{home}-{away}".lower()).strip("-")
    log = (LIVE_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{slug}.jsonl").open("a")
    send_lock = asyncio.Lock()

    async def emit(events: list[dict]) -> None:
        for e in events:
            log.write(json.dumps(e) + "\n")
        log.flush()
        async with send_lock:
            for e in events:
                await ws.send_json(e)

    await emit(
        [
            dict(
                kind="session",
                wall=time.time(),
                started=datetime.now().isoformat(timespec="seconds"),
                home=home,
                away=away,
                match_id=match_id,
                feed=feed.name,
                vocabulary=vocab[:60],
                score=list(snap.score) if snap else None,
                clock=snap.clock if snap else None,
            )
        ]
    )
    stop = asyncio.Event()

    async def watch_feed():
        prev = snap
        while match_id and not stop.is_set():
            await asyncio.sleep(POLL_S)
            try:
                cur = await asyncio.to_thread(feed.snapshot, match_id)
            except Exception:
                continue
            events = diff(prev, cur)
            events.append(dict(kind="score", wall=time.time(), score=list(cur.score), clock=cur.clock, state=cur.state))
            await emit(events)
            prev = cur

    async def process():
        while not stop.is_set():
            await asyncio.sleep(TICK_S)
            if pipe.audio_seconds > 0:
                await emit(await asyncio.to_thread(pipe.tick))

    tasks = [asyncio.create_task(watch_feed()), asyncio.create_task(process())]
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes"):
                pipe.feed(np.frombuffer(msg["bytes"], dtype=np.float32))
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        log.close()
