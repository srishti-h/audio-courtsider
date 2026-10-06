"""FastAPI backend for the dashboard.

  GET  /api/games                 test-split matches that can be replayed
  GET  /api/results               headline experiment results (results/*.json)
  WS   /ws/replay?game_id=...     runs a deterministic simulation, then streams its tape in
                                  (scaled) real time: transcript, beliefs, feed, books, trades, P&L

Run: uvicorn courtsider.api.server:app --port 8000   (the React app proxies /api and /ws)
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import fields
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from courtsider.data.taxonomy import split_global
from courtsider.paths import PROCESSED, RESULTS, ROOT
from courtsider.sim.match import MatchSim, SimConfig
from courtsider.sim.signals import PricerFactory, final_score, signals_for_game

app = FastAPI(title="Audio Courtsider")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def _data():
    games = pd.read_parquet(PROCESSED / "games.parquet")
    events = pd.read_parquet(PROCESSED / "events.parquet")
    results = pd.read_parquet(PROCESSED / "results.parquet")
    beliefs = pd.read_parquet(PROCESSED / "beliefs.parquet")
    return games, events, beliefs, PricerFactory(games, results, events)


def _latency() -> float:
    try:
        asr = json.loads((RESULTS / "asr.json").read_text())["pipeline_latency_s"]
    except (FileNotFoundError, KeyError):
        asr = 1.5
    try:
        margin = json.loads((RESULTS / "text_eval.json").read_text())["timestamp_margin_s"]
    except (FileNotFoundError, KeyError):
        margin = 0.0
    return float(asr) + float(margin)


@app.get("/api/games")
def list_games() -> list[dict]:
    games, events, beliefs, _ = _data()
    ok = set(beliefs.game_id)
    g = games[(games.split == "test") & games.fd_home.notna() & games.game_id.isin(ok)]
    goals = events[events.event == "goal"].groupby("game_id").g.apply(list)
    return [
        dict(
            game_id=r.game_id,
            label=f"{r.home} {r.home_goals}-{r.away_goals} {r.away}",
            league=r.league,
            date=str(r.date.date()),
            goals=[round(x, 1) for x in goals.get(r.game_id, [])],
        )
        for r in g.sort_values("date").itertuples()
    ]


@app.get("/api/results")
def results() -> dict:
    out = {}
    for name in ("text_eval", "asr", "market", "pricing", "bench"):
        p = RESULTS / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def build_tape(game_id: str, cfg: SimConfig) -> tuple[list[dict], dict]:
    games, events, beliefs, factory = _data()
    ev = events[events.game_id == game_id]
    b = beliefs[beliefs.game_id == game_id]
    sim = MatchSim(
        factory(game_id), ev, signals_for_game(b, _latency()), cfg, final_score=final_score(games, game_id), record=True
    )
    summary = sim.run()
    tape = list(sim.tape)
    for x in sim.timeline:
        if x["kind"] == "signal":
            tape.append(dict(kind="signal", t=x["t"], probs=x["probs"], team=x["team"], text=x["text"]))
        else:
            tape.append(dict(x))
    tape.sort(key=lambda x: x["t"])
    for x in tape:
        half, t = split_global(x["t"])
        x["half"], x["clock"] = half, t
    meta = dict(
        game_id=game_id,
        home=games.set_index("game_id").loc[game_id, "home"],
        away=games.set_index("game_id").loc[game_id, "away"],
        latency=_latency(),
        summary={k: v for k, v in summary.items() if k != "pnl"} | {"pnl": summary["pnl"]},
    )
    return tape, meta


def _clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    return x


@app.websocket("/ws/replay")
async def replay(ws: WebSocket):
    await ws.accept()
    q = ws.query_params
    cfg_names = {f.name for f in fields(SimConfig)}
    overrides = {}
    for k, v in q.items():
        if k in cfg_names:
            overrides[k] = v.lower() == "true" if k == "mm_audio_guard" else float(v)
    cfg = SimConfig(**overrides)
    speed = float(q.get("speed", 5))
    start = float(q.get("start", 0))
    tape, meta = await asyncio.to_thread(build_tape, q["game_id"], cfg)
    await ws.send_json(_clean(dict(kind="meta", **meta)))
    try:
        prev = None
        for x in tape:
            if x["t"] < start:
                continue
            if prev is not None and x["t"] > prev:
                await asyncio.sleep(min((x["t"] - prev) / speed, 2.0))
            prev = x["t"]
            await ws.send_json(_clean(x))
        await ws.send_json(dict(kind="end"))
    except WebSocketDisconnect:
        return


_dist = ROOT / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
