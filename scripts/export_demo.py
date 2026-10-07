"""Export a static, backend-free demo for GitHub Pages: uv run python scripts/export_demo.py

For a few test matches, runs the simulation under a small grid of settings (official-feed delay
x audio-aware market maker) and keeps a window around each goal. Each window carries the state at
its start (books, fair values, P&L, earlier feed events) so the React app can replay it client-side.
Output: web/public/demo/{index,results}.json and one JSON tape per (match, goal, setting).
"""

from __future__ import annotations

import json
import re

from courtsider.api.server import _clean, _data, build_tape
from courtsider.paths import RESULTS, ROOT
from courtsider.sim.match import SimConfig

OUT = ROOT / "web" / "public" / "demo"
MATCHES = [
    "spain_laliga/2015-2016/2015-11-08 - 18-00 Barcelona 3 - 0 Villarreal",
    "spain_laliga/2016-2017/2017-02-11 - 22-45 Osasuna 1 - 3 Real Madrid",
    "italy_serie-a/2016-2017/2016-10-02 - 21-45 AS Roma 2 - 1 Inter",
    "italy_serie-a/2014-2015/2015-04-29 - 21-45 Juventus 3 - 2 Fiorentina",
    "italy_serie-a/2016-2017/2016-08-27 - 21-45 Napoli 4 - 2 AC Milan",
]
FEED_DELAYS = (1.0, 8.0)
GUARDS = (False, True)
THRESHOLD = 0.2  # chosen on validation in run_text
BEFORE, AFTER = 50.0, 70.0
ROUND = {"t": 2, "clock": 1, "cs": 0, "mm": 0, "minute": 1, "p": 3}


def slug(game_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", game_id.split("/")[-1].lower()).strip("-")


def _round(x):
    if isinstance(x, dict):
        return {k: (round(v, ROUND[k]) if k in ROUND and isinstance(v, float) else _round(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, float):
        return round(x, 3)
    return x


def window(tape: list[dict], start: float, end: float) -> list[dict]:
    """Messages in [start, end] plus the state needed to render the first frame."""
    prior, last_book, last = [], {}, {}
    for x in tape:
        if x["t"] >= start:
            break
        if x["kind"] == "book":
            last_book[x["instrument"]] = x
        elif x["kind"] in ("fair", "pnl"):
            last[x["kind"]] = x
        elif x["kind"] == "feed":
            prior.append(x)
        elif x["kind"] == "signal":
            last.setdefault("signals", []).append(x)
    head = prior + last.get("signals", [])[-8:] + list(last_book.values())
    head += [last[k] for k in ("fair", "pnl") if k in last]
    return [dict(x, t=max(x["t"], start - 0.01) if x["kind"] in ("book", "fair", "pnl") else x["t"]) for x in head] + [
        x for x in tape if start <= x["t"] <= end
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    games, events, _, _ = _data()
    g_idx = games.set_index("game_id")
    index = []
    for gid in MATCHES:
        goals = events[(events.game_id == gid) & (events.event == "goal")].sort_values("g")
        g = g_idx.loc[gid]
        entry = dict(
            id=slug(gid),
            label=f"{g.home} {g.home_goals}-{g.away_goals} {g.away}",
            date=str(g.date.date()),
            home=g.home,
            away=g.away,
            goals=[],
        )
        for fd in FEED_DELAYS:
            for guard in GUARDS:
                cfg = SimConfig(feed_delay=fd, cs_threshold=THRESHOLD, mm_audio_guard=guard, mm_guard_threshold=0.4)
                tape, meta = build_tape(gid, cfg)
                for k, goal in enumerate(goals.itertuples()):
                    w = window(tape, goal.g - BEFORE, goal.g + AFTER)
                    name = f"{slug(gid)}__goal{k + 1}__fd{int(fd)}__{'guard' if guard else 'plain'}.json"
                    payload = dict(
                        meta=dict(_clean(meta), kind="meta"),
                        start=round(goal.g - BEFORE, 2),
                        messages=_round(_clean(w)),
                    )
                    (OUT / name).write_text(json.dumps(payload, separators=(",", ":")))
                    if fd == FEED_DELAYS[0] and not guard:
                        entry["goals"].append(dict(n=k + 1, minute=round(float(goal.minute)), team=goal.team))
        index.append(entry)
        print(entry["label"], len(entry["goals"]), "goals")
    (OUT / "index.json").write_text(json.dumps(dict(matches=index, feed_delays=FEED_DELAYS, threshold=THRESHOLD)))
    results = {}
    for name in ("text_eval", "asr", "market", "pricing", "bench", "real_audio"):
        p = RESULTS / f"{name}.json"
        if p.exists():
            results[name] = json.loads(p.read_text())
    (OUT / "results.json").write_text(json.dumps(_clean(results)))
    size = sum(p.stat().st_size for p in OUT.glob("*.json")) / 1e6
    print(f"wrote {len(list(OUT.glob('*.json')))} files, {size:.1f} MB -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
