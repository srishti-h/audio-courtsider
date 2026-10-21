"""Lead time of live commentary detections over the live-score feed, from session logs.

python -m courtsider.live.report            # summarises data/live/*.jsonl -> results/live.json

A detection "beats" a feed goal if it fired in the WINDOW seconds before the feed reported it; the
closest such detection is used, so unrelated earlier chatter never inflates the lead. When the feed
event names its match, ESPN's recorded time of the goal anchors a tighter window (10 s before to
30 s after the goal), so chatter about other matches on multi-match radio isn't credited.
Lead = feed wall time - detection wall time (positive = commentary was first).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from courtsider.paths import DATA, RESULTS

LIVE_DIR = DATA / "live"
WINDOW = 90.0
ANCHOR_BEFORE, ANCHOR_AFTER = 10.0, 30.0
SNIPPET_WORDS = 15


def goal_walls(match_id: str) -> list[float]:
    """ESPN's wall-clock time of every goal in a match (cached summaries; empty if unavailable)."""
    from courtsider.data import espn

    s = espn._safe(espn.summary, "all", match_id)
    return sorted(e["wall"] for e in espn.key_events(s) if e["event"] == "goal") if s and s.get("header") else []


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(events: list[dict]) -> dict:
    meta = next((e for e in events if e.get("kind") == "session"), {})
    feed = [e for e in events if e.get("kind") == "feed" and e.get("event") == "goal"]
    # fast speculative calls count as text; "audio" is the learned crowd+commentator goal model
    group = {"text": ("text", "text-fast"), "audio": ("audio",), "crowd": ("crowd",)}
    dets = {src: [e for e in events if e.get("kind") == "detection" and e.get("source") in group[src]] for src in group}
    goals = []
    used = {k: set() for k in dets}
    for f in feed:
        row = dict(clock=f.get("clock"), team=f.get("team"))
        if f.get("home"):
            row["match"] = f"{f['home']} vs {f['away']}"
        walls = (
            [w for w in goal_walls(f["match_id"]) if f["wall"] - 180 <= w <= f["wall"] + 5] if f.get("match_id") else []
        )
        lo, hi = (
            (walls[-1] - ANCHOR_BEFORE, min(f["wall"], walls[-1] + ANCHOR_AFTER))
            if walls
            else (f["wall"] - WINDOW, f["wall"])
        )
        for src, ds in dets.items():
            cands = [
                (i, d)
                for i, d in enumerate(ds)
                if i not in used[src] and lo <= d["wall"] <= hi and (src == "crowd" or d.get("event") == "goal")
            ]
            if cands:
                i, d = max(cands, key=lambda x: x[1]["wall"])  # the call closest before the feed (conservative)
                used[src].add(i)
                row[f"{src}_lead_s"] = round(f["wall"] - d["wall"], 1)
                if walls:
                    row[f"{src}_after_goal_s"] = round(d["wall"] - walls[-1], 1)
                if d.get("text"):  # a short quote only: the commentary belongs to the broadcaster
                    row[f"{src}_quote"] = " ".join(d["text"].split()[:SNIPPET_WORDS])
        goals.append(row)
    audio_min = max((e.get("audio_seconds", 0) for e in events if e.get("kind") == "status"), default=0) / 60
    fa = {
        src: len([d for i, d in enumerate(ds) if i not in used[src] and (src == "crowd" or d.get("event") == "goal")])
        for src, ds in dets.items()
    }
    return dict(
        match=f"{meta.get('home', '?')} vs {meta.get('away', '?')}",
        started=meta.get("started"),
        feed=meta.get("feed"),
        audio_minutes=round(audio_min, 1),
        feed_goals=len(feed),
        goals=goals,
        false_alarms=fa,
    )


def main() -> None:
    sessions = [summarize(load(p)) for p in sorted(LIVE_DIR.glob("*.jsonl"))]
    leads = {
        src: [g[f"{src}_lead_s"] for s in sessions for g in s["goals"] if f"{src}_lead_s" in g]
        for src in ("text", "audio", "crowd")
    }
    n_goals = sum(s["feed_goals"] for s in sessions)
    report = dict(
        sessions=sessions,
        n_sessions=len(sessions),
        n_feed_goals=n_goals,
        audio_minutes=round(sum(s["audio_minutes"] for s in sessions), 1),
        **{f"{src}_beats_feed": f"{len(v)}/{n_goals}" for src, v in leads.items()},
        **{f"{src}_lead_s_median": (float(np.median(v)) if v else None) for src, v in leads.items()},
    )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "live.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "sessions"}, indent=1))


if __name__ == "__main__":
    main()
