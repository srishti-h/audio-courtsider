"""Lead time of live commentary detections over the live-score feed, from session logs.

python -m courtsider.live.report            # summarises data/live/*.jsonl -> results/live.json

A detection "beats" a feed goal if it fired in the WINDOW seconds before the feed reported it.
Lead = feed wall time - detection wall time (positive = commentary was first).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from courtsider.paths import DATA, RESULTS

LIVE_DIR = DATA / "live"
WINDOW = 90.0


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(events: list[dict]) -> dict:
    meta = next((e for e in events if e.get("kind") == "session"), {})
    feed = [e for e in events if e.get("kind") == "feed" and e.get("event") == "goal"]
    dets = {
        src: [e for e in events if e.get("kind") == "detection" and e.get("source") == src] for src in ("text", "crowd")
    }
    goals = []
    used = {k: set() for k in dets}
    for f in feed:
        row = dict(clock=f.get("clock"), team=f.get("team"))
        for src, ds in dets.items():
            cands = [
                (i, d)
                for i, d in enumerate(ds)
                if i not in used[src]
                and 0 <= f["wall"] - d["wall"] <= WINDOW
                and (src == "crowd" or d.get("event") == "goal")
            ]
            if cands:
                i, d = min(cands, key=lambda x: x[1]["wall"])
                used[src].add(i)
                row[f"{src}_lead_s"] = round(f["wall"] - d["wall"], 1)
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
        for src in ("text", "crowd")
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
