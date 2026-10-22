"""Label recorded live radio with ESPN's event times: python -m courtsider.data.radio

The recorder (live.recorder) logs Whisper's English translation of Spanish radio commentary, each
segment stamped on the stream's audio clock, plus `status` rows that map that clock to wall time.
ESPN's key events carry wall-clock times derived from the match clock. Radio reaches us some seconds
after the action (encoder + CDN buffering), so per station we estimate that delay as the shift that
lines up the most ESPN goals with a goal call in the transcript (pooled over its sessions), then
label segments exactly like the SoccerNet ones (text.dataset.label_segments). Sessions are split by
time: the latest ~30% are test.

Spanish radio covers several matches at once, so the model input uses the session's header match
(as live inference does) and the team label is kept only for events in that match.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from courtsider.data import espn
from courtsider.live.recorder import LEAGUES, SPANISH_CLUBS, SPANISH_ONLY_IN
from courtsider.live.report import LIVE_DIR
from courtsider.paths import PROCESSED, RESULTS
from courtsider.text.dataset import LOOKAHEAD, LOOKBACK, build_inputs, label_segments

GOAL_CALL = re.compile(r"\bgo+a+l|\bgo+l\b|\bscores?\b|\bscored\b", re.I)
DELAYS = np.arange(-30.0, 120.5, 1.0)  # negative: ESPN's match-clock times can trail the real moment
DEFAULT_DELAY = 20.0  # used when a session has too few goals to estimate its own
TEST_FRACTION = 0.3


def sessions() -> list[dict]:
    out = []
    for path in sorted(LIVE_DIR.glob("*_radio-*.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        head = next((r for r in rows if r["kind"] == "session"), None)
        segs = pd.DataFrame([r for r in rows if r["kind"] == "segment"])
        status = pd.DataFrame([r for r in rows if r["kind"] == "status"])
        if head is None or len(segs) < 20 or len(status) < 2:
            continue
        mel = [r for r in rows if r["kind"] == "mel"]
        out.append(dict(id=path.stem, head=head, segs=segs, status=status, mel=mel))
    return out


def espn_events(t0: float, t1: float) -> pd.DataFrame:
    """Price-moving ESPN events between two wall times, in matches Spanish radio covers."""
    days = {(datetime.fromtimestamp(t, UTC) + timedelta(d)).date() for t in (t0, t1) for d in (-1, 0, 1)}
    rows = []
    for league in LEAGUES:
        for day in sorted(days):
            for m in espn._safe(espn.scoreboard, league, day) or []:
                s = espn._safe(espn.summary, league, m["id"])
                if not s or not s.get("header"):
                    continue
                home, away = espn.teams(s)
                if league in SPANISH_ONLY_IN and not ({home, away} & SPANISH_CLUBS):
                    continue
                rows += [dict(match_id=m["id"], **e) for e in espn.key_events(s) if t0 - 60 <= e["wall"] <= t1]
    return pd.DataFrame(rows).drop_duplicates(["match_id", "event", "wall"]) if rows else pd.DataFrame()


def delay_hits(segs: pd.DataFrame, goal_audio_at, delays: np.ndarray = DELAYS) -> np.ndarray:
    """For each candidate delay: how many ESPN goals have a transcript goal call next to them."""
    calls = segs[segs.text.str.contains(GOAL_CALL)].audio_start.to_numpy()
    out = []
    for d in delays:
        a = goal_audio_at(d)
        out.append(sum(bool(((calls >= t - LOOKBACK) & (calls <= t + LOOKAHEAD["goal"])).any()) for t in a))
    return np.array(out)


def best_delay(hits: np.ndarray, goals: int, delays: np.ndarray = DELAYS, default: float | None = DEFAULT_DELAY):
    """Middle of the best-scoring plateau (the window makes a range of shifts score equally)."""
    if goals < 3 or hits.max() < 2:
        return default
    return float(np.median(delays[hits == hits.max()]))


def build() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """-> (labelled segments with model inputs, events on the audio clock, per-session alignment report)."""
    seg_frames, ev_frames, report = [], [], []
    ss = sessions()
    n_test = max(1, round(TEST_FRACTION * len(ss))) if ss else 0
    for s in ss:
        st = s["status"]
        s["to_audio"] = lambda wall, st=st: np.interp(wall, st.wall, st.audio_seconds)
        s["events"] = espn_events(float(st.wall.min()), float(st.wall.max()))
        g = s["events"][s["events"].event == "goal"].wall.to_numpy() if len(s["events"]) else np.zeros(0)
        s["hits"], s["goals"] = delay_hits(s["segs"], lambda d, g=g, f=s["to_audio"]: f(g + d)), len(g)
    station = {}
    for s in ss:
        k = s["head"].get("station", "")
        h, n = station.get(k, (np.zeros(len(DELAYS)), 0))
        station[k] = (h + s["hits"], n + s["goals"])
    for i, s in enumerate(ss):
        segs, status, head, ev, to_audio = s["segs"], s["status"], s["head"], s["events"], s["to_audio"]
        if ev.empty:
            report.append(dict(session=s["id"], events=0))
            continue
        delay = best_delay(*station[head.get("station", "")])
        hits, n = int(s["hits"][np.abs(DELAYS - delay).argmin()]), s["goals"]
        header = (head["home"], head["away"])
        ev = ev.assign(
            game_id=s["id"],
            g=to_audio(ev.wall.to_numpy() + delay),
            team=[t if (h, a) == header else "unknown" for t, h, a in zip(ev.team, ev.home, ev.away, strict=True)],
        )
        ev = ev[(ev.g > 0) & (ev.g < status.audio_seconds.max())]  # inside the recorded audio
        split = "test" if i >= len(ss) - n_test else "train"
        seg = pd.DataFrame(
            dict(
                game_id=s["id"],
                split=split,
                half=1,
                seg=np.arange(len(segs)),
                start=segs.audio_start,
                end=segs.audio_end,
                g_start=segs.audio_start,
                g_end=segs.audio_end,
                text=segs.text,
            )
        )
        seg_frames.append(seg)
        ev_frames.append(ev.assign(split=split))
        report.append(
            dict(
                session=s["id"],
                split=split,
                delay_s=delay,
                goals=n,
                goals_with_call=hits,
                events=len(ev),
                minutes=round(float(status.audio_seconds.max()) / 60, 1),
                mel_files=len(s["mel"]),
            )
        )
    if not seg_frames:
        return pd.DataFrame(), pd.DataFrame(), report
    segs = pd.concat(seg_frames, ignore_index=True)
    events = pd.concat(ev_frames, ignore_index=True)
    df = label_segments(segs, events)
    games = pd.DataFrame(
        [
            dict(game_id=s["id"], home=s["head"]["home"], away=s["head"]["away"])
            for s in ss
            if s["id"] in set(segs.game_id)
        ]
    )
    df["input"] = build_inputs(df, games)
    return df, events, report


def audio_halves(split: str) -> list[dict]:
    """Logged radio log-mel files with goal times relative to each file, in acoustic.goal_model's format.

    shortcut: a goal in the first seconds of a 10-minute file loses its pre-goal context; fine while
    files are long relative to the 6 s target window.
    """
    path = PROCESSED / "radio_events.parquet"
    if not path.exists():
        return []
    ev = pd.read_parquet(path)
    out = []
    for s in sessions():
        goals = ev[(ev.game_id == s["id"]) & (ev.event == "goal") & (ev.split == split)].g.to_numpy()
        if not len(goals) and not len(ev[(ev.game_id == s["id"]) & (ev.split == split)]):
            continue  # session not labelled for this split
        for k, m in enumerate(s["mel"]):
            f = LIVE_DIR / f"{s['id']}_mel" / m["file"]
            if f.exists():
                g = goals - m["audio_start"]
                out.append(dict(game_id=s["id"], half=k, path=f, goals=np.sort(g[(g >= 0) & (g < m["frames"] / 50)])))
    return out


def main() -> None:
    df, events, report = build()
    if not df.empty:
        df.to_parquet(PROCESSED / "radio_segments.parquet")
        events.to_parquet(PROCESSED / "radio_events.parquet")
    (RESULTS / "radio_alignment.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))
    if not df.empty:
        print(df.groupby("split")[["goal", "red_card", "penalty"]].sum().assign(segments=df.groupby("split").size()))


if __name__ == "__main__":
    main()
