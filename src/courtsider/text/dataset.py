"""Weakly-supervised segment labels from the alignment between commentary and true event times.

A segment is positive for event type E when an E event happened in
[segment start - LOOKBACK, segment end + LOOKAHEAD[E]]. Lookahead is needed because (a) commentators
announce some events before the labelled moment (a corner is "given" before it is taken) and (b)
Echoes timestamps come from offline Whisper and are shifted by a few seconds relative to the
video labels (e.g. "Goal by Herrera" can be stamped ~8 s before the labelled goal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from courtsider.data.taxonomy import EVENT_TYPES

LOOKBACK = 15.0
LOOKAHEAD = {
    "goal": 8.0,
    "shot": 8.0,
    "red_card": 8.0,
    "yellow_card": 8.0,
    "offside": 8.0,
    "corner": 12.0,
    "penalty": 12.0,
    "substitution": 12.0,
}
TEAM_CLASSES = ("home", "away", "unknown")
CONTEXT_SEGMENTS = 2


def build_inputs(segments: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    """Model input = 'Home vs Away | previous segments || current segment' (streaming-safe: past only)."""
    teams = games.set_index("game_id")[["home", "away"]]
    out = []
    for gid, grp in segments.groupby("game_id", sort=False):
        home, away = teams.loc[gid]
        texts = grp.text.tolist()
        for i, cur in enumerate(texts):
            prev = " ".join(texts[max(0, i - CONTEXT_SEGMENTS) : i])
            out.append(f"{home} vs {away} | {prev} || {cur}")
    return pd.Series(out, index=segments.index)


def label_segments(segments: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Adds one 0/1 column per event type plus `team` (home/away/unknown) for the price-moving events."""
    labels = np.zeros((len(segments), len(EVENT_TYPES)), dtype=np.int8)
    team = np.full(len(segments), "unknown", dtype=object)
    seg_pos = {gid: idx for gid, idx in segments.groupby("game_id", sort=False).indices.items()}
    for gid, ev in events[events.event.notna()].groupby("game_id"):
        if gid not in seg_pos:
            continue
        idx = seg_pos[gid]
        gs = segments.g_start.to_numpy()[idx]
        ge = segments.g_end.to_numpy()[idx]
        for e in ev.itertuples():
            k = EVENT_TYPES.index(e.event)
            hit = (e.g >= gs - LOOKBACK) & (e.g <= ge + LOOKAHEAD[e.event])
            rows = idx[hit]
            labels[rows, k] = 1
            if e.event in ("goal", "red_card", "penalty") and e.team:
                team[rows] = e.team
    out = segments.copy()
    for k, name in enumerate(EVENT_TYPES):
        out[name] = labels[:, k]
    out["team"] = team
    return out
