"""Event-level detection metrics: precision, recall and lead time over the official feed.

A detection is a rising edge of P(event) >= threshold (one per refractory window), stamped at
segment end + ASR latency + radio delay. It matches a true event of the same type if it lands
in [t_e - early, t_e + late]. Lead over the feed = (t_e + feed_delay) - t_detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from courtsider.data.taxonomy import EVENT_TYPES


def detections(
    df: pd.DataFrame, probs: np.ndarray, event: str, threshold: float, latency: float = 0.0, refractory: float = 20.0
) -> pd.DataFrame:
    k = EVENT_TYPES.index(event)
    rows = []
    for gid, idx in df.groupby("game_id", sort=False).indices.items():
        p = probs[idx, k]
        t = df.g_end.to_numpy()[idx] + latency
        above = p >= threshold
        last = -np.inf
        for i in np.flatnonzero(above & ~np.concatenate([[False], above[:-1]])):
            if t[i] - last >= refractory:
                rows.append(dict(game_id=gid, t=t[i], p=p[i]))
                last = t[i]
    return pd.DataFrame(rows, columns=["game_id", "t", "p"])


def match(dets: pd.DataFrame, truth: pd.DataFrame, early: float = 10.0, late: float = 30.0) -> tuple[pd.DataFrame, int]:
    """Greedy one-to-one matching per game; returns matched pairs and # false positives."""
    pairs, fp = [], 0
    for gid, d in dets.groupby("game_id"):
        tr = truth[truth.game_id == gid].g.to_numpy()
        used = np.zeros(len(tr), bool)
        for td in d.t.to_numpy():
            dt = td - tr
            ok = np.flatnonzero((dt >= -early) & (dt <= late) & ~used)
            if len(ok):
                j = ok[np.argmin(np.abs(dt[ok]))]
                used[j] = True
                pairs.append(dict(game_id=gid, t_true=tr[j], t_det=td))
            else:
                fp += 1
    return pd.DataFrame(pairs, columns=["game_id", "t_true", "t_det"]), fp


def evaluate(
    df: pd.DataFrame,
    probs: np.ndarray,
    events: pd.DataFrame,
    event: str,
    threshold: float,
    latency: float,
    feed_delays: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0),
) -> dict:
    games = set(df.game_id)
    truth = events[(events.event == event) & events.game_id.isin(games)]
    dets = detections(df, probs, event, threshold, latency)
    pairs, fp = match(dets, truth)
    tp = len(pairs)
    delay = (pairs.t_det - pairs.t_true).to_numpy() if tp else np.array([np.nan])
    out = dict(
        event=event,
        threshold=threshold,
        n_true=len(truth),
        n_detections=len(dets),
        precision=tp / max(1, tp + fp),
        recall=tp / max(1, len(truth)),
        delay_p50=float(np.nanmedian(delay)),
        delay_p25=float(np.nanpercentile(delay, 25)),
        delay_p75=float(np.nanpercentile(delay, 75)),
    )
    for fd in feed_delays:
        lead = fd - delay
        out[f"beats_feed_{fd:g}s"] = float(np.mean(lead > 0)) if tp else 0.0
    return out
