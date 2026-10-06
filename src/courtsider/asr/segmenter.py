"""Group committed streaming words into commentary segments the text model can score.

A segment closes at a pause (gap between words) or sentence-final punctuation. Its availability
time is the commit (emit) time of its last word, so downstream detections stay causal.
"""

from __future__ import annotations

import pandas as pd

from courtsider.asr.streaming import Word


def segments_from_words(
    words: list[Word], game_id: str, half: int, offset: float = 0.0, pause: float = 0.6, max_words: int = 25
) -> pd.DataFrame:
    rows, cur = [], []

    def close():
        if cur:
            rows.append(
                dict(
                    game_id=game_id,
                    half=half,
                    start=offset + cur[0].start,
                    end=offset + cur[-1].end,
                    available=offset + max(w.emit for w in cur),
                    text=" ".join(w.text.strip() for w in cur),
                )
            )
            cur.clear()

    for i, w in enumerate(words):
        if cur and (w.start - cur[-1].end > pause or len(cur) >= max_words):
            close()
        cur.append(w)
        if w.text.strip().endswith((".", "!", "?")) and (i + 1 == len(words) or words[i + 1].emit > w.emit):
            close()
    close()
    df = pd.DataFrame(rows, columns=["game_id", "half", "start", "end", "available", "text"])
    df["g_start"] = df.start + (half - 1) * 3600.0
    df["g_end"] = df.end + (half - 1) * 3600.0
    return df
