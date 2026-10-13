"""Load SoccerNet-Echoes commentary transcripts (Whisper segments, video-aligned)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from courtsider.data.taxonomy import global_time
from courtsider.paths import ECHOES_DIR

ECHOES_REPO = "https://github.com/SoccerNet/sn-echoes.git"
# English translations (Whisper translate task) cover every language in the dataset.
PREFERRED_SOURCES = ("whisper_v2_en", "whisper_v1_en")
# Untranslated Whisper v3 output, used only when the commentary is already English.
FALLBACK_SOURCE = "whisper_v3"
_STOPWORDS = set("the and is a to of in he it that for on with this was but at".split())


def _is_english(path: Path, threshold: float = 0.2) -> bool:
    words = " ".join(v[2] for v in json.loads(path.read_text())["segments"].values()).lower().split()
    return sum(w in _STOPWORDS for w in words) / max(1, len(words)) > threshold


def _find_files(root: Path, game_id: str) -> tuple[str, list[Path]] | None:
    for source in PREFERRED_SOURCES:
        files = [root / source / game_id / f"{h}_asr.json" for h in (1, 2)]
        if all(f.exists() for f in files):
            return source, files
    files = [root / FALLBACK_SOURCE / game_id / f"{h}_asr.json" for h in (1, 2)]
    if all(f.exists() for f in files) and _is_english(files[0]):
        return FALLBACK_SOURCE, files
    return None


def load(game_ids: list[str], root: Path = ECHOES_DIR) -> pd.DataFrame:
    rows = []
    for game_id in game_ids:
        found = _find_files(root, game_id)
        if found is None:
            continue
        source, files = found
        for half, f in enumerate(files, start=1):
            segs = json.loads(f.read_text())["segments"]
            # A few files carry malformed keys; order by start time instead of key.
            for idx, (start, end, text) in enumerate(sorted(segs.values(), key=lambda v: v[0])):
                text = text.strip()
                if not text or end <= start:
                    continue
                rows.append(
                    dict(
                        game_id=game_id,
                        source=source,
                        half=half,
                        seg=int(idx),
                        start=float(start),
                        end=float(end),
                        g_start=global_time(half, float(start)),
                        g_end=global_time(half, float(end)),
                        text=text,
                    )
                )
    return pd.DataFrame(rows).sort_values(["game_id", "g_start"]).reset_index(drop=True)
