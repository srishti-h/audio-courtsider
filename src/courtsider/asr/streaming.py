"""Streaming Whisper on Apple silicon (mlx-whisper) with the LocalAgreement-2 commit policy.

Whisper is an offline model, so streaming re-transcribes a sliding audio buffer every `step`
seconds and only *commits* words on which two consecutive hypotheses agree (Macháček et al.,
2023). The simulation runs on a virtual clock that advances by the real measured compute time,
so emission latencies reflect this hardware: if transcription is slower than real time, the
lag accumulates exactly as it would live.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

SR = 16_000
MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


@dataclass
class Word:
    text: str
    start: float  # audio time (s)
    end: float
    emit: float = 0.0  # virtual wall time when committed

    @property
    def norm(self) -> str:
        return re.sub(r"[^\w']", "", self.text.lower())


@dataclass
class StreamResult:
    words: list[Word]
    compute_times: list[float] = field(default_factory=list)
    chunks: list[tuple[float, list[Word]]] = field(default_factory=list)  # (emit time, words committed)

    def latencies(self) -> np.ndarray:
        return np.array([w.emit - w.end for w in self.words])

    def text(self) -> str:
        return " ".join(w.text.strip() for w in self.words)


def _hypothesis(audio: np.ndarray, offset: float, repo: str, prompt: str | None) -> list[Word]:
    import mlx_whisper

    res = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=repo,
        word_timestamps=True,
        language="en",
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=prompt,
        verbose=None,
    )
    out = []
    for seg in res["segments"]:
        for w in seg.get("words", []):
            out.append(Word(w["word"], offset + float(w["start"]), offset + float(w["end"])))
    return out


def stream(
    y: np.ndarray,
    model: str = "turbo",
    step: float = 1.0,
    max_buffer: float = 12.0,
    vocabulary: list[str] | None = None,
) -> StreamResult:
    """Simulate live streaming over audio `y`; `vocabulary` biases decoding (e.g. team-sheet names)."""
    repo = MODELS.get(model, model)
    dur = len(y) / SR
    committed: list[Word] = []
    prev: list[Word] = []
    buf_start = 0.0
    clock = step
    result = StreamResult(committed)
    base_prompt = ("Football commentary. Players: " + ", ".join(vocabulary) + ".") if vocabulary else None
    while True:
        t_avail = min(clock, dur)
        audio = y[int(buf_start * SR) : int(t_avail * SR)].astype(np.float32)
        tail = " ".join(w.text.strip() for w in committed[-20:])
        prompt = " ".join(p for p in (base_prompt, tail) if p) or None
        t0 = time.perf_counter()
        hyp = _hypothesis(audio, buf_start, repo, prompt) if len(audio) > SR * 0.3 else []
        compute = time.perf_counter() - t0
        result.compute_times.append(compute)
        last_end = committed[-1].end if committed else 0.0
        new = [w for w in hyp if w.start >= last_end - 0.05]
        old = [w for w in prev if w.start >= last_end - 0.05]
        agreed = []
        for a, b in zip(new, old, strict=False):
            if a.norm != b.norm or not a.norm:
                break
            agreed.append(a)
        final = t_avail >= dur
        if final:  # flush everything at end of audio
            agreed = new
        emit = t_avail + compute
        for w in agreed:
            w.emit = emit
        if agreed:
            committed.extend(agreed)
            result.chunks.append((emit, agreed))
        prev = hyp
        if final:
            break
        # trim the buffer at the last committed word once it grows too long
        if t_avail - buf_start > max_buffer and committed:
            buf_start = max(buf_start, committed[-1].end)
            prev = [w for w in prev if w.start >= buf_start]
        clock = t_avail + max(step, compute)
    return result


# ---- evaluation helpers -----------------------------------------------------------------

_STOP = set(
    """The He His And But It This That What With From For Into Now Here There They Their Them She Her
    We You I A An In On At Of To By As Is Was Be So Well Yes No Oh Not All One Two Three Good Great
    Very Just Still Then When Where Who Why How Again Back Out Up Down Over Off Free Kick Corner Goal
    Penalty Ball Half Time Minute Minutes Game Match League Champions Cup Premier Liga Serie United City
    Real Madrid Barcelona Football Referee Second First Third Left Right Mister Yeah Okay Let Come Go""".split()
)


def mine_names(texts: list[str], teams: tuple[str, str], top: int = 40, min_count: int = 3) -> list[str]:
    """Proxy team sheet: capitalised, non-sentence-initial tokens that recur in the commentary."""
    team_tokens = {t for team in teams for t in team.split()}
    c: Counter[str] = Counter()
    for text in texts:
        toks = re.findall(r"[A-Za-zÀ-ÿ'\-]+", text)
        for i, tok in enumerate(toks):
            if i > 0 and tok[:1].isupper() and len(tok) > 2 and tok not in _STOP and tok not in team_tokens:
                c[tok] += 1
    return [w for w, n in c.most_common(top) if n >= min_count]


def _norm_tokens(text: str) -> list[str]:
    return [t for t in re.sub(r"[^\w' ]", " ", text.lower()).split() if t]


def wer(ref: str, hyp: str) -> float:
    from rapidfuzz.distance import Levenshtein

    r, h = _norm_tokens(ref), _norm_tokens(hyp)
    return Levenshtein.distance(r, h) / max(1, len(r))


def name_recall(ref: str, hyp: str, names: list[str]) -> tuple[int, int]:
    """(# name mentions recovered, # name mentions in reference), counted per name."""
    r, h = Counter(_norm_tokens(ref)), Counter(_norm_tokens(hyp))
    hit = tot = 0
    for n in {x.lower() for x in names}:
        tot += r[n]
        hit += min(r[n], h[n])
    return hit, tot
