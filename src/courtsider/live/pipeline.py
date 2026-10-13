"""Live commentary pipeline: audio chunks in, timestamped events out.

feed(chunk)  -> buffer 16 kHz mono float32 audio as it arrives
tick()       -> about once a second: incremental Whisper (LocalAgreement-2), close segments,
                score them with the text model, update crowd-roar / excitement levels, and fire
                detections. Every event carries `wall` (time.time()) for lead-time measurement.
"""

from __future__ import annotations

import time

import numpy as np

from courtsider.acoustic.features import SR, analyse
from courtsider.asr.streaming import LocalAgreement, Word
from courtsider.data.taxonomy import EVENT_TYPES

ACOUSTIC_WINDOW_S = 25.0  # recent audio the crowd/excitement baseline is computed over
ROAR_THRESHOLD = 4.0
DETECT = ("goal", "penalty", "red_card")


class Segmenter:
    """Incremental version of segmenter.segments_from_words: close on pause, punctuation or length."""

    def __init__(self, pause: float = 0.6, max_words: int = 25):
        self.pause, self.max_words, self.cur = pause, max_words, []

    def add(self, words: list[Word]) -> list[list[Word]]:
        closed = []
        for w in words:
            if self.cur and (w.start - self.cur[-1].end > self.pause or len(self.cur) >= self.max_words):
                closed.append(self.cur)
                self.cur = []
            self.cur.append(w)
            if w.text.strip().endswith((".", "!", "?")):
                closed.append(self.cur)
                self.cur = []
        return closed


class LivePipeline:
    def __init__(
        self,
        home: str,
        away: str,
        vocabulary: list[str] | None = None,
        model: str = "small",
        threshold: float = 0.5,
        text_model=None,
    ):
        from courtsider.paths import MODELS
        from courtsider.text.models import TransformerModel

        self.home, self.away, self.threshold = home, away, threshold
        self.asr = LocalAgreement(model, vocabulary)
        self.text_model = text_model or TransformerModel.load(MODELS / "text_extractor")
        self.segmenter = Segmenter()
        self.history: list[str] = []
        self.recent = np.zeros(0, dtype=np.float32)
        self.started = time.time()
        self.last_fire: dict[str, float] = {}
        self.last_roar = -1e9

    @property
    def audio_seconds(self) -> float:
        return self.asr.t_avail

    def feed(self, chunk: np.ndarray) -> None:
        self.asr.append(chunk)
        self.recent = np.concatenate([self.recent, chunk.astype(np.float32)])[-int(ACOUSTIC_WINDOW_S * SR) :]

    def _score(self, segments: list[list[Word]], wall: float) -> list[dict]:
        if not segments:
            return []
        texts = [" ".join(w.text.strip() for w in s) for s in segments]
        inputs = []
        for t in texts:
            inputs.append(f"{self.home} vs {self.away} | {' '.join(self.history[-2:])} || {t}")
            self.history.append(t)
        probs, team = self.text_model.predict(inputs)
        out = []
        for seg, text, p, tm in zip(segments, texts, probs, team, strict=True):
            ev = dict(
                kind="segment",
                wall=wall,
                audio_start=seg[0].start,
                audio_end=seg[-1].end,
                text=text,
                probs={e: round(float(p[k]), 4) for k, e in enumerate(EVENT_TYPES)},
                team={"home": round(float(tm[0]), 3), "away": round(float(tm[1]), 3)},
            )
            out.append(ev)
            for e in DETECT:
                pe = ev["probs"][e]
                if pe >= self.threshold and wall - self.last_fire.get(e, -1e9) > 20:
                    self.last_fire[e] = wall
                    side = "home" if tm[0] >= tm[1] else "away"
                    out.append(dict(kind="detection", source="text", event=e, p=pe, team=side, text=text, wall=wall))
        return out

    def _acoustic(self, wall: float) -> list[dict]:
        if len(self.recent) < SR * 3:
            return []
        tr = analyse(self.recent)
        z, ex = float(tr.crowd_z[-1]), float(tr.excitement[-1])
        out = [dict(kind="acoustic", wall=wall, crowd_z=round(z, 2), excitement=round(ex, 2))]
        if z >= ROAR_THRESHOLD and wall - self.last_roar > 8:
            self.last_roar = wall
            out.append(dict(kind="detection", source="crowd", event="roar", p=None, team=None, wall=wall))
        return out

    def tick(self, final: bool = False) -> list[dict]:
        words, compute = self.asr.process(final)
        wall = time.time()
        for w in words:
            w.emit = wall
        out: list[dict] = []
        if words:
            out.append(
                dict(
                    kind="words",
                    wall=wall,
                    text=" ".join(w.text.strip() for w in words),
                    latency=round(float(self.audio_seconds - words[-1].end + compute), 2),
                )
            )
        closed = self.segmenter.add(words)
        if final and self.segmenter.cur:
            closed.append(self.segmenter.cur)
            self.segmenter.cur = []
        out += self._score(closed, wall)
        out += self._acoustic(wall)
        out.append(
            dict(kind="status", wall=wall, audio_seconds=round(self.audio_seconds, 1), asr_compute=round(compute, 3))
        )
        return out
