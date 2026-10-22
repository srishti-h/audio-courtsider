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

ACOUSTIC_WINDOW_S = 40.0  # recent audio kept for the crowd/excitement baseline and the audio goal model
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


class AudioGoalDetector:
    """The learned causal crowd/commentator goal detectors (acoustic.goal_model), applied to recent audio.
    With the fast detector trained too, P(goal) is the higher of the two calibrated probabilities."""

    def __init__(self, models: list, threshold: float):
        self.models, self.threshold = models, threshold  # [(net, calibrator), ...]

    @classmethod
    def load(cls) -> AudioGoalDetector | None:
        import json
        import pickle

        import torch

        from courtsider.acoustic.goal_model import DETECTORS, CausalTCN, _device
        from courtsider.paths import MODELS, RESULTS

        r = RESULTS / "audio_goal.json"
        models = []
        for d in DETECTORS.values():
            w, c = MODELS / f"audio_goal_tcn{d['file']}.pt", MODELS / f"audio_goal_calibrator{d['file']}.pkl"
            if w.exists() and c.exists():
                net = CausalTCN().to(_device())
                net.load_state_dict(torch.load(w, map_location=_device()))
                net.eval()
                models.append((net, pickle.loads(c.read_bytes())))
        if not models or not r.exists():
            return None
        return cls(models, json.loads(r.read_text())["threshold_selected_on_valid"])

    def p_goal(self, audio: np.ndarray) -> float:
        import torch

        from courtsider.acoustic.extract import log_mel
        from courtsider.acoustic.goal_model import _device, normalise

        x = torch.tensor(normalise(log_mel(audio))[None], device=_device())
        with torch.no_grad():
            return max(float(cal.predict([torch.sigmoid(net(x)[0, -1]).item()])[0]) for net, cal in self.models)


class LivePipeline:
    """`fast=True` adds speculative scoring: every tick the text model also reads the sentence still in
    progress plus Whisper's newest *unconfirmed* words, so a shouted "GOAL!" can fire before the
    sentence ends or the commit policy confirms it. Confirmed segments are still scored as before."""

    def __init__(
        self,
        home: str,
        away: str,
        vocabulary: list[str] | None = None,
        model: str = "small",
        threshold: float = 0.5,
        text_model=None,
        fast: bool = True,
        acoustic: bool = True,
        language: str = "en",
        task: str = "transcribe",
    ):
        from courtsider.paths import MODELS
        from courtsider.text.models import TransformerModel

        self.home, self.away, self.threshold, self.fast, self.acoustic = home, away, threshold, fast, acoustic
        self.asr = LocalAgreement(model, vocabulary, language=language, task=task)
        self.text_model = text_model or TransformerModel.load(MODELS / "text_extractor")
        self.segmenter = Segmenter()
        self.history: list[str] = []
        self.recent = np.zeros(0, dtype=np.float32)
        self.started = time.time()
        self.last_fire: dict[str, float] = {}
        self.last_roar = -1e9
        self.last_spec = ""
        self.audio_goal = AudioGoalDetector.load() if acoustic else None

    @property
    def audio_seconds(self) -> float:
        return self.asr.t_avail

    def feed(self, chunk: np.ndarray) -> None:
        self.asr.append(chunk)
        if self.acoustic:
            self.recent = np.concatenate([self.recent, chunk.astype(np.float32)])[-int(ACOUSTIC_WINDOW_S * SR) :]

    def _input(self, text: str) -> str:
        return f"{self.home} vs {self.away} | {' '.join(self.history[-2:])} || {text}"

    def _fire(self, probs, team, text: str, wall: float, source: str) -> list[dict]:
        out = []
        for k, e in enumerate(EVENT_TYPES):
            if e in DETECT and probs[k] >= self.threshold and wall - self.last_fire.get(e, -1e9) > 20:
                self.last_fire[e] = wall  # shared across sources: one alert per event
                side = "home" if team[0] >= team[1] else "away"
                out.append(
                    dict(
                        kind="detection",
                        source=source,
                        event=e,
                        p=round(float(probs[k]), 4),
                        team=side,
                        text=text,
                        wall=wall,
                    )
                )
        return out

    def _score(self, segments: list[list[Word]], wall) -> list[dict]:
        if not segments:
            return []
        texts = [" ".join(w.text.strip() for w in s) for s in segments]
        inputs = []
        for t in texts:
            inputs.append(self._input(t))
            self.history.append(t)
        probs, team = self.text_model.predict(inputs)
        out = []
        for seg, text, p, tm in zip(segments, texts, probs, team, strict=True):
            w = wall()
            out.append(
                dict(
                    kind="segment",
                    wall=w,
                    audio_start=seg[0].start,
                    audio_end=seg[-1].end,
                    text=text,
                    probs={e: round(float(p[k]), 4) for k, e in enumerate(EVENT_TYPES)},
                    team={"home": round(float(tm[0]), 3), "away": round(float(tm[1]), 3)},
                )
            )
            out += self._fire(p, tm, text, w, "text")
        return out

    def _speculate(self, wall) -> list[dict]:
        words = self.segmenter.cur + self.asr.pending()
        text = " ".join(w.text.strip() for w in words)
        if not text or text == self.last_spec:
            return []
        self.last_spec = text
        probs, team = self.text_model.predict([self._input(text)])
        w = wall()
        out = [dict(kind="partial", wall=w, text=text, p_goal=round(float(probs[0][0]), 4))]
        return out + self._fire(probs[0], team[0], text, w, "text-fast")

    def _acoustic(self, wall: float) -> list[dict]:
        if not self.acoustic or len(self.recent) < SR * 3:
            return []
        tr = analyse(self.recent)
        z, ex = float(tr.crowd_z[-1]), float(tr.excitement[-1])
        p_audio = self.audio_goal.p_goal(self.recent) if self.audio_goal else None
        out = [
            dict(
                kind="acoustic",
                wall=wall,
                crowd_z=round(z, 2),
                excitement=round(ex, 2),
                p_goal_audio=None if p_audio is None else round(p_audio, 4),
            )
        ]
        if z >= ROAR_THRESHOLD and wall - self.last_roar > 8:
            self.last_roar = wall
            out.append(dict(kind="detection", source="crowd", event="roar", p=None, team=None, wall=wall))
        if (
            p_audio is not None
            and p_audio >= self.audio_goal.threshold
            and wall - self.last_fire.get("audio", -1e9) > 20
        ):
            self.last_fire["audio"] = wall
            out.append(dict(kind="detection", source="audio", event="goal", p=round(p_audio, 4), team=None, wall=wall))
        return out

    def tick(self, final: bool = False, audio_clock: float | None = None) -> list[dict]:
        """Process new audio. With `audio_clock` (simulation), event times are audio time + measured
        compute so far; otherwise they are wall-clock times."""
        t0 = time.perf_counter()

        def wall() -> float:
            return audio_clock + time.perf_counter() - t0 if audio_clock is not None else time.time()

        words, compute = self.asr.process(final)
        w = wall()
        for word in words:
            word.emit = w
        out: list[dict] = []
        if words:
            out.append(
                dict(
                    kind="words",
                    wall=w,
                    text=" ".join(x.text.strip() for x in words),
                    latency=round(float(self.audio_seconds - words[-1].end + compute), 2),
                )
            )
        closed = self.segmenter.add(words)
        if final and self.segmenter.cur:
            closed.append(self.segmenter.cur)
            self.segmenter.cur = []
        out += self._score(closed, wall)
        if self.fast and not final:
            out += self._speculate(wall)
        out += self._acoustic(wall())
        out.append(
            dict(kind="status", wall=wall(), audio_seconds=round(self.audio_seconds, 1), asr_compute=round(compute, 3))
        )
        return out
