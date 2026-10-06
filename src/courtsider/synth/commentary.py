"""Synthetic commentary audio for validating the audio pipeline without broadcast rights.

Real commentary text (SoccerNet-Echoes) is re-voiced with macOS text-to-speech at its real
timestamps, and a crowd bed with reaction swells at the true event times is mixed underneath.
Speech timing and wording are therefore real; voice timbre, excitement and crowd reactions are
synthetic, and every number measured on this audio is reported as synthetic.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

SR = 16_000
VOICE = "Daniel"
# crowd reaction model: delay after the event (s), peak gain over the bed, decay time (s)
REACTIONS = {
    "goal": (0.4, 6.0, 9.0),
    "penalty": (0.6, 3.0, 4.0),
    "red_card": (0.8, 2.5, 4.0),
    "shot": (0.3, 2.0, 2.5),
    "corner": (0.5, 1.2, 2.0),
}
EXCITED = re.compile(r"\bgoa+l+\b|\bscores?\b|\bpenalt|\bred card\b|\bsent off\b|!\s*$", re.I)


@dataclass
class Excerpt:
    game_id: str
    half: int
    start: float
    end: float


def _say(text: str, rate: int, path: Path) -> np.ndarray:
    subprocess.run(["say", "-v", VOICE, "-r", str(rate), "-o", str(path), text], check=True, capture_output=True)
    y, _ = librosa.load(path, sr=SR)
    return y


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = np.fft.rfft(white) / np.sqrt(np.maximum(f, 20.0))
    x = np.fft.irfft(spec, n)
    return x / (np.abs(x).max() + 1e-9)


def crowd_track(n: int, event_times: list[tuple[float, str]], rng: np.random.Generator) -> np.ndarray:
    """Pink-noise crowd bed with slow murmur modulation plus event-locked swells."""
    t = np.arange(n) / SR
    bed = _pink_noise(n, rng) * (0.6 + 0.15 * np.sin(2 * np.pi * t / rng.uniform(7, 13)))
    env = np.ones(n)
    for te, ev in event_times:
        if ev not in REACTIONS:
            continue
        delay, gain, decay = REACTIONS[ev]
        onset = te + delay + rng.normal(0, 0.15)
        rise = np.clip((t - onset) / 0.6, 0, 1)
        fall = np.exp(-np.clip(t - onset - 0.6, 0, None) / decay)
        env += (gain - 1) * rise * fall * (t >= onset)
    roar = _pink_noise(n, rng)
    roar = librosa.effects.preemphasis(roar)  # brighter: roars carry more high-frequency energy than murmur
    return 0.05 * bed * env + 0.04 * roar / (np.abs(roar).max() + 1e-9) * (env - 1)


def render(excerpt: Excerpt, segments: pd.DataFrame, events: pd.DataFrame, out: Path, seed: int = 0) -> dict:
    """Write a 16 kHz wav for the excerpt; returns the reference script with timings."""
    rng = np.random.default_rng(seed)
    dur = excerpt.end - excerpt.start
    n = int(dur * SR)
    speech = np.zeros(n)
    segs = segments[
        (segments.game_id == excerpt.game_id)
        & (segments.half == excerpt.half)
        & (segments.start >= excerpt.start)
        & (segments.end <= excerpt.end)
    ]
    script = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, s in enumerate(segs.itertuples()):
            words = len(s.text.split())
            slot = max(s.end - s.start, 0.5)
            rate = int(np.clip(60 * words / slot, 150, 330))
            excited = bool(EXCITED.search(s.text))
            y = _say(s.text, rate + (40 if excited else 0), Path(tmp) / f"{i}.aiff")
            if excited:  # raise pitch ~3 semitones by resampling (also slightly faster speech)
                y = resample_poly(y, 84, 100) * 1.6
            a = int((s.start - excerpt.start) * SR)
            y = y[: n - a]
            speech[a : a + len(y)] += y * 0.5
            script.append(dict(start=s.start - excerpt.start, end=(a + len(y)) / SR, text=s.text, excited=excited))
    ev = events[(events.game_id == excerpt.game_id) & (events.half == excerpt.half)]
    ev = ev[(ev.t >= excerpt.start) & (ev.t < excerpt.end) & ev.event.notna()]
    event_times = [(e.t - excerpt.start, e.event) for e in ev.itertuples()]
    mix = speech + crowd_track(n, event_times, rng)
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.9
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, mix.astype(np.float32), SR)
    return dict(
        game_id=excerpt.game_id,
        half=excerpt.half,
        offset=excerpt.start,
        duration=dur,
        script=script,
        events=[dict(t=t, event=e) for t, e in event_times],
    )
