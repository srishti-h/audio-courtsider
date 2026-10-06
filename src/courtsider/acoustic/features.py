"""Non-speech signals that move before the words do: crowd roar and commentator excitement.

Both are unsupervised DSP detectors (no training data needed), so they run unchanged on real
broadcast audio:
  * crowd roar  — log energy in a high band where crowd noise dominates speech, z-scored against
                   a trailing median baseline, then a one-sided CUSUM change detector.
  * excitement  — commentator pitch (YIN f0) and loudness, each z-scored against a trailing
                   baseline and combined.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
import pandas as pd

SR = 16_000
HOP = 160  # 10 ms frames
FPS = SR // HOP


def _causal_median(x: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(x).rolling(window, min_periods=1).median().to_numpy()


def _trailing_z(x: np.ndarray, window: int) -> np.ndarray:
    """z-score of each frame against the median/MAD of the preceding `window` frames (strictly causal)."""
    s = pd.Series(x)
    med = s.shift(1).rolling(window, min_periods=10).median()
    mad = (s - med).abs().shift(1).rolling(window, min_periods=10).median()
    return ((s - med) / (1.4826 * mad + 1e-6)).fillna(0.0).to_numpy()


@dataclass
class AcousticTrack:
    times: np.ndarray  # seconds
    crowd_z: np.ndarray
    crowd_cusum: np.ndarray
    excitement: np.ndarray


def crowd_band_energy(y: np.ndarray, lo: float = 2500.0, hi: float = 7900.0) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=HOP, center=False)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=1024)
    band = S[(f >= lo) & (f <= hi)].sum(0)
    return np.log(band + 1e-10)


def cusum(z: np.ndarray, drift: float = 0.5, decay: float = 0.98) -> np.ndarray:
    """One-sided CUSUM on z-scores; leaky so the statistic returns to zero after a roar."""
    s = np.zeros_like(z)
    for i in range(1, len(z)):
        s[i] = max(0.0, decay * s[i - 1] + z[i] - drift)
    return s


def excitement(y: np.ndarray, baseline_s: float = 20.0) -> np.ndarray:
    f0 = librosa.yin(y, fmin=70, fmax=400, sr=SR, frame_length=1024, hop_length=HOP, center=False)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP, center=False)[0]
    n = min(len(f0), len(rms))
    f0, rms = np.log(f0[:n]), np.log(rms[:n] + 1e-6)
    w = int(baseline_s * FPS)
    zf = _trailing_z(_causal_median(f0, 25), w)
    zr = _trailing_z(_causal_median(rms, 25), w)
    return 0.5 * np.clip(zf, -5, 8) + 0.5 * np.clip(zr, -5, 8)


def analyse(y: np.ndarray, baseline_s: float = 20.0) -> AcousticTrack:
    e = crowd_band_energy(y)
    # 1 s trailing median: roars are sustained, speech fricatives last <200 ms
    z = _trailing_z(_causal_median(e, 100), int(baseline_s * FPS))
    ex = excitement(y, baseline_s)
    n = min(len(z), len(ex))
    # frames are stamped at their *end* so a feature is only used once its audio has arrived
    return AcousticTrack((np.arange(n) * HOP + 1024) / SR, z[:n], cusum(np.clip(z[:n], -5, 10)), ex[:n])


def onsets(track: np.ndarray, times: np.ndarray, threshold: float, refractory_s: float = 8.0) -> list[float]:
    """Times where `track` first crosses `threshold`, at most one per refractory window."""
    out: list[float] = []
    above = track >= threshold
    for i in np.flatnonzero(above[1:] & ~above[:-1]) + 1:
        if not out or times[i] - out[-1] >= refractory_s:
            out.append(float(times[i]))
    return out
