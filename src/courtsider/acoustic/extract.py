"""Log-mel features for every SoccerNet half: python -m courtsider.acoustic.extract [--workers 3]

Downloads each 224p half from the gated Hugging Face dataset (SoccerNet NDA), extracts 16 kHz mono
audio, stores float16 log-mel (64 bands, 50 frames/s, ~19 MB per half) and deletes the video and
audio again, so disk use stays small. Resumable: halves that already have features are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from courtsider.paths import DATA, PROCESSED

REPO = "SoccerNet/SoccerNet_raw_HQ"
FEATURES = DATA / "audio_features"
SR, N_MELS, HOP = 16_000, 64, 320  # 50 frames per second
FPS = SR // HOP


def feature_path(game_id: str, half: int) -> Path:
    return FEATURES / game_id / f"{half}_mel.npy"


def log_mel(y: np.ndarray) -> np.ndarray:
    """(frames, 64) float16 log-mel; frame i covers audio up to (i * HOP + 1024) / SR (causal stamping)."""
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=HOP, n_mels=N_MELS, center=False, fmax=8000)
    return np.log(m + 1e-6).T.astype(np.float16)


def process(game_id: str, half: int) -> str:
    out = feature_path(game_id, half)
    if out.exists():
        return "skip"
    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory(dir=DATA) as tmp:
        video = hf_hub_download(
            REPO, f"{game_id}/{half}_224p.mkv", repo_type="dataset", revision="videos-224p", local_dir=tmp
        )
        wav = Path(tmp) / "a.wav"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", video, "-ac", "1", "-ar", str(SR), str(wav)], check=True
        )
        y, _ = librosa.load(wav, sr=SR)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, log_mel(y))
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--splits", default="train,valid,test")
    args = ap.parse_args()
    games = pd.read_parquet(PROCESSED / "games.parquet")
    order = {s: i for i, s in enumerate(args.splits.split(","))}
    games = games[games.split.isin(order)].assign(o=lambda d: d.split.map(order)).sort_values(["o", "game_id"])
    jobs = [(g, h) for g in games.game_id for h in (1, 2) if not feature_path(g, h).exists()]
    print(f"{len(jobs)} halves to process", flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(process, g, h): (g, h) for g, h in jobs}
        for f in as_completed(futs):
            g, h = futs[f]
            try:
                f.result()
                done += 1
            except Exception as e:  # keep going; a rerun retries failures
                print(f"FAILED {g} {h}: {e}", flush=True)
            if done % 10 == 0:
                rate = done / (time.time() - t0)
                eta = (len(jobs) - done) / max(rate, 1e-9) / 3600
                print(f"{done}/{len(jobs)} halves, {rate * 3600:.0f}/h, eta {eta:.1f} h", flush=True)


if __name__ == "__main__":
    main()
