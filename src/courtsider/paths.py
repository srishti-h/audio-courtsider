"""Filesystem locations. Override the data root with COURTSIDER_DATA."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("COURTSIDER_DATA", ROOT / "data"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = DATA / "models"
AUDIO = DATA / "audio"
RESULTS = ROOT / "results"

SOCCERNET_DIR = RAW / "soccernet"
ECHOES_DIR = RAW / "echoes" / "Dataset"
FOOTBALLDATA_DIR = RAW / "footballdata"

for _p in (PROCESSED, MODELS, AUDIO, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)
