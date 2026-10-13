"""Does speculative scoring detect goals sooner on real broadcast audio? python -m courtsider.eval.run_fast_text

Windows around every labelled goal in the real-audio test matches (see run_real_audio), plus
goal-free windows for false alarms, streamed through the live pipeline on a virtual clock that
advances by measured compute time. Compares the confirmed-segment path (1 s step) with the fast
path (0.5 s step + speculative scoring of unconfirmed words).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import soundfile as sf

from courtsider.asr.streaming import SR, mine_names
from courtsider.eval.run_real_audio import pick_games
from courtsider.live.pipeline import LivePipeline
from courtsider.paths import AUDIO, MODELS, PROCESSED, RESULTS
from courtsider.text.models import TransformerModel

BEFORE, AFTER = 45.0, 35.0
CONFIGS = {
    "confirmed (1 s step)": dict(step=1.0, fast=False),
    "fast (0.5 s step + speculative)": dict(step=0.5, fast=True),
}


def stream_window(y: np.ndarray, home: str, away: str, names: list[str], model, step: float, fast: bool) -> list[dict]:
    pipe = LivePipeline(home, away, names, text_model=model, fast=fast, acoustic=False)
    dets, t = [], 0.0
    dur = len(y) / SR
    while t < dur:
        nxt = min(t + step, dur)
        pipe.feed(y[int(t * SR) : int(nxt * SR)])
        dets += [e for e in pipe.tick(final=nxt >= dur, audio_clock=nxt) if e["kind"] == "detection"]
        t = nxt
    return [d for d in dets if d["event"] == "goal"]


def windows(n_neg: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    games = pick_games(5)
    events = pd.read_parquet(PROCESSED / "events.parquet")
    segs = pd.read_parquet(PROCESSED / "segments.parquet")
    out = []
    for g in games.itertuples():
        names = mine_names(segs[segs.game_id == g.game_id].text.tolist(), (g.home, g.away))
        for half in (1, 2):
            wav = AUDIO / f"real_{g.Index}_{half}.wav"
            if not wav.exists():
                continue
            goals = np.sort(events[(events.game_id == g.game_id) & (events.half == half) & (events.event == "goal")].t)
            goals = goals[np.concatenate([[True], np.diff(goals) > 5])] if len(goals) else goals
            for t in goals:
                out.append(dict(wav=wav, home=g.home, away=g.away, names=names, start=max(0, t - BEFORE), goal=t))
            for _ in range(n_neg // 5):
                for _ in range(50):
                    s = float(rng.uniform(60, 2500))
                    if not ((goals > s - 40) & (goals < s + 120)).any():
                        out.append(dict(wav=wav, home=g.home, away=g.away, names=names, start=s, goal=None))
                        break
    return out


def main() -> None:
    model = TransformerModel.load(MODELS / "text_extractor")
    wins = windows()
    cache: dict = {}
    report = {}
    for name, cfg in CONFIGS.items():
        delays, misses, fa, neg_minutes = [], 0, 0, 0.0
        for w in wins:
            if w["wav"] not in cache:
                cache[w["wav"]] = sf.read(w["wav"])[0].astype(np.float32)
            y = cache[w["wav"]][int(w["start"] * SR) : int((w["start"] + BEFORE + AFTER) * SR)]
            dets = stream_window(y, w["home"], w["away"], w["names"], model, **cfg)
            times = np.array([d["wall"] + w["start"] for d in dets])
            if w["goal"] is None:
                fa += len(times)
                neg_minutes += len(y) / SR / 60
            else:
                hit = times[(times >= w["goal"] - 10) & (times <= w["goal"] + 30)]
                if len(hit):
                    delays.append(float(hit.min() - w["goal"]))
                else:
                    misses += 1
        n = len(delays) + misses
        report[name] = dict(
            goals=n,
            recall=round(len(delays) / max(1, n), 3),
            delay_p50=round(float(np.median(delays)), 2) if delays else None,
            delay_p25=round(float(np.percentile(delays, 25)), 2) if delays else None,
            false_alarms_per_min=round(fa / max(neg_minutes, 1e-9), 3),
            negative_minutes=round(neg_minutes, 1),
        )
        print(name, report[name], flush=True)
    (RESULTS / "fast_text.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
