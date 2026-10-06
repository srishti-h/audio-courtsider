"""Real broadcast audio (requires SoccerNet NDA access): python -m courtsider.eval.run_real_audio --games 5

Download the low-res videos first (password from the SoccerNet NDA):
    from SoccerNet.Downloader import SoccerNetDownloader
    d = SoccerNetDownloader(LocalDirectory="data/raw/soccernet"); d.password = "<NDA password>"
    d.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["test"])

For each half: extract 16 kHz audio, stream it through Whisper (turbo, team-sheet biasing), score
segments with the text model, run the crowd/excitement detectors, and measure how long after each
labelled goal each signal fires. Unlike the Echoes-based results, these are timed on real audio.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from courtsider.acoustic.features import analyse, onsets
from courtsider.asr.segmenter import segments_from_words
from courtsider.asr.streaming import mine_names, stream
from courtsider.paths import AUDIO, MODELS, PROCESSED, RESULTS, SOCCERNET_DIR


def extract_audio(video: Path, out: Path) -> Path:
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", str(out)], check=True
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    from courtsider.text.models import TransformerModel

    games = pd.read_parquet(PROCESSED / "games.parquet")
    events = pd.read_parquet(PROCESSED / "events.parquet")
    segs = pd.read_parquet(PROCESSED / "segments.parquet")
    model = TransformerModel.load(MODELS / "text_extractor")
    avail = [
        g for g in games[games.split == "test"].itertuples() if (SOCCERNET_DIR / g.game_id / "1_224p.mkv").exists()
    ][: args.games]
    if not avail:
        raise SystemExit("No SoccerNet videos found; see the module docstring for NDA download steps.")
    rows = []
    for g in avail:
        names = mine_names(segs[segs.game_id == g.game_id].text.tolist(), (g.home, g.away))
        for half in (1, 2):
            wav = extract_audio(
                SOCCERNET_DIR / g.game_id / f"{half}_224p.mkv", AUDIO / f"real_{abs(hash(g.game_id))}_{half}.wav"
            )
            y, _ = sf.read(wav)
            r = stream(y, "turbo", vocabulary=names)
            seg = segments_from_words(r.words, g.game_id, half)
            texts = seg.text.tolist()
            inputs = [
                f"{g.home} vs {g.away} | {' '.join(texts[max(0, i - 2) : i])} || {t}" for i, t in enumerate(texts)
            ]
            p = model.predict(inputs)[0][:, 0] if inputs else np.array([])
            fired = seg.available.to_numpy()[p >= args.threshold]
            tr = analyse(y)
            roar = np.array(onsets(tr.crowd_z, tr.times, threshold=4.0))
            for e in events[
                (events.game_id == g.game_id) & (events.half == half) & (events.event == "goal")
            ].itertuples():

                def first(xs, t=e.t):
                    d = xs[(xs >= t - 10) & (xs <= t + 30)] - t
                    return float(d.min()) if len(d) else np.nan

                rows.append(
                    dict(
                        game_id=g.game_id,
                        half=half,
                        t=e.t,
                        text_delay=first(fired),
                        roar_delay=first(roar),
                        asr_rtf=float(np.sum(r.compute_times) / (len(y) / 16000)),
                    )
                )
    df = pd.DataFrame(rows)
    report = dict(
        n_goals=len(df),
        text_recall=float(df.text_delay.notna().mean()),
        text_delay_p50=float(df.text_delay.median()),
        roar_recall=float(df.roar_delay.notna().mean()),
        roar_delay_p50=float(df.roar_delay.median()),
        asr_real_time_factor=float(df.asr_rtf.mean()),
    )
    (RESULTS / "real_audio.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
