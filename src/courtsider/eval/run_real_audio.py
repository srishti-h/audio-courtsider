"""Real broadcast audio: python -m courtsider.eval.run_real_audio [--games 5] [--download]

Access: request it on https://huggingface.co/datasets/SoccerNet/SoccerNet_raw_HQ (gated; the old
NDA password form is retired), then `hf auth login`. `--download` fetches only the 224p halves of
the selected test matches (~180 MB each; the full 224p branch is 182 GB).

Matches: test-split games with native English commentary (Echoes whisper_v3 English), most goals
first. For each half: extract 16 kHz audio, stream it through Whisper (the operating point chosen
in run_audio, with match-name biasing), score segments with the text model, run the crowd-roar
detector, and time every signal against the labelled goals. Unlike the Echoes-based results,
these timings are measured on real broadcast audio end to end.
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

REPO = "SoccerNet/SoccerNet_raw_HQ"
EARLY, LATE = 10.0, 30.0  # a signal counts for a goal if it lands in [t - EARLY, t + LATE]


def pick_games(n: int) -> pd.DataFrame:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    segs = pd.read_parquet(PROCESSED / "segments.parquet", columns=["game_id", "source"])
    native = set(segs[segs.source == "whisper_v3"].game_id)
    t = games[(games.split == "test") & games.game_id.isin(native)].copy()
    t["goals"] = t.home_goals + t.away_goals
    return t.sort_values(["goals", "game_id"], ascending=[False, True]).head(n)


def download(game_id: str) -> None:
    from huggingface_hub import hf_hub_download

    for half in (1, 2):
        dest = SOCCERNET_DIR / game_id / f"{half}_224p.mkv"
        if not dest.exists():
            hf_hub_download(
                REPO, f"{game_id}/{half}_224p.mkv", repo_type="dataset", revision="videos-224p", local_dir=SOCCERNET_DIR
            )


def extract_audio(video: Path, out: Path) -> Path:
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-ac", "1", "-ar", "16000", str(out)], check=True
        )
    return out


def first_in_window(times: np.ndarray, t: float) -> float:
    d = times[(times >= t - EARLY) & (times <= t + LATE)] - t
    return float(d.min()) if len(d) else np.nan


def false_alarms(times: np.ndarray, goals: np.ndarray) -> int:
    return int(sum(not ((goals - EARLY <= x) & (x <= goals + LATE)).any() for x in times))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5, help="raw text-model P(goal) to fire")
    args = ap.parse_args()
    from courtsider.text.models import TransformerModel

    asr_cfg = json.loads((RESULTS / "asr.json").read_text())
    whisper = asr_cfg["pipeline_model"].split("+")[0]
    events = pd.read_parquet(PROCESSED / "events.parquet")
    segs = pd.read_parquet(PROCESSED / "segments.parquet")
    model = TransformerModel.load(MODELS / "text_extractor")
    chosen = pick_games(args.games)
    goal_rows, half_rows = [], []
    for g in chosen.itertuples():
        if args.download:
            download(g.game_id)
        names = mine_names(segs[segs.game_id == g.game_id].text.tolist(), (g.home, g.away))
        for half in (1, 2):
            video = SOCCERNET_DIR / g.game_id / f"{half}_224p.mkv"
            if not video.exists():
                print(f"missing {video}; run with --download")
                continue
            wav = extract_audio(video, AUDIO / f"real_{g.Index}_{half}.wav")
            y, _ = sf.read(wav)
            minutes = len(y) / 16000 / 60
            r = stream(y, whisper, vocabulary=names)
            seg = segments_from_words(r.words, g.game_id, half)
            texts = seg.text.tolist()
            inputs = [
                f"{g.home} vs {g.away} | {' '.join(texts[max(0, i - 2) : i])} || {t}" for i, t in enumerate(texts)
            ]
            p_goal = model.predict(inputs)[0][:, 0] if inputs else np.array([])
            fired = seg.available.to_numpy()[p_goal >= args.threshold]
            # one detection per 20 s, as in the text evaluation
            fired = fired[np.concatenate([[True], np.diff(fired) > 20])] if len(fired) else fired
            tr = analyse(y)
            roar = np.array(onsets(tr.crowd_z, tr.times, threshold=4.0))
            goals = events[
                (events.game_id == g.game_id) & (events.half == half) & (events.event == "goal")
            ].t.to_numpy()
            for t in goals:
                goal_rows.append(
                    dict(
                        game_id=g.game_id,
                        half=half,
                        t=t,
                        text_delay=first_in_window(fired, t),
                        roar_delay=first_in_window(roar, t),
                    )
                )
            lat = r.latencies()
            half_rows.append(
                dict(
                    game_id=g.game_id,
                    half=half,
                    minutes=minutes,
                    goals=len(goals),
                    text_false_alarms=false_alarms(fired, goals),
                    roar_false_alarms=false_alarms(roar, goals),
                    word_latency_p50=float(np.median(lat)) if len(lat) else np.nan,
                    segment_latency_p50=float(np.median(seg.available - seg.end)) if len(seg) else np.nan,
                    rtf=float(np.sum(r.compute_times) / (len(y) / 16000)),
                )
            )
            print(half_rows[-1], flush=True)
    gd, hd = pd.DataFrame(goal_rows), pd.DataFrame(half_rows)
    mins = hd.minutes.sum()
    report = dict(
        real_audio=True,
        whisper=f"{whisper}+name_bias",
        n_games=int(hd.game_id.nunique()),
        n_goals=len(gd),
        audio_minutes=round(float(mins), 1),
        text_goal_recall=round(float(gd.text_delay.notna().mean()), 3),
        text_goal_delay_p50=round(float(gd.text_delay.median()), 2),
        text_false_alarms_per_min=round(float(hd.text_false_alarms.sum() / mins), 3),
        roar_goal_recall=round(float(gd.roar_delay.notna().mean()), 3),
        roar_goal_delay_p50=round(float(gd.roar_delay.median()), 2),
        roar_false_alarms_per_min=round(float(hd.roar_false_alarms.sum() / mins), 3),
        word_latency_p50=round(float(hd.word_latency_p50.median()), 2),
        segment_latency_p50=round(float(hd.segment_latency_p50.median()), 2),
        real_time_factor=round(float(hd.rtf.mean()), 3),
        games=sorted(hd.game_id.unique()),
    )
    (RESULTS / "real_audio.json").write_text(json.dumps(report, indent=2))
    gd.to_csv(RESULTS / "real_audio_goals.csv", index=False)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
