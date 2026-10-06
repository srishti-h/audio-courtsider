"""Audio-pipeline experiments on synthetic commentary audio (re-voiced Echoes text + synthetic crowd).

python -m courtsider.eval.run_audio [--excerpts 16] [--models tiny,base,small,turbo]

Measures, per Whisper size: streaming word latency, real-time factor, WER against the re-voiced
script, and player-name recall with/without team-sheet biasing. Also measures how early the
crowd-roar and excitement detectors fire relative to the transcript. All numbers are SYNTHETIC.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from courtsider.acoustic.features import analyse, onsets
from courtsider.asr.streaming import mine_names, name_recall, stream, wer
from courtsider.paths import AUDIO, PROCESSED, RESULTS
from courtsider.synth.commentary import Excerpt, render

FIG = RESULTS / "figures"


def pick_excerpts(games: pd.DataFrame, events: pd.DataFrame, n: int, seed: int = 0) -> list[Excerpt]:
    """Windows of 60 s before to 30 s after goals in test-split games, plus as many goal-free windows."""
    rng = np.random.default_rng(seed)
    test = games[(games.split == "test") & games.has_transcript].game_id
    goals = events[(events.event == "goal") & events.game_id.isin(test) & (events.t > 70)]
    goals = goals.sample(frac=1, random_state=seed).head(n // 2)
    out = [Excerpt(r.game_id, int(r.half), r.t - 60, r.t + 30) for r in goals.itertuples()]
    for gid in rng.choice(test.to_numpy(), size=n - len(out), replace=True):
        ev = events[(events.game_id == gid) & (events.half == 1)]
        gt = ev[ev.event == "goal"].t.to_numpy()
        for _ in range(20):
            s = float(rng.uniform(120, 2400))
            if not ((gt > s - 30) & (gt < s + 120)).any():
                out.append(Excerpt(gid, 1, s, s + 90))
                break
    return out


def end_to_end(clips: list[dict], streams: dict, teams: pd.DataFrame, threshold: float = 0.5) -> dict:
    from courtsider.asr.segmenter import segments_from_words
    from courtsider.paths import MODELS
    from courtsider.text.models import TransformerModel

    model = TransformerModel.load(MODELS / "text_extractor")
    seg_lat, delays, n_goals, false_alarms, minutes_no_goal = [], [], 0, 0, 0.0
    for ci, r in streams.items():
        c = clips[ci]
        gid = c["meta"]["game_id"]
        seg = segments_from_words(r.words, gid, 1)
        if seg.empty:
            continue
        seg_lat += list(seg.available - seg.end)
        home, away = teams.loc[gid, "home"], teams.loc[gid, "away"]
        texts = seg.text.tolist()
        inputs = [f"{home} vs {away} | {' '.join(texts[max(0, i - 2) : i])} || {t}" for i, t in enumerate(texts)]
        p_goal = model.predict(inputs)[0][:, 0]
        fired = seg.available.to_numpy()[p_goal >= threshold]
        goals = [e["t"] for e in c["meta"]["events"] if e["event"] == "goal"]
        for g in goals:
            n_goals += 1
            d = fired[(fired >= g - 10) & (fired <= g + 30)] - g
            if len(d):
                delays.append(float(d.min()))
        if not goals:
            false_alarms += int((np.diff(np.concatenate([[-99.0], fired])) > 20).sum())
            minutes_no_goal += c["meta"]["duration"] / 60
    return dict(
        segment_latency_p50=float(np.median(seg_lat)),
        segment_latency_p90=float(np.percentile(seg_lat, 90)),
        goal_recall=len(delays) / max(1, n_goals),
        goal_detection_delay_p50=float(np.median(delays)) if delays else float("nan"),
        false_alarms_per_min=false_alarms / max(1e-9, minutes_no_goal),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--excerpts", type=int, default=16)
    ap.add_argument("--models", default="tiny,base,small,turbo")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    games = pd.read_parquet(PROCESSED / "games.parquet")
    events = pd.read_parquet(PROCESSED / "events.parquet")
    segs = pd.read_parquet(PROCESSED / "segments.parquet")
    teams = games.set_index("game_id")

    excerpts = pick_excerpts(games, events, args.excerpts)
    clips = []
    for i, ex in enumerate(excerpts):
        path = AUDIO / f"synthetic_{i:02d}.wav"
        meta_path = path.with_suffix(".json")
        if not path.exists() or not meta_path.exists():
            meta = render(ex, segs, events, path, seed=i)
            meta_path.write_text(json.dumps(meta))
        meta = json.loads(meta_path.read_text())
        names = mine_names(
            segs[segs.game_id == ex.game_id].text.tolist(),
            (teams.loc[ex.game_id, "home"], teams.loc[ex.game_id, "away"]),
        )
        clips.append(dict(path=path, meta=meta, names=names))
    print(f"{len(clips)} synthetic clips, {sum(c['meta']['duration'] for c in clips) / 60:.1f} min")

    # ---- ASR sweep ----------------------------------------------------------------------
    asr_rows = []
    kept: dict[int, object] = {}  # turbo + name-bias stream results, reused for the end-to-end test
    for model in args.models.split(","):
        for bias in (False, True) if model in ("small", "turbo") else (False,):
            lat, wers, hits, tots, compute, audio_s = [], [], 0, 0, 0.0, 0.0
            for ci, c in enumerate(clips):
                y, _ = sf.read(c["path"])
                ref = " ".join(s["text"] for s in c["meta"]["script"])
                t0 = time.perf_counter()
                r = stream(y, model, vocabulary=c["names"] if bias else None)
                if model == "turbo" and bias:
                    kept[ci] = r
                compute += time.perf_counter() - t0
                audio_s += len(y) / 16000
                lat += list(r.latencies())
                wers.append(wer(ref, r.text()))
                h, t = name_recall(ref, r.text(), c["names"])
                hits, tots = hits + h, tots + t
            lat = np.array(lat)
            row = dict(
                model=model,
                name_bias=bias,
                wer=float(np.mean(wers)),
                name_recall=hits / max(1, tots),
                latency_p50=float(np.median(lat)),
                latency_p90=float(np.percentile(lat, 90)),
                latency_p99=float(np.percentile(lat, 99)),
                real_time_factor=compute / audio_s,
            )
            asr_rows.append(row)
            print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    asr = pd.DataFrame(asr_rows)

    # ---- acoustic lead over transcript ----------------------------------------------------
    ac_rows = []
    for c in clips:
        y, _ = sf.read(c["path"])
        tr = analyse(y)
        goal_t = [e["t"] for e in c["meta"]["events"] if e["event"] == "goal"]
        excited = [s["start"] for s in c["meta"]["script"] if s["excited"]]
        roar = onsets(tr.crowd_z, tr.times, threshold=4.0)
        exc = onsets(tr.excitement, tr.times, threshold=2.5)
        for g in goal_t:

            def first_after(xs, lo=-2.0, hi=20.0, g=g):
                d = [x - g for x in xs if lo <= x - g <= hi]
                return min(d) if d else np.nan

            ac_rows.append(
                dict(
                    goal_t=g,
                    roar_delay=first_after(roar),
                    excitement_delay=first_after(exc),
                    excited_speech_delay=first_after(excited, -10),
                )
            )
        # false alarms: onsets in goal-free clips
        if not goal_t:
            ac_rows.append(
                dict(goal_t=np.nan, roar_false=len(roar), excitement_false=len(exc), minutes=c["meta"]["duration"] / 60)
            )
    ac = pd.DataFrame(ac_rows)
    goals = ac[ac.goal_t.notna()]
    nogoal = ac[ac.goal_t.isna()]
    acoustic = dict(
        n_goal_clips=int(len(goals)),
        roar_recall=float(goals.roar_delay.notna().mean()),
        roar_delay_p50=float(goals.roar_delay.median()),
        excitement_recall=float(goals.excitement_delay.notna().mean()),
        excitement_delay_p50=float(goals.excitement_delay.median()),
        roar_false_alarms_per_min=float(nogoal.roar_false.sum() / max(1e-9, nogoal.minutes.sum())),
        excitement_false_alarms_per_min=float(nogoal.excitement_false.sum() / max(1e-9, nogoal.minutes.sum())),
    )

    # ---- end to end: audio -> streaming ASR -> segments -> text model -> goal detection ----
    e2e = end_to_end(clips, kept, teams) if kept else {}
    # latency added downstream to Echoes segment ends = measured (segment available - segment end)
    pipeline = e2e.get("segment_latency_p50", float(asr.latency_p50.iloc[-1]))
    report = dict(
        synthetic=True,
        n_clips=len(clips),
        asr=asr.round(4).to_dict(orient="records"),
        acoustic={k: round(v, 3) for k, v in acoustic.items()},
        end_to_end={k: round(v, 3) for k, v in e2e.items()},
        pipeline_latency_s=round(pipeline, 3),
        pipeline_model="turbo+name_bias",
    )
    (RESULTS / "asr.json").write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(6, 4))
    for bias, d in asr.groupby("name_bias"):
        ax.scatter(d.latency_p50, d.wer, s=60, label="name-biased" if bias else "plain")
        for r in d.itertuples():
            ax.annotate(r.model, (r.latency_p50, r.wer), textcoords="offset points", xytext=(5, 5))
    ax.set(xlabel="median word latency (s)", ylabel="WER vs script", title="Streaming Whisper on M5 Pro (synthetic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "asr_tradeoff.png", dpi=130)
    print(json.dumps(report["acoustic"], indent=1))


if __name__ == "__main__":
    main()
