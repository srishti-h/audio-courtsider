"""Text-pipeline experiments on real commentary (SoccerNet-Echoes), test split.

python -m courtsider.eval.run_text [--latency 1.5]

Protocol (no test-set peeking):
  1. validation games are split in halves A/B; calibration + fusion models are fit on A,
     and each method's decision threshold is chosen by event-level goal F1 on B;
  2. the method with the best B score is selected, everything is refit on all of validation,
     and every number is reported once on the test split.
Outputs results/text_eval.json, figures, and data/processed/beliefs.parquet (the calibrated,
fused per-segment beliefs used by the market simulation and dashboard).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from courtsider.data.taxonomy import EVENT_TYPES
from courtsider.eval.detection import detections, evaluate, match
from courtsider.fusion.models import P_COLS, Calibrator, GRUFusion, HMMFilter
from courtsider.paths import PROCESSED, RESULTS
from courtsider.text.dataset import label_segments

FIG = RESULTS / "figures"
THRESHOLDS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
REPORT_EVENTS = ("goal", "penalty", "red_card", "corner", "yellow_card", "offside", "substitution", "shot")


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(
        sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    )


def timestamp_margin(segments: pd.DataFrame, events: pd.DataFrame, games: pd.DataFrame, q: float = 10) -> float:
    """Conservative shift for Echoes timestamps, estimated on TRAIN games only.

    Offline Whisper timestamps can place a goal call before the labelled goal. Shifting every
    segment later by the q-th percentile of that early offset means the simulated courtsider is
    almost never credited with knowing a goal before it happened.
    """
    from courtsider.text.models import _KW

    train = set(games[games.split == "train"].game_id)
    seg = segments[segments.game_id.isin(train)]
    hits = seg[seg.text.str.contains(_KW["goal"])]
    offs = []
    for gid, ev in events[(events.event == "goal") & events.game_id.isin(train)].groupby("game_id"):
        hs = hits[hits.game_id == gid].g_end.to_numpy()
        for t in ev.g.to_numpy():
            d = hs - t
            d = d[(d > -30) & (d < 60)]
            if len(d):
                offs.append(d[np.argmin(np.abs(d))])
    return float(max(0.0, -np.percentile(offs, q)))


def asr_latency_default() -> float:
    path = RESULTS / "asr.json"
    if path.exists():
        return float(json.loads(path.read_text()).get("pipeline_latency_s", 1.5))
    return 1.5


def _methods(fit_df: pd.DataFrame, y_fit: np.ndarray, baselines: dict[str, pd.DataFrame]) -> dict:
    """Fit every candidate on `fit_df` (a validation subset). Returns name -> transform(df)."""
    out = {}
    for name, df in baselines.items():
        if name == "keywords":  # binary rule output: evaluate as-is (a match is a detection)
            out[name] = lambda d, src=df: src.loc[d.index, P_COLS].to_numpy()
            continue
        cal_b = Calibrator().fit(df.loc[fit_df.index, P_COLS].to_numpy(), y_fit)
        out[name] = lambda d, cal_b=cal_b, src=df: cal_b.transform(src.loc[d.index, P_COLS].to_numpy())
    cal = Calibrator().fit(fit_df[P_COLS].to_numpy(), y_fit)

    def _cal(d: pd.DataFrame) -> pd.DataFrame:
        c = d.copy()
        c[P_COLS] = cal.transform(d[P_COLS].to_numpy())
        return c

    cal_fit = _cal(fit_df)
    hmm = HMMFilter().fit(cal_fit, y_fit)
    gru = GRUFusion().fit(cal_fit, y_fit)
    out["distilroberta"] = lambda d: cal.transform(d[P_COLS].to_numpy())
    out["distilroberta+hmm"] = lambda d: hmm.transform(_cal(d))
    out["distilroberta+gru"] = lambda d: gru.transform(_cal(d))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", type=float, default=None, help="ASR+processing latency added to segment end (s)")
    args = ap.parse_args()
    latency = args.latency if args.latency is not None else asr_latency_default()
    FIG.mkdir(exist_ok=True)
    events = pd.read_parquet(PROCESSED / "events.parquet")
    games = pd.read_parquet(PROCESSED / "games.parquet")
    margin = timestamp_margin(pd.read_parquet(PROCESSED / "segments.parquet"), events, games)
    lat = latency + margin
    print(f"latency {latency:.2f}s + conservative timestamp margin {margin:.2f}s (train p10 of early goal calls)")

    base = label_segments(pd.read_parquet(PROCESSED / "text_preds_distilroberta.parquet"), events)
    others = {n: pd.read_parquet(PROCESSED / f"text_preds_{n}.parquet") for n in ("keywords", "tfidf_lr")}
    for d in others.values():
        assert (d.game_id.to_numpy() == base.game_id.to_numpy()).all()
        d.index = base.index
    valid, test = base[base.split == "valid"], base[base.split == "test"]
    yv, yt = valid[EVENT_TYPES].to_numpy(), test[EVENT_TYPES].to_numpy()

    # ---- 1. selection on validation halves ---------------------------------------------
    vgames = valid.game_id.unique()
    half_a = valid.game_id.isin(vgames[::2]).to_numpy()
    va, vb = valid[half_a], valid[~half_a]
    methods = _methods(va, yv[half_a], others)
    selection = {}
    for name, fn in methods.items():
        p = fn(vb)
        scores = []
        for th in THRESHOLDS:
            r = evaluate(vb, p, events, "goal", th, lat)
            f1 = 2 * r["precision"] * r["recall"] / max(1e-9, r["precision"] + r["recall"])
            scores.append((f1, th))
        f1, th = max(scores)
        selection[name] = dict(goal_f1_valid=round(f1, 4), threshold=th)
    chosen = max(selection, key=lambda n: selection[n]["goal_f1_valid"])
    print("selection on validation half B:", json.dumps(selection), "->", chosen)

    # ---- 2. refit on all validation, report on test ------------------------------------
    methods = _methods(valid, yv, others)
    test_probs = {name: fn(test) for name, fn in methods.items()}
    seg_ap = {
        name: {e: round(float(average_precision_score(yt[:, k], p[:, k])), 4) for k, e in enumerate(EVENT_TYPES)}
        for name, p in test_probs.items()
    }
    detection = []
    for name, p in test_probs.items():
        th = selection[name]["threshold"]
        for e in REPORT_EVENTS:
            detection.append(dict(method=name, **evaluate(test, p, events, e, th, lat)))
    det = pd.DataFrame(detection)
    best = test_probs[chosen]
    th_best = selection[chosen]["threshold"]
    sweep = pd.DataFrame([evaluate(test, best, events, "goal", th, lat) for th in THRESHOLDS])
    calib = {
        "uncalibrated": round(ece(test.p_goal.to_numpy(), yt[:, 0]), 4),
        chosen: round(ece(best[:, 0], yt[:, 0]), 4),
    }

    # team attribution where it matters: confident goal calls
    team_acc = None
    if "team_home" in test:
        conf = (best[:, 0] >= th_best) & (test.team != "unknown").to_numpy()
        if conf.any():
            pred = np.where(test.team_home.to_numpy() >= test.team_away.to_numpy(), "home", "away")
            team_acc = dict(n=int(conf.sum()), accuracy=round(float((pred[conf] == test.team[conf]).mean()), 4))

    # ---- beliefs table for the simulator / dashboard ------------------------------------
    beliefs = test[["game_id", "half", "g_start", "g_end", "text"]].copy()
    for k, e in enumerate(EVENT_TYPES):
        beliefs[f"p_{e}"] = best[:, k]
    if "team_home" in test:
        beliefs[["team_home", "team_away"]] = test[["team_home", "team_away"]].to_numpy()
    beliefs.to_parquet(PROCESSED / "beliefs.parquet")

    # ---- figures ------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ("keywords", "tfidf_lr", chosen):
        sw = pd.DataFrame([evaluate(test, test_probs[name], events, "goal", th, lat) for th in THRESHOLDS])
        axes[0].plot(sw.recall, sw.precision, "o-", label=name)
    axes[0].set(title="Goal detection, event level (test)", xlabel="recall", ylabel="precision")
    axes[0].legend()
    pairs, _ = match(detections(test, best, "goal", th_best, lat), events[events.event == "goal"])
    if len(pairs):
        axes[1].hist(pairs.t_det - pairs.t_true, bins=np.arange(-10, 31, 1.5), color="#f59e0b")
        for fd, ls in ((2, "--"), (5, ":")):
            axes[1].axvline(fd, color="k", ls=ls, lw=1, label=f"feed delay {fd}s")
        axes[1].set(
            title=f"When the commentary detector fires (goals, threshold {th_best})",
            xlabel="seconds after the goal (incl. ASR latency + margin)",
            ylabel="goals",
        )
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "goal_detection.png", dpi=130)

    report = dict(
        latency_s=latency,
        timestamp_margin_s=round(margin, 3),
        fusion_selected=chosen,
        selection_valid_half=selection,
        detection_test=det.round(4).to_dict(orient="records"),
        goal_threshold_sweep_test=sweep.round(4).to_dict(orient="records"),
        segment_avg_precision_test=seg_ap,
        goal_ece_test=calib,
        team_attribution_on_confident_goal_calls=team_acc,
        n_test_games=int(test.game_id.nunique()),
        n_test_segments=len(test),
    )
    (RESULTS / "text_eval.json").write_text(json.dumps(report, indent=2))
    cols = ["event", "n_true", "precision", "recall", "delay_p50", "beats_feed_2s", "beats_feed_5s"]
    print(det[det.event == "goal"][["method", "threshold", "precision", "recall", "delay_p50"]].round(3).to_string())
    print(det[det.method == chosen][cols].round(3).to_string(index=False))
    print("team attribution:", team_acc)


if __name__ == "__main__":
    main()
