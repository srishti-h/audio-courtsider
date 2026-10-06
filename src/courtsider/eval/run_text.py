"""Text-pipeline experiments on real commentary (SoccerNet-Echoes), test split.

python -m courtsider.eval.run_text [--latency 1.5]

Outputs results/text_eval.json, figures, and data/processed/beliefs.parquet (calibrated, fused
per-segment beliefs used by the market simulation and the dashboard).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from courtsider.data.taxonomy import EVENT_TYPES
from courtsider.eval.detection import evaluate
from courtsider.fusion.models import P_COLS, Calibrator, GRUFusion, HMMFilter
from courtsider.paths import PROCESSED, RESULTS
from courtsider.text.dataset import label_segments

FIG = RESULTS / "figures"


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
        d = json.loads(path.read_text())
        return float(d.get("pipeline_latency_s", 1.5))
    return 1.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", type=float, default=None, help="ASR+processing latency added to segment end (s)")
    args = ap.parse_args()
    latency = args.latency if args.latency is not None else asr_latency_default()
    FIG.mkdir(exist_ok=True)
    events = pd.read_parquet(PROCESSED / "events.parquet")
    margin = timestamp_margin(
        pd.read_parquet(PROCESSED / "segments.parquet"), events, pd.read_parquet(PROCESSED / "games.parquet")
    )
    print(f"conservative timestamp margin (train, p10 of early goal calls): {margin:.2f}s")

    preds = {}
    for name in ("keywords", "tfidf_lr", "distilroberta"):
        path = PROCESSED / f"text_preds_{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            preds[name] = label_segments(df, events)
    base = preds["distilroberta"] if "distilroberta" in preds else preds["tfidf_lr"]
    valid, test = base[base.split == "valid"], base[base.split == "test"]
    yv, yt = valid[EVENT_TYPES].to_numpy(), test[EVENT_TYPES].to_numpy()

    # ---- segment-level comparison ----------------------------------------------------
    seg_report: dict = {}
    for name, df in preds.items():
        p = df[df.split == "test"][P_COLS].to_numpy()
        seg_report[name] = {
            e: round(float(average_precision_score(yt[:, k], p[:, k])), 4) for k, e in enumerate(EVENT_TYPES)
        }

    cal = Calibrator().fit(valid[P_COLS].to_numpy(), yv)
    raw_t = cal.transform(test[P_COLS].to_numpy())
    valid_cal = valid.copy()
    valid_cal[P_COLS] = cal.transform(valid[P_COLS].to_numpy())
    test_cal = test.copy()
    test_cal[P_COLS] = raw_t
    hmm = HMMFilter().fit(valid_cal, yv)
    hmm_t = hmm.transform(test_cal)
    gru = GRUFusion().fit(valid_cal, yv)
    gru_t = gru.transform(test_cal)
    fused = {"calibrated": raw_t, "hmm": hmm_t, "gru": gru_t}
    for name, p in fused.items():
        seg_report[f"distilroberta+{name}"] = {
            e: round(float(average_precision_score(yt[:, k], p[:, k])), 4) for k, e in enumerate(EVENT_TYPES)
        }
    calib = {
        name: {e: round(ece(p[:, k], yt[:, k]), 4) for k, e in enumerate(EVENT_TYPES)}
        for name, p in {"uncalibrated": test[P_COLS].to_numpy(), **fused}.items()
    }

    # pick the fusion by validation AP on the price-moving events (no peeking at test)
    nv = len(valid_cal)
    half = valid_cal.game_id.isin(valid_cal.game_id.unique()[::2]).to_numpy()
    sel = {}
    for name, model in (("hmm", HMMFilter()), ("gru", GRUFusion())):
        m = model.fit(valid_cal[half], yv[half])
        p = m.transform(valid_cal[~half])
        sel[name] = float(
            np.mean([average_precision_score(yv[~half, k], p[:, k]) for k in (0, 4, 5) if yv[~half, k].any()])
        )
    sel["calibrated"] = float(
        np.mean(
            [
                average_precision_score(yv[~half, k], valid_cal[P_COLS].to_numpy()[~half, k])
                for k in (0, 4, 5)
                if yv[~half, k].any()
            ]
        )
    )
    chosen = max(sel, key=sel.get)
    best = fused[chosen]
    print(f"fusion selection on validation halves: {sel} -> {chosen} (n_valid={nv})")

    # ---- event-level detection on test ------------------------------------------------
    det = []
    for e in EVENT_TYPES:
        for th in (0.3, 0.5, 0.7, 0.9):
            det.append(evaluate(test, best, events, e, th, latency + margin))
    det_df = pd.DataFrame(det)

    # ---- beliefs table for the simulator ----------------------------------------------
    beliefs = test[["game_id", "half", "g_start", "g_end", "text"]].copy()
    for k, e in enumerate(EVENT_TYPES):
        beliefs[f"p_{e}"] = best[:, k]
    if "team_home" in test:
        beliefs[["team_home", "team_away"]] = test[["team_home", "team_away"]].to_numpy()
    beliefs.to_parquet(PROCESSED / "beliefs.parquet")

    # ---- figures ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, e in zip(axes, ("goal", "penalty", "red_card"), strict=True):
        k = EVENT_TYPES.index(e)
        for name, p in [
            ("keywords", preds["keywords"][preds["keywords"].split == "test"][P_COLS].to_numpy()),
            ("tf-idf + LR", preds["tfidf_lr"][preds["tfidf_lr"].split == "test"][P_COLS].to_numpy()),
            (f"DistilRoBERTa + {chosen}", best),
        ]:
            pr, rc, _ = precision_recall_curve(yt[:, k], p[:, k])
            ax.plot(rc, pr, label=f"{name} (AP {average_precision_score(yt[:, k], p[:, k]):.2f})")
        ax.set(title=f"{e}: segment-level PR (test)", xlabel="recall", ylabel="precision")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "pr_curves.png", dpi=130)

    fig, ax = plt.subplots(figsize=(5, 5))
    k = 0
    for name, p in {"uncalibrated": test[P_COLS].to_numpy(), chosen: best}.items():
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p[:, k], edges) - 1, 0, 9)
        xs = [p[idx == b, k].mean() for b in range(10) if (idx == b).sum() > 20]
        ys = [yt[idx == b, k].mean() for b in range(10) if (idx == b).sum() > 20]
        ax.plot(xs, ys, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(title="goal: reliability (test)", xlabel="predicted P(goal)", ylabel="observed frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "reliability_goal.png", dpi=130)

    report = dict(
        latency_s=latency,
        timestamp_margin_s=round(margin, 3),
        fusion_selected=chosen,
        fusion_selection_valid_ap=sel,
        segment_avg_precision=seg_report,
        segment_ece=calib,
        detection=det_df.round(4).to_dict(orient="records"),
        n_test_games=int(test.game_id.nunique()),
        n_test_segments=len(test),
    )
    (RESULTS / "text_eval.json").write_text(json.dumps(report, indent=2))
    show = det_df[det_df.threshold == 0.5][
        ["event", "n_true", "precision", "recall", "delay_p50", "beats_feed_2s", "beats_feed_5s"]
    ]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
