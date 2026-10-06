"""Train/evaluate text event extractors: `python -m courtsider.text.train [--skip-transformer]`.

Classifiers train on the SoccerNet train split. Valid-split predictions feed calibration and
the fusion models; every reported number is on the held-out test split.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from courtsider.data.taxonomy import EVENT_TYPES
from courtsider.paths import MODELS, PROCESSED, RESULTS
from courtsider.text.dataset import TEAM_CLASSES, build_inputs, label_segments
from courtsider.text.models import KeywordModel, TfidfModel, TransformerModel


def load_labelled() -> pd.DataFrame:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    events = pd.read_parquet(PROCESSED / "events.parquet")
    segs = pd.read_parquet(PROCESSED / "segments.parquet")
    segs = segs.merge(games[["game_id", "split"]], on="game_id")
    df = label_segments(segs, events)
    df["input"] = build_inputs(df, games)
    return df


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    for k, e in enumerate(EVENT_TYPES):
        if y[:, k].sum() == 0:
            continue
        out[e] = {
            "avg_precision": round(float(average_precision_score(y[:, k], p[:, k])), 4),
            "roc_auc": round(float(roc_auc_score(y[:, k], p[:, k])), 4),
            "positives": int(y[:, k].sum()),
        }
    return out


def save_preds(df: pd.DataFrame, probs: np.ndarray, team: np.ndarray | None, name: str) -> None:
    out = df[["game_id", "split", "half", "seg", "start", "end", "g_start", "g_end", "text"]].copy()
    for k, e in enumerate(EVENT_TYPES):
        out[f"p_{e}"] = probs[:, k].astype(np.float32)
    if team is not None:
        for k, t in enumerate(TEAM_CLASSES):
            out[f"team_{t}"] = team[:, k].astype(np.float32)
    out.to_parquet(PROCESSED / f"text_preds_{name}.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-transformer", action="store_true")
    ap.add_argument("--neg-frac", type=float, default=0.4, help="fraction of all-negative train segments kept")
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    df = load_labelled()
    y_all = df[EVENT_TYPES].to_numpy()
    train = df[df.split == "train"]
    evalset = df[df.split.isin(["valid", "test"])].reset_index(drop=True)
    y_eval = evalset[EVENT_TYPES].to_numpy()
    test_mask = (evalset.split == "test").to_numpy()
    print(f"train segments={len(train)} eval segments={len(evalset)} positive rate={y_all.mean(0).round(4)}")

    report = {}
    kw = KeywordModel()
    p = kw.predict_proba(evalset.input.tolist())
    report["keywords"] = metrics(y_eval[test_mask], p[test_mask])
    save_preds(evalset, p, None, "keywords")

    tf = TfidfModel().fit(train.input.tolist(), train[EVENT_TYPES].to_numpy())
    p = tf.predict_proba(evalset.input.tolist())
    report["tfidf_lr"] = metrics(y_eval[test_mask], p[test_mask])
    save_preds(evalset, p, None, "tfidf_lr")
    print(json.dumps({k: {e: v["avg_precision"] for e, v in r.items()} for k, r in report.items()}, indent=1))

    if not args.skip_transformer:
        rng = np.random.default_rng(0)
        any_pos = train[EVENT_TYPES].to_numpy().any(1) | (train.team != "unknown").to_numpy()
        keep = any_pos | (rng.random(len(train)) < args.neg_frac)
        tr = train[keep]
        print(f"transformer train segments={len(tr)}")
        model = TransformerModel().fit(
            tr.input.tolist(), tr[EVENT_TYPES].to_numpy(), tr.team.to_numpy(), epochs=args.epochs
        )
        model.save(MODELS / "text_extractor")
        p, team = model.predict(evalset.input.tolist())
        report["distilroberta"] = metrics(y_eval[test_mask], p[test_mask])
        # team attribution accuracy on test segments that carry a team label
        has = (evalset.team != "unknown").to_numpy() & test_mask
        pred_team = np.array(TEAM_CLASSES)[team[has, :2].argmax(1)]
        report["distilroberta"]["team_accuracy"] = round(float((pred_team == evalset.team[has]).mean()), 4)
        save_preds(evalset, p, team, "distilroberta")

    (RESULTS / "text_metrics.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {k: {e: v["avg_precision"] if isinstance(v, dict) else v for e, v in r.items()} for k, r in report.items()},
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
