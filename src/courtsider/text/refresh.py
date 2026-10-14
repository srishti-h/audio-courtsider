"""Retrain the text event model with recent data: python -m courtsider.text.refresh [--epochs 2]

Candidate = the same DistilRoBERTa recipe as text.train, trained on SoccerNet commentary (2014-17)
plus ESPN's written 2025/26 commentary (data.espn) plus labelled live-radio sessions from the train
split (data.radio). It replaces models/text_extractor only if it is no worse on the SoccerNet
validation split, the domain every backtest runs on. Both models are reported on the SoccerNet test
split and on the held-out 2026 radio sessions -> results/text_recent.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil

import numpy as np
import pandas as pd
import torch

from courtsider.data.taxonomy import EVENT_TYPES, PRICE_MOVING
from courtsider.paths import MODELS, PROCESSED, RESULTS
from courtsider.text.dataset import build_inputs
from courtsider.text.models import TransformerModel
from courtsider.text.train import load_labelled, metrics, save_preds

TOLERANCE = 0.005  # mean validation AP on price-moving events the candidate may lose and still ship


def espn_rows() -> pd.DataFrame:
    path = PROCESSED / "espn_commentary.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["input"] = build_inputs(df, df[["game_id", "home", "away"]].drop_duplicates())
    return df.assign(split="train", source="espn")


def radio_rows() -> pd.DataFrame:
    path = PROCESSED / "radio_segments.parquet"
    return pd.read_parquet(path).assign(source="radio") if path.exists() else pd.DataFrame()


def price_ap(m: dict) -> float:
    return float(np.mean([m[e]["avg_precision"] for e in PRICE_MOVING if e in m]))


def radio_event_recall(df: pd.DataFrame, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Event level on radio: share of ESPN goals with a segment at P(goal) >= threshold in the label window,
    and alarms per minute outside every window."""
    ev = pd.read_parquet(PROCESSED / "radio_events.parquet")
    ev = ev[(ev.event == "goal") & (ev.split == "test")]
    fire = p[:, EVENT_TYPES.index("goal")] >= threshold
    hit = sum(
        bool((fire & (df.game_id == g.game_id) & (df.g_end >= g.g - 6) & (df.g_start <= g.g + 8)).any())
        for g in ev.itertuples()
    )
    minutes = float(df.groupby("game_id").g_end.max().sum() / 60)
    false = int((fire & (df.goal == 0)).sum())
    return dict(
        goals=len(ev),
        recall=round(hit / max(1, len(ev)), 3),
        false_alarms_per_min=round(false / max(minutes, 1e-9), 3),
        minutes=round(minutes, 1),
    )


def evaluate(model: TransformerModel, sn: pd.DataFrame, radio: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    p, team = model.predict(sn.input.tolist())
    y = sn[EVENT_TYPES].to_numpy()
    va, te = (sn.split == "valid").to_numpy(), (sn.split == "test").to_numpy()
    out = dict(valid=metrics(y[va], p[va]), test=metrics(y[te], p[te]))
    out["valid_price_moving_ap"] = round(price_ap(out["valid"]), 4)
    out["test_price_moving_ap"] = round(price_ap(out["test"]), 4)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()  # inference over ~285k variable-length inputs leaves a large cache behind
    rt = radio[radio.split == "test"] if len(radio) else radio
    if len(rt):
        pr, _ = model.predict(rt.input.tolist())
        out["radio_2026_test"] = dict(
            segments=metrics(rt[EVENT_TYPES].to_numpy(), pr), events=radio_event_recall(rt.reset_index(drop=True), pr)
        )
    return out, p, team


def eval_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_labelled()
    return df[df.split.isin(["valid", "test"])].reset_index(drop=True), radio_rows()


def training_set(neg_frac: float, rare_oversample: int) -> pd.DataFrame:
    df, espn, radio = load_labelled(), espn_rows(), radio_rows()
    cols = ["input", "team", *EVENT_TYPES]
    parts = [df[df.split == "train"].assign(source="soccernet")[cols + ["source"]]]
    if len(espn):
        parts.append(espn[cols + ["source"]])
    if len(radio):
        parts.append(radio[radio.split == "train"][cols + ["source"]])
    train = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(0)  # same sampling recipe as text.train
    keep = (
        train[EVENT_TYPES].to_numpy().any(1)
        | (train.team != "unknown").to_numpy()
        | (rng.random(len(train)) < neg_frac)
    )
    tr = train[keep]
    rare = (tr.penalty == 1) | (tr.red_card == 1)
    return pd.concat([tr] + [tr[rare]] * rare_oversample).sample(frac=1, random_state=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--neg-frac", type=float, default=0.4)
    ap.add_argument("--rare-oversample", type=int, default=6)
    args = ap.parse_args()

    # the current model's scores only change when its weights or the radio test set do
    cache = PROCESSED / "text_current_eval.json"
    key = [(MODELS / "text_extractor" / "weights.pt").stat().st_mtime, len(radio_rows())]
    cached = json.loads(cache.read_text()) if cache.exists() else {}
    if cached.get("key") == key:
        report = dict(current=cached["report"])
    else:
        report = dict(current=evaluate(TransformerModel.load(MODELS / "text_extractor"), *eval_data())[0])
        cache.write_text(json.dumps(dict(key=key, report=report["current"])))

    # build the training set in its own scope and keep only what fit() needs: unified memory is shared
    # with the GPU, and holding every table through training pushed a busy Mac into swap (10x slower steps)
    tr = training_set(args.neg_frac, args.rare_oversample)
    report["train_rows"] = tr.source.value_counts().to_dict()
    texts, y, team = tr.input.tolist(), tr[EVENT_TYPES].to_numpy(), tr.team.to_numpy()
    del tr
    gc.collect()
    print(f"candidate train rows: {report['train_rows']}", flush=True)

    cand = TransformerModel().fit(texts, y, team, epochs=args.epochs)
    del texts, y, team
    gc.collect()
    cand.save(MODELS / "text_extractor_recent")
    sn, radio = eval_data()
    report["candidate"], p, team = evaluate(cand, sn, radio)
    promote = report["candidate"]["valid_price_moving_ap"] >= report["current"]["valid_price_moving_ap"] - TOLERANCE
    report["promoted"] = bool(promote)
    if promote:  # the live pipeline, benchmarks and backtests all load models/text_extractor
        shutil.rmtree(MODELS / "text_extractor_previous", ignore_errors=True)
        shutil.copytree(MODELS / "text_extractor", MODELS / "text_extractor_previous")
        shutil.rmtree(MODELS / "text_extractor")
        shutil.copytree(MODELS / "text_extractor_recent", MODELS / "text_extractor")
        save_preds(sn, p, team, "distilroberta")
    (RESULTS / "text_recent.json").write_text(json.dumps(report, indent=2))
    summary = {
        k: {m: v.get(m) for m in ("valid_price_moving_ap", "test_price_moving_ap", "radio_2026_test")}
        for k, v in report.items()
        if isinstance(v, dict) and "valid" in v
    }
    print(json.dumps(dict(summary, promoted=promote), indent=1))


if __name__ == "__main__":
    main()
