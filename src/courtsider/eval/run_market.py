"""Market experiments on test-split matches: how much is commentary worth, and can the MM defend?

python -m courtsider.eval.run_market [--workers 10]

Sweeps (feed delay x courtsider threshold x radio delay) with and without an audio-aware market
maker. Writes results/market.json and figures.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from courtsider.paths import PROCESSED, RESULTS
from courtsider.sim.match import MatchSim, SimConfig
from courtsider.sim.signals import PricerFactory, final_score, signals_for_game

FIG = RESULTS / "figures"
_STATE: dict = {}


def _init() -> None:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    events = pd.read_parquet(PROCESSED / "events.parquet")
    results = pd.read_parquet(PROCESSED / "results.parquet")
    beliefs = pd.read_parquet(PROCESSED / "beliefs.parquet")
    _STATE.update(games=games, events=events, beliefs=beliefs, factory=PricerFactory(games, results, events))


def run_one(args: tuple[str, dict, float]) -> dict:
    game_id, cfg_dict, latency = args
    if not _STATE:
        _init()
    s = _STATE
    pricer = s["factory"](game_id)
    ev = s["events"][s["events"].game_id == game_id]
    sig = signals_for_game(s["beliefs"][s["beliefs"].game_id == game_id], latency)
    cfg = SimConfig(**cfg_dict)
    out = MatchSim(pricer, ev, sig, cfg, final_score=final_score(s["games"], game_id)).run()
    return dict(
        game_id=game_id,
        latency=latency,
        **cfg_dict,
        cs_pnl=out["pnl"].get("cs", 0.0),
        mm_pnl=out["pnl"].get("mm", 0.0),
        noise_pnl=out["pnl"].get("noise", 0.0),
        tv_pnl=out["pnl"].get("tv", 0.0),
        mm_markout_vs_cs=out["mm_markout"].get("cs", 0.0),
        mm_markout_vs_noise=out["mm_markout"].get("noise", 0.0),
        mm_markout_vs_tv=out["mm_markout"].get("tv", 0.0),
        mm_spread_vs_noise=out["mm_spread_capture"].get("noise", 0.0),
        cs_entries=out["cs_entries"],
        cs_volume=out["cs_volume"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-games", type=int, default=0)
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    _init()
    games, beliefs = _STATE["games"], _STATE["beliefs"]
    test_games = [
        g for g in games[(games.split == "test") & games.fd_home.notna()].game_id if g in set(beliefs.game_id)
    ]
    if args.max_games:
        test_games = test_games[: args.max_games]
    asr = json.loads((RESULTS / "asr.json").read_text()) if (RESULTS / "asr.json").exists() else {}
    text_eval = json.loads((RESULTS / "text_eval.json").read_text())
    # ASR latency + conservative Echoes timestamp margin (see run_text.timestamp_margin)
    pipeline = float(asr.get("pipeline_latency_s", 1.5)) + float(text_eval.get("timestamp_margin_s", 0.0))

    jobs = []
    # 1) feed delay x threshold (radio delay 0), plain market maker
    for fd, th in itertools.product((0.5, 1.0, 2.0, 3.0, 5.0), (0.3, 0.5, 0.7, 0.9)):
        jobs += [(g, asdict(SimConfig(feed_delay=fd, cs_threshold=th)), pipeline) for g in test_games]
    # 2) radio/stream delay on top of ASR latency
    for extra in (1.0, 3.0, 6.0):
        jobs += [(g, asdict(SimConfig(feed_delay=2.0, cs_threshold=0.5)), pipeline + extra) for g in test_games]
    # 3) audio-aware market maker (it listens to the same commentary detector)
    for fd, guard in itertools.product((1.0, 2.0, 3.0, 5.0), (0.5, 0.7)):
        jobs += [
            (
                g,
                asdict(SimConfig(feed_delay=fd, cs_threshold=0.5, mm_audio_guard=True, mm_guard_threshold=guard)),
                pipeline,
            )
            for g in test_games
        ]
    # 4) no courtsider baseline (threshold above 1 disables entries)
    for fd in (1.0, 2.0, 3.0, 5.0):
        jobs += [(g, asdict(SimConfig(feed_delay=fd, cs_threshold=1.1)), pipeline) for g in test_games]
    print(f"{len(test_games)} games, {len(jobs)} simulations, pipeline latency {pipeline:.2f}s")
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        rows = list(pool.map(run_one, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    df.to_parquet(PROCESSED / "market_runs.parquet")

    keys = ["feed_delay", "cs_threshold", "latency", "mm_audio_guard", "mm_guard_threshold"]

    def per_game(d: pd.DataFrame):
        return d.groupby(keys)

    agg_cols = [
        "cs_pnl",
        "mm_pnl",
        "mm_markout_vs_cs",
        "mm_markout_vs_noise",
        "mm_spread_vs_noise",
        "cs_entries",
        "cs_volume",
    ]
    summary = per_game(df)[agg_cols].mean().reset_index()
    sem = per_game(df)["cs_pnl"].sem().reset_index(drop=True)
    summary["cs_pnl_sem"] = sem.to_numpy()

    plain = summary[(~summary.mm_audio_guard) & (summary.latency == pipeline) & (summary.cs_threshold <= 1)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for th, d in plain.groupby("cs_threshold"):
        axes[0].errorbar(
            d.feed_delay, d.cs_pnl / 100, yerr=d.cs_pnl_sem / 100, marker="o", capsize=3, label=f"threshold {th}"
        )
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set(
        title="Courtsider P&L per match vs official-feed delay",
        xlabel="feed delay (s)",
        ylabel="P&L per match (contracts x $1)",
    )
    axes[0].legend()
    g05 = summary[(summary.cs_threshold == 0.5) & (summary.latency == pipeline)]
    for guard, d in g05.groupby(g05.mm_audio_guard.astype(str) + "_" + g05.mm_guard_threshold.astype(str)):
        lab = "plain MM" if guard.startswith("False") else f"audio-aware MM (guard {guard.split('_')[1]})"
        if guard.startswith("False") and not guard.endswith("0.6"):
            continue
        axes[1].plot(d.feed_delay, -d.mm_markout_vs_cs / 100, marker="o", label=lab)
    axes[1].set(
        title="MM losses to the courtsider (30 s markout)",
        xlabel="feed delay (s)",
        ylabel="adverse selection per match ($)",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG / "market.png", dpi=130)

    radio = summary[(summary.feed_delay == 2.0) & (summary.cs_threshold == 0.5) & (~summary.mm_audio_guard)]
    report = dict(
        n_games=len(test_games),
        pipeline_latency_s=pipeline,
        units="P&L in price points x contracts; 100 points = $1 per contract",
        summary=summary.round(3).to_dict(orient="records"),
        radio_delay=radio[["latency", "cs_pnl", "cs_pnl_sem", "cs_entries"]].round(3).to_dict(orient="records"),
        bootstrap_note="cs_pnl_sem is the standard error across matches",
    )
    (RESULTS / "market.json").write_text(json.dumps(report, indent=2))
    print(plain.pivot(index="feed_delay", columns="cs_threshold", values="cs_pnl").round(1).to_string())
    print(
        g05[["feed_delay", "mm_audio_guard", "mm_guard_threshold", "cs_pnl", "mm_markout_vs_cs"]]
        .round(1)
        .to_string(index=False)
    )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
