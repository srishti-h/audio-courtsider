"""Which signal earns more: commentary text, the learned crowd/audio detector, or both?

python -m courtsider.eval.run_market_audio      (after acoustic.goal_model has written its tracks)

Same simulator and test matches as run_market, with the courtsider listening to (a) text only,
(b) the audio goal detector only, (c) both. Paired per-match differences, written to
results/market_audio.json and a figure.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from courtsider.paths import PROCESSED, RESULTS
from courtsider.sim.match import MatchSim, SimConfig
from courtsider.sim.signals import PricerFactory, audio_signals_for_game, final_score, signals_for_game

FEED_DELAYS = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)
SOURCES = ("text", "audio", "text+audio")
_S: dict = {}


def _init() -> None:
    games = pd.read_parquet(PROCESSED / "games.parquet")
    events = pd.read_parquet(PROCESSED / "events.parquet")
    results = pd.read_parquet(PROCESSED / "results.parquet")
    _S.update(
        games=games,
        events=events,
        beliefs=pd.read_parquet(PROCESSED / "beliefs.parquet"),
        tracks=pd.read_parquet(PROCESSED / "audio_goal_tracks.parquet"),
        factory=PricerFactory(games, results, events),
        text_eval=json.loads((RESULTS / "text_eval.json").read_text()),
        audio=json.loads((RESULTS / "audio_goal.json").read_text()),
        market=json.loads((RESULTS / "market.json").read_text()),
    )


def run_one(job: tuple[str, str, float]) -> dict:
    gid, source, fd = job
    if not _S:
        _init()
    s = _S
    text_th = s["market"]["entry_threshold_selected_on_valid"]
    sig = []
    if "text" in source:
        sig += signals_for_game(s["beliefs"][s["beliefs"].game_id == gid], s["market"]["pipeline_latency_s"])
    if "audio" in source:
        sig += audio_signals_for_game(
            s["tracks"][s["tracks"].game_id == gid], s["audio"]["threshold_selected_on_valid"]
        )
    sig.sort(key=lambda x: x.t)
    cfg = SimConfig(feed_delay=fd, cs_threshold=text_th)
    out = MatchSim(
        s["factory"](gid), s["events"][s["events"].game_id == gid], sig, cfg, final_score=final_score(s["games"], gid)
    ).run()
    return dict(
        game_id=gid,
        source=source,
        feed_delay=fd,
        cs_pnl=out["pnl"].get("cs", 0.0) / 100,
        cs_entries=out["cs_entries"],
        mm_markout_vs_cs=out["mm_markout"].get("cs", 0.0) / 100,
    )


def main() -> None:
    _init()
    have = set(_S["beliefs"].game_id) & set(_S["tracks"].game_id)
    games = [g for g in _S["games"][(_S["games"].split == "test") & _S["games"].fd_home.notna()].game_id if g in have]
    jobs = [(g, src, fd) for g in games for src in SOURCES for fd in FEED_DELAYS]
    print(f"{len(games)} matches, {len(jobs)} simulations", flush=True)
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 4) - 2), initializer=_init) as pool:
        df = pd.DataFrame(list(pool.map(run_one, jobs, chunksize=4)))
    piv = df.pivot_table(index=["game_id", "feed_delay"], columns="source", values="cs_pnl")
    summary = (
        df.groupby(["source", "feed_delay"])
        .agg(cs_pnl=("cs_pnl", "mean"), sem=("cs_pnl", "sem"), entries=("cs_entries", "mean"))
        .reset_index()
    )
    paired = {}
    for fd in FEED_DELAYS:
        d = piv.xs(fd, level="feed_delay")
        for a, b in (("audio", "text"), ("text+audio", "text")):
            diff = d[a] - d[b]
            paired[f"{a}_minus_{b}_feed_{fd:g}s"] = dict(
                mean=round(float(diff.mean()), 3), sem=round(float(diff.sem()), 3)
            )
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"text": "#c4214f", "audio": "#f2c230", "text+audio": "#1f5fd6"}
    for src, d in summary.groupby("source"):
        ax.errorbar(d.feed_delay, d.cs_pnl, yerr=d["sem"], marker="o", capsize=3, label=src, color=colors[src])
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Courtsider P&L per match by signal source", xlabel="market's feed delay (s)", ylabel="$ per match")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "market_audio.png", dpi=130)
    report = dict(n_games=len(games), summary=summary.round(3).to_dict(orient="records"), paired=paired)
    (RESULTS / "market_audio.json").write_text(json.dumps(report, indent=2))
    print(summary.round(2).to_string(index=False))
    print(json.dumps(paired, indent=1))


if __name__ == "__main__":
    main()
