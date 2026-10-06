"""Latency budget for the non-ASR stages: python -m courtsider.eval.run_bench [--skip-nlp]

  * exchange: per-message processing latency and throughput under a realistic message mix
  * text classifier: per-segment inference latency (batch of 1, i.e. streaming) on MPS/CPU
  * acoustic features: compute time per second of audio
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from courtsider.exchange.engine import Cancel, Exchange, NewOrder, RiskLimits
from courtsider.exchange.orderbook import TIF, Side
from courtsider.paths import MODELS, RESULTS


def bench_exchange(n: int = 200_000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ex = Exchange(["HOME", "DRAW", "AWAY", "OVER25", "BTTS"], RiskLimits(max_order_qty=1000, max_position=10**9))
    inst = ["HOME", "DRAW", "AWAY", "OVER25", "BTTS"]
    live: list[tuple[str, str, int]] = []
    lat = np.empty(n)
    kinds = rng.random(n)
    for i in range(n):
        if kinds[i] < 0.3 and live:  # cancel
            j = int(rng.integers(len(live)))
            acc, ins, oid = live[j]
            live[j] = live[-1]
            live.pop()
            msg = Cancel(acc, ins, oid)
        else:
            ins = inst[int(rng.integers(5))]
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            aggressive = kinds[i] > 0.9
            mid = 50
            px = int(mid + (3 if side is Side.BUY else -3)) if aggressive else int(
                mid - side * rng.integers(1, 10))
            msg = NewOrder(f"a{int(rng.integers(20))}", ins, int(side), px, int(rng.integers(1, 20)),
                           int(TIF.IOC if aggressive else TIF.GTC))
        t0 = time.perf_counter_ns()
        out = ex.process(msg)
        lat[i] = time.perf_counter_ns() - t0
        if isinstance(msg, NewOrder) and msg.tif == TIF.GTC:
            for e in out:
                if type(e).__name__ == "Ack" and e.resting_qty:
                    live.append((msg.account, msg.instrument, e.oid))
    us = lat / 1000
    return dict(messages=n, p50_us=float(np.percentile(us, 50)), p99_us=float(np.percentile(us, 99)),
                p999_us=float(np.percentile(us, 99.9)), throughput_msgs_per_s=float(n / (lat.sum() / 1e9)))


def bench_nlp(n: int = 200) -> dict:
    from courtsider.text.models import TransformerModel

    m = TransformerModel.load(MODELS / "text_extractor")
    text = "Real Madrid vs Barcelona | Benzema to Ronaldo, he's in on the left || and he scores! Ronaldo!"
    m.predict([text] * 8)  # warm-up
    lat = []
    for _ in range(n):
        t0 = time.perf_counter()
        m.predict([text], bs=1)
        lat.append(time.perf_counter() - t0)
    ms = np.array(lat) * 1000
    return dict(p50_ms=float(np.percentile(ms, 50)), p99_ms=float(np.percentile(ms, 99)))


def bench_acoustic(seconds: float = 120.0) -> dict:
    from courtsider.acoustic.features import analyse

    y = np.random.default_rng(0).standard_normal(int(seconds * 16000)).astype(np.float32) * 0.1
    t0 = time.perf_counter()
    analyse(y)
    el = time.perf_counter() - t0
    return dict(ms_per_audio_second=1000 * el / seconds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-nlp", action="store_true")
    args = ap.parse_args()
    out = dict(exchange=bench_exchange(), acoustic=bench_acoustic())
    if not args.skip_nlp:
        out["text_classifier"] = bench_nlp()
    path = RESULTS / "bench.json"
    prev = json.loads(path.read_text()) if path.exists() else {}
    prev.update(out)
    path.write_text(json.dumps(prev, indent=2))
    print(json.dumps(prev, indent=1))


if __name__ == "__main__":
    main()
