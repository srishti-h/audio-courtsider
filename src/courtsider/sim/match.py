"""Event-driven replay of one match on the exchange.

Information arrives on separate channels with their own delays:
  truth      — SoccerNet event time t_e (what happened on the pitch)
  feed       — official data feed at t_e + feed_delay; the market maker reprices on it
  TV         — viewers at t_e + tv_delay
  commentary — courtsider's detector, at segment end + ASR latency + radio delay
All agents act through the exchange, which processes messages in timestamp order.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from courtsider.exchange.engine import CancelAll, Exchange, NewOrder, RiskLimits, Settle, Trade
from courtsider.exchange.orderbook import TIF, Side
from courtsider.pricing.inplay import CONTRACTS, InPlayPricer, MatchState, settle

PRICE_EVENTS = ("goal", "red_card", "penalty")


@dataclass
class SimConfig:
    feed_delay: float = 2.0
    feed_jitter: float = 0.3
    tv_delay: float = 7.0
    mm_latency: float = 0.05
    mm_half_spread: float = 1.5
    mm_size: int = 20
    mm_requote_every: float = 5.0
    mm_skew_per_100: float = 1.0
    mm_audio_guard: bool = False  # pull quotes when the MM's own audio detector fires
    mm_guard_threshold: float = 0.6
    mm_guard_timeout: float = 20.0
    noise_rate: float = 1 / 15  # orders per second per contract
    noise_max_qty: int = 5
    tv_trader_qty: int = 20
    cs_threshold: float = 0.6
    cs_latency: float = 0.10  # decision + order entry; the MM (co-located) is assumed faster at 50 ms
    cs_edge: float = 2.0
    cs_max_qty: int = 40
    cs_unwind_after_feed: float = 1.0
    cs_timeout: float = 25.0
    markout_horizon: float = 30.0
    seed: int = 0


@dataclass(order=True)
class _Ev:
    t: float
    seq: int
    fn: Callable = field(compare=False)


@dataclass
class Signal:
    """A commentary-derived belief update available to traders at time `t`."""

    t: float
    probs: dict[str, float]  # event -> P(event just happened)
    team: dict[str, float]  # 'home'/'away' -> P(team | event)
    text: str = ""


class MinuteClock:
    """Maps the global video clock to the match minute using labelled events as anchors."""

    def __init__(self, events: pd.DataFrame):
        e = events.sort_values("g")
        self.g, self.m = e.g.to_numpy(), e.minute.to_numpy()

    def __call__(self, g: float) -> float:
        if len(self.g) == 0:
            return 0.0
        return float(np.interp(g, self.g, self.m))


class MatchSim:
    def __init__(
        self,
        pricer: InPlayPricer,
        events: pd.DataFrame,
        signals: list[Signal],
        cfg: SimConfig,
        guard_signals: list[Signal] | None = None,
        final_score: tuple[int, int] | None = None,
        record: bool = False,
    ):
        self.cfg, self.pricer = cfg, pricer
        self.rng = np.random.default_rng(cfg.seed)
        self.ex = Exchange(list(CONTRACTS), RiskLimits(max_order_qty=500, max_position=5000))
        self.minute = MinuteClock(events)
        self.truth = events[events.event.isin(PRICE_EVENTS) & events.team.notna()].sort_values("g")
        self.signals, self.guard_signals = signals, guard_signals or signals
        self.final_score = final_score
        self.q: list[_Ev] = []
        self._seq = itertools.count()
        self.now = 0.0
        self.end = float(events.g.max()) + 5 if len(events) else 0.0
        # agent state
        self.feed_state = MatchState()
        self.mm_orders_live = False
        self.mm_guard_until = -1.0
        self.cs_pending: dict | None = None
        self.fills: list[dict] = []
        self.timeline: list[dict] = []
        self.mid_history: dict[str, list[tuple[float, float]]] = {c: [] for c in CONTRACTS}
        self.record = record
        self.tape: list[dict] = []  # time-ordered replay log for the dashboard

    # ---- scheduling ---------------------------------------------------------------------
    def at(self, t: float, fn: Callable) -> None:
        heapq.heappush(self.q, _Ev(t, next(self._seq), fn))

    def send(self, msg, kind: str) -> list:
        out = self.ex.process(msg, self.now)
        touched = set()
        for e in out:
            if isinstance(e, Trade):
                f = e.fill
                buyer, seller = (f.taker, f.maker) if f.taker_side is Side.BUY else (f.maker, f.taker)
                fill = dict(
                    t=self.now,
                    instrument=e.instrument,
                    price=f.price,
                    qty=f.qty,
                    buyer=buyer,
                    seller=seller,
                    maker=f.maker,
                    taker=f.taker,
                )
                self.fills.append(fill)
                if self.record and kind != "mm":
                    self.tape.append(dict(kind="trade", **fill))
            elif self.record and hasattr(e, "md_seq"):
                touched.add(e.instrument)
        if self.record and kind != "mm":  # MM requotes are batched in mm_quote
            for inst in touched:
                self._record_book(inst)
        return out

    def _record_book(self, inst: str) -> None:
        b = self.ex.books[inst]
        self.tape.append(
            dict(kind="book", t=self.now, instrument=inst, bids=b.depth(Side.BUY, 5), asks=b.depth(Side.SELL, 5))
        )

    def _record_pnl(self) -> None:
        marks = {c: self.fair(self.feed_state)[c] for c in CONTRACTS}
        out = {}
        for name in ("cs", "mm"):
            acc = self.ex.account(name)
            out[name] = acc.cash + sum(acc.positions.get(c, 0) * marks[c] for c in CONTRACTS)
        self.tape.append(dict(kind="pnl", t=self.now, minute=self.minute(self.now), **out))

    def fair(self, state: MatchState) -> dict[str, float]:
        return self.pricer.prices(
            MatchState(
                self.minute(self.now),
                state.home_goals,
                state.away_goals,
                state.home_reds,
                state.away_reds,
                state.pending_penalty,
            )
        )

    # ---- market maker -------------------------------------------------------------------
    def mm_quote(self) -> None:
        cfg = self.cfg
        self.send(CancelAll("mm"), "mm")
        if self.now < self.mm_guard_until:
            self.mm_orders_live = False
            return
        fair = self.fair(self.feed_state)
        for c, v in fair.items():
            pos = self.ex.position("mm", c)
            mid = v - cfg.mm_skew_per_100 * pos / 100
            bid = int(np.floor(mid - cfg.mm_half_spread))
            ask = int(np.ceil(mid + cfg.mm_half_spread))
            if bid >= 1:
                self.send(NewOrder("mm", c, int(Side.BUY), min(bid, 98), cfg.mm_size), "mm")
            if ask <= 99:
                self.send(NewOrder("mm", c, int(Side.SELL), max(ask, 2), cfg.mm_size), "mm")
            self.mid_history[c].append((self.now, v))
        self.mm_orders_live = True
        if self.record:
            for c in CONTRACTS:
                self._record_book(c)
            self.tape.append(dict(kind="fair", t=self.now, prices=fair))

    def mm_periodic(self) -> None:
        self.mm_quote()
        if self.record:
            self._record_pnl()
        if self.now + self.cfg.mm_requote_every < self.end:
            self.at(self.now + self.cfg.mm_requote_every, self.mm_periodic)

    def on_feed(self, ev) -> None:
        if ev.event == "goal":
            self.feed_state = self.feed_state.with_goal(ev.team)
        elif ev.event == "red_card":
            self.feed_state = self.feed_state.with_red(ev.team)
        elif ev.event == "penalty":
            from dataclasses import replace

            self.feed_state = replace(self.feed_state, pending_penalty=ev.team)
        self.timeline.append(dict(t=self.now, kind="feed", event=ev.event, team=ev.team))
        self.mm_guard_until = -1.0
        self.at(self.now + self.cfg.mm_latency, self.mm_quote)
        if self.cs_pending is not None:
            self.at(self.now + self.cfg.cs_unwind_after_feed, self.cs_unwind)

    def on_penalty_resolved(self) -> None:
        if self.feed_state.pending_penalty is not None:
            from dataclasses import replace

            self.feed_state = replace(self.feed_state, pending_penalty=None)

    def on_guard_signal(self, s: Signal) -> None:
        if max(s.probs.get(e, 0.0) for e in PRICE_EVENTS) >= self.cfg.mm_guard_threshold:
            self.mm_guard_until = self.now + self.cfg.mm_guard_timeout
            self.timeline.append(dict(t=self.now, kind="mm_guard"))
            self.at(self.now + self.cfg.mm_latency, self.mm_quote)
            self.at(self.mm_guard_until + 0.01, self.mm_quote)

    # ---- noise and TV traders ------------------------------------------------------------
    def noise(self, contract: str) -> None:
        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        qty = int(self.rng.integers(1, self.cfg.noise_max_qty + 1))
        self.send(NewOrder("noise", contract, int(side), 99 if side is Side.BUY else 1, qty, int(TIF.IOC)), "noise")
        nxt = self.now + self.rng.exponential(1 / self.cfg.noise_rate)
        if nxt < self.end:
            self.at(nxt, lambda: self.noise(contract))

    def on_tv(self, ev) -> None:
        """Slow informed traders: trade at the stale price only if the book still hasn't moved."""
        state = self.feed_state if self.now >= ev.g + self.cfg.feed_delay else self._state_with(self.feed_state, ev)
        target = self.fair(state)
        self._take("tv", target, self.cfg.tv_trader_qty, edge=3.0)

    @staticmethod
    def _state_with(state: MatchState, ev) -> MatchState:
        if ev.event == "goal":
            return state.with_goal(ev.team)
        if ev.event == "red_card":
            return state.with_red(ev.team)
        return state

    def _take(self, account: str, target: dict[str, float], max_qty: int, edge: float) -> int:
        traded = 0
        for c, v in target.items():
            book = self.ex.books[c]
            ask, bid = book.best_ask(), book.best_bid()
            if ask is not None and ask < v - edge:
                px = int(min(98, np.floor(v - edge)))
                out = self.send(NewOrder(account, c, int(Side.BUY), px, max_qty, int(TIF.IOC)), account)
                traded += sum(e.fill.qty for e in out if isinstance(e, Trade))
            elif bid is not None and bid > v + edge:
                px = int(max(2, np.ceil(v + edge)))
                out = self.send(NewOrder(account, c, int(Side.SELL), px, max_qty, int(TIF.IOC)), account)
                traded += sum(e.fill.qty for e in out if isinstance(e, Trade))
        return traded

    # ---- courtsider ---------------------------------------------------------------------
    def on_signal(self, s: Signal) -> None:
        self.timeline.append(dict(t=self.now, kind="signal", probs=s.probs, team=s.team, text=s.text))
        if self.cs_pending is not None:
            return
        best = max(PRICE_EVENTS, key=lambda e: s.probs.get(e, 0.0))
        p = s.probs.get(best, 0.0)
        if p < self.cfg.cs_threshold:
            return
        now_state = self.feed_state
        base = self.fair(now_state)
        th, ta = s.team.get("home", 0.5), s.team.get("away", 0.5)
        m = self.minute(self.now)
        st = MatchState(
            m,
            now_state.home_goals,
            now_state.away_goals,
            now_state.home_reds,
            now_state.away_reds,
            now_state.pending_penalty,
        )
        if_home = self.pricer.prices_if(st, best, "home")
        if_away = self.pricer.prices_if(st, best, "away")
        target = {c: p * (th * if_home[c] + ta * if_away[c]) / max(th + ta, 1e-9) + (1 - p) * base[c] for c in base}
        traded = self._take("cs", target, self.cfg.cs_max_qty, self.cfg.cs_edge)
        if traded:
            self.cs_pending = dict(t=self.now, event=best, p=p)
            self.timeline.append(dict(t=self.now, kind="cs_entry", event=best, p=p))
            self.at(self.now + self.cfg.cs_timeout, self.cs_unwind)

    def cs_unwind(self) -> None:
        if self.cs_pending is None:
            return
        for c in CONTRACTS:
            pos = self.ex.position("cs", c)
            if pos:
                side = Side.SELL if pos > 0 else Side.BUY
                self.send(NewOrder("cs", c, int(side), 1 if pos > 0 else 99, min(abs(pos), 500), int(TIF.IOC)), "cs")
        self.timeline.append(dict(t=self.now, kind="cs_exit"))
        self.cs_pending = None

    # ---- run ----------------------------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        start = float(self.truth.g.min() - 1) if len(self.truth) else 0.0
        start = 0.0 if not np.isfinite(start) else 0.0
        self.at(start, self.mm_periodic)
        for c in CONTRACTS:
            self.at(start + 1 + self.rng.exponential(1 / cfg.noise_rate), lambda c=c: self.noise(c))
        for ev in self.truth.itertuples():
            self.timeline.append(dict(t=ev.g, kind="truth", event=ev.event, team=ev.team))
            d = cfg.feed_delay + abs(self.rng.normal(0, cfg.feed_jitter))
            self.at(ev.g + d, lambda ev=ev: self.on_feed(ev))
            self.at(ev.g + cfg.tv_delay, lambda ev=ev: self.on_tv(ev))
            if ev.event == "penalty":
                self.at(ev.g + 60.0, self.on_penalty_resolved)
        for s in self.signals:
            self.at(s.t + cfg.cs_latency, lambda s=s: self.on_signal(s))
        if cfg.mm_audio_guard:
            for s in self.guard_signals:
                self.at(s.t, lambda s=s: self.on_guard_signal(s))
        while self.q:
            ev = heapq.heappop(self.q)
            self.now = ev.t
            ev.fn()
        self.cs_unwind()
        hg, ag = self.final_score or (self.feed_state.home_goals, self.feed_state.away_goals)
        for c in CONTRACTS:
            self.send(Settle(c, settle(c, hg, ag)), "admin")
        return self.summary()

    # ---- analytics ----------------------------------------------------------------------
    def _mid_at(self, contract: str, t: float) -> float:
        h = self.mid_history[contract]
        ts = [x[0] for x in h]
        i = np.searchsorted(ts, t, side="right") - 1
        return h[max(i, 0)][1] if h else 50.0

    def summary(self) -> dict:
        """P&L per account plus MM markouts split by counterparty (spread capture vs adverse selection)."""
        pnl = {name: acc.cash for name, acc in self.ex.accounts.items()}
        mark = {"noise": 0.0, "cs": 0.0, "tv": 0.0}
        spread = {"noise": 0.0, "cs": 0.0, "tv": 0.0}
        for f in self.fills:
            if "mm" not in (f["buyer"], f["seller"]):
                continue
            cp = f["seller"] if f["buyer"] == "mm" else f["buyer"]
            sign = 1 if f["buyer"] == "mm" else -1
            mid_now = self._mid_at(f["instrument"], f["t"])
            mid_later = self._mid_at(f["instrument"], f["t"] + self.cfg.markout_horizon)
            spread[cp] = spread.get(cp, 0.0) + sign * (mid_now - f["price"]) * f["qty"]
            mark[cp] = mark.get(cp, 0.0) + sign * (mid_later - f["price"]) * f["qty"]
        cs_trades = [f for f in self.fills if "cs" in (f["buyer"], f["seller"])]
        entries = [x for x in self.timeline if x["kind"] == "cs_entry"]
        return dict(
            pnl=pnl,
            mm_spread_capture=spread,
            mm_markout=mark,
            cs_entries=len(entries),
            cs_volume=sum(f["qty"] for f in cs_trades),
            n_fills=len(self.fills),
        )
