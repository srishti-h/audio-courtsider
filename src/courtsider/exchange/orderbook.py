"""Price-time-priority limit order book for binary contracts priced in integer ticks 1..99.

Fixed ticks let each side be an array of FIFO queues: O(1) insert at a level, O(1) cancel
(lazy deletion), and best-price scans bounded by 99 levels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum

MIN_PRICE, MAX_PRICE = 1, 99


class Side(IntEnum):
    BUY = 1
    SELL = -1


class TIF(IntEnum):
    GTC = 0  # rest any remainder
    IOC = 1  # fill what you can, cancel the rest


@dataclass(slots=True)
class Order:
    oid: int
    account: str
    side: Side
    price: int
    qty: int
    seq: int


@dataclass(slots=True, frozen=True)
class Fill:
    maker_oid: int
    taker_oid: int
    maker: str
    taker: str
    price: int
    qty: int
    taker_side: Side


class OrderBook:
    def __init__(self) -> None:
        self.levels: dict[Side, list[deque[Order]]] = {
            Side.BUY: [deque() for _ in range(MAX_PRICE + 1)],
            Side.SELL: [deque() for _ in range(MAX_PRICE + 1)],
        }
        self.level_qty: dict[Side, list[int]] = {
            Side.BUY: [0] * (MAX_PRICE + 1),
            Side.SELL: [0] * (MAX_PRICE + 1),
        }
        self.orders: dict[int, Order] = {}
        self.touched: set[tuple[Side, int]] = set()  # levels changed since last drain (for market data)

    # ---- queries -------------------------------------------------------------------------
    def best_bid(self) -> int | None:
        q = self.level_qty[Side.BUY]
        for p in range(MAX_PRICE, MIN_PRICE - 1, -1):
            if q[p]:
                return p
        return None

    def best_ask(self) -> int | None:
        q = self.level_qty[Side.SELL]
        for p in range(MIN_PRICE, MAX_PRICE + 1):
            if q[p]:
                return p
        return None

    def depth(self, side: Side, n: int = 5) -> list[tuple[int, int]]:
        q = self.level_qty[side]
        rng = range(MAX_PRICE, MIN_PRICE - 1, -1) if side is Side.BUY else range(MIN_PRICE, MAX_PRICE + 1)
        out = [(p, q[p]) for p in rng if q[p]]
        return out[:n]

    def drain_touched(self) -> list[tuple[Side, int, int]]:
        out = [(s, p, self.level_qty[s][p]) for s, p in sorted(self.touched)]
        self.touched.clear()
        return out

    # ---- mutations -----------------------------------------------------------------------
    def _rest(self, o: Order) -> None:
        self.levels[o.side][o.price].append(o)
        self.level_qty[o.side][o.price] += o.qty
        self.orders[o.oid] = o
        self.touched.add((o.side, o.price))

    def submit(self, o: Order, tif: TIF = TIF.GTC) -> list[Fill]:
        """Match `o` against the opposite side; rest the remainder if GTC. Self-trades cancel the resting order."""
        fills: list[Fill] = []
        opp = Side(-o.side)
        levels, lq = self.levels[opp], self.level_qty[opp]
        prices = range(MIN_PRICE, o.price + 1) if o.side is Side.BUY else range(MAX_PRICE, o.price - 1, -1)
        for p in prices:
            if o.qty == 0:
                break
            if not lq[p]:
                continue
            queue = levels[p]
            while o.qty and queue:
                rest = queue[0]
                if rest.qty == 0:  # lazily cancelled
                    queue.popleft()
                    continue
                if rest.account == o.account:  # self-trade prevention: cancel resting
                    self.cancel(rest.oid)
                    continue
                q = min(o.qty, rest.qty)
                o.qty -= q
                rest.qty -= q
                lq[p] -= q
                self.touched.add((opp, p))
                fills.append(Fill(rest.oid, o.oid, rest.account, o.account, p, q, o.side))
                if rest.qty == 0:
                    queue.popleft()
                    del self.orders[rest.oid]
        if o.qty and tif is TIF.GTC:
            self._rest(o)
        return fills

    def cancel(self, oid: int) -> Order | None:
        o = self.orders.pop(oid, None)
        if o is None:
            return None
        self.level_qty[o.side][o.price] -= o.qty
        self.touched.add((o.side, o.price))
        cancelled = Order(o.oid, o.account, o.side, o.price, o.qty, o.seq)
        o.qty = 0  # lazy removal from the level queue
        return cancelled

    def reduce(self, oid: int, new_qty: int) -> bool:
        """Shrink an order in place, keeping its queue priority."""
        o = self.orders.get(oid)
        if o is None or not 0 < new_qty < o.qty:
            return False
        self.level_qty[o.side][o.price] -= o.qty - new_qty
        o.qty = new_qty
        self.touched.add((o.side, o.price))
        return True

    def check_invariants(self) -> None:
        for side in Side:
            for p in range(MIN_PRICE, MAX_PRICE + 1):
                live = sum(o.qty for o in self.levels[side][p] if o.qty and o.oid in self.orders)
                assert live == self.level_qty[side][p], (side, p, live, self.level_qty[side][p])
        bb, ba = self.best_bid(), self.best_ask()
        assert bb is None or ba is None or bb < ba, f"crossed book {bb} >= {ba}"
