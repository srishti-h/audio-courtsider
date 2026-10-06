"""Client-side order-book mirror built from snapshots + sequenced incremental deltas.

A gap in `md_seq` means a lost update; the mirror marks itself stale and must be re-seeded
from a fresh snapshot before it can be trusted again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from courtsider.exchange.engine import BookDelta, Exchange
from courtsider.exchange.orderbook import Side


@dataclass
class Snapshot:
    instrument: str
    md_seq: int
    bids: dict[int, int]
    asks: dict[int, int]


def snapshot(ex: Exchange, instrument: str) -> Snapshot:
    book = ex.books[instrument]
    return Snapshot(
        instrument,
        ex.md_seq[instrument],
        dict(book.depth(Side.BUY, 99)),
        dict(book.depth(Side.SELL, 99)),
    )


@dataclass
class BookMirror:
    instrument: str
    md_seq: int = -1
    bids: dict[int, int] = field(default_factory=dict)
    asks: dict[int, int] = field(default_factory=dict)
    stale: bool = True
    gaps: int = 0

    def seed(self, snap: Snapshot) -> None:
        self.md_seq, self.bids, self.asks, self.stale = snap.md_seq, dict(snap.bids), dict(snap.asks), False

    def apply(self, d: BookDelta) -> bool:
        """Apply a delta; returns False (and goes stale) on a sequence gap."""
        if self.stale or d.md_seq <= self.md_seq:
            return not self.stale
        if d.md_seq != self.md_seq + 1:
            self.stale = True
            self.gaps += 1
            return False
        side = self.bids if d.side == Side.BUY else self.asks
        if d.qty:
            side[d.price] = d.qty
        else:
            side.pop(d.price, None)
        self.md_seq = d.md_seq
        return True

    def best_bid(self) -> int | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> int | None:
        return min(self.asks) if self.asks else None
