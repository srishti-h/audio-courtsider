"""Deterministic, event-sourced exchange: sequencer -> risk checks -> matching -> accounting.

Every inbound message is stamped with a sequence number and appended to the journal before it
is applied, so replaying the journal reproduces the exact state (see `state_hash`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import msgspec

from courtsider.exchange.orderbook import TIF, Fill, Order, OrderBook, Side

# ---- inbound messages -------------------------------------------------------------------


class NewOrder(msgspec.Struct, tag=True):
    account: str
    instrument: str
    side: int
    price: int
    qty: int
    tif: int = TIF.GTC


class Cancel(msgspec.Struct, tag=True):
    account: str
    instrument: str
    oid: int


class Reduce(msgspec.Struct, tag=True):
    account: str
    instrument: str
    oid: int
    qty: int


class CancelAll(msgspec.Struct, tag=True):
    account: str
    instrument: str | None = None


class KillSwitch(msgspec.Struct, tag=True):
    account: str
    enabled: bool = True


class Settle(msgspec.Struct, tag=True):
    instrument: str
    value: float


Inbound = NewOrder | Cancel | Reduce | CancelAll | KillSwitch | Settle


class JournalEntry(msgspec.Struct):
    seq: int
    ts: float
    msg: Inbound


# ---- outbound events --------------------------------------------------------------------


@dataclass(slots=True)
class Ack:
    seq: int
    account: str
    oid: int
    resting_qty: int


@dataclass(slots=True)
class Reject:
    seq: int
    account: str
    reason: str


@dataclass(slots=True)
class Trade:
    seq: int
    ts: float
    instrument: str
    fill: Fill


@dataclass(slots=True)
class BookDelta:
    seq: int
    instrument: str
    md_seq: int
    side: int
    price: int
    qty: int


@dataclass(slots=True)
class Canceled:
    seq: int
    account: str
    oid: int


@dataclass
class RiskLimits:
    max_order_qty: int = 500
    max_position: int = 2000


@dataclass
class Account:
    cash: float = 0.0
    positions: dict[str, int] = field(default_factory=dict)
    killed: bool = False


class Exchange:
    def __init__(self, instruments: list[str], limits: RiskLimits | None = None, journal_path: Path | None = None):
        self.books = {i: OrderBook() for i in instruments}
        self.limits = limits or RiskLimits()
        self.accounts: dict[str, Account] = {}
        self.settled: dict[str, float] = {}
        self.seq = 0
        self.next_oid = 1
        self.md_seq = {i: 0 for i in instruments}
        self.owner: dict[tuple[str, int], str] = {}
        self.journal: list[JournalEntry] = []
        self._journal_file = journal_path.open("wb") if journal_path else None
        self._enc = msgspec.msgpack.Encoder()

    # ---- helpers ------------------------------------------------------------------------
    def account(self, name: str) -> Account:
        return self.accounts.setdefault(name, Account())

    def position(self, account: str, instrument: str) -> int:
        return self.account(account).positions.get(instrument, 0)

    def _apply_fill(self, instrument: str, f: Fill) -> None:
        buyer, seller = (f.taker, f.maker) if f.taker_side is Side.BUY else (f.maker, f.taker)
        for name, sign in ((buyer, 1), (seller, -1)):
            acc = self.account(name)
            acc.positions[instrument] = acc.positions.get(instrument, 0) + sign * f.qty
            acc.cash -= sign * f.price * f.qty

    def _deltas(self, instrument: str) -> list[BookDelta]:
        out = []
        for side, price, qty in self.books[instrument].drain_touched():
            self.md_seq[instrument] += 1
            out.append(BookDelta(self.seq, instrument, self.md_seq[instrument], int(side), price, qty))
        return out

    # ---- entry point --------------------------------------------------------------------
    def process(self, msg: Inbound, ts: float = 0.0) -> list:
        self.seq += 1
        entry = JournalEntry(self.seq, ts, msg)
        self.journal.append(entry)
        if self._journal_file:
            payload = self._enc.encode(entry)  # length-prefixed framing: msgpack bytes may contain newlines
            self._journal_file.write(len(payload).to_bytes(4, "big") + payload)
        handler = getattr(self, f"_on_{type(msg).__name__}")
        return handler(msg, ts)

    def _reject(self, account: str, reason: str) -> list:
        return [Reject(self.seq, account, reason)]

    def _on_NewOrder(self, m: NewOrder, ts: float) -> list:
        acc = self.account(m.account)
        if m.instrument not in self.books or m.instrument in self.settled:
            return self._reject(m.account, "unknown or settled instrument")
        if acc.killed:
            return self._reject(m.account, "kill switch engaged")
        if not 1 <= m.price <= 99:
            return self._reject(m.account, "price out of range")
        if not 0 < m.qty <= self.limits.max_order_qty:
            return self._reject(m.account, "bad quantity")
        if abs(self.position(m.account, m.instrument) + m.side * m.qty) > self.limits.max_position:
            return self._reject(m.account, "position limit")
        oid = self.next_oid
        self.next_oid += 1
        order = Order(oid, m.account, Side(m.side), m.price, m.qty, self.seq)
        book = self.books[m.instrument]
        fills = book.submit(order, TIF(m.tif))
        out: list = []
        for f in fills:
            self._apply_fill(m.instrument, f)
            out.append(Trade(self.seq, ts, m.instrument, f))
        if oid in book.orders:
            self.owner[(m.instrument, oid)] = m.account
        out.append(Ack(self.seq, m.account, oid, book.orders[oid].qty if oid in book.orders else 0))
        return out + self._deltas(m.instrument)

    def _on_Cancel(self, m: Cancel, ts: float) -> list:
        if self.owner.get((m.instrument, m.oid)) != m.account:
            return self._reject(m.account, "unknown order")
        if self.books[m.instrument].cancel(m.oid) is None:
            return self._reject(m.account, "unknown order")
        del self.owner[(m.instrument, m.oid)]
        return [Canceled(self.seq, m.account, m.oid), *self._deltas(m.instrument)]

    def _on_Reduce(self, m: Reduce, ts: float) -> list:
        if self.owner.get((m.instrument, m.oid)) != m.account or not self.books[m.instrument].reduce(m.oid, m.qty):
            return self._reject(m.account, "cannot reduce")
        return self._deltas(m.instrument)

    def _cancel_all(self, account: str, instrument: str | None) -> list:
        out: list = []
        for inst, book in self.books.items():
            if instrument and inst != instrument:
                continue
            for oid in [o for o, a in self.owner.items() if a == account and o[0] == inst]:
                book.cancel(oid[1])
                del self.owner[oid]
                out.append(Canceled(self.seq, account, oid[1]))
            out += self._deltas(inst)
        return out

    def _on_CancelAll(self, m: CancelAll, ts: float) -> list:
        return self._cancel_all(m.account, m.instrument)

    def _on_KillSwitch(self, m: KillSwitch, ts: float) -> list:
        self.account(m.account).killed = m.enabled
        return self._cancel_all(m.account, None) if m.enabled else []

    def _on_Settle(self, m: Settle, ts: float) -> list:
        out: list = []
        book = self.books[m.instrument]
        for oid in [o for o in self.owner if o[0] == m.instrument]:
            book.cancel(oid[1])
            out.append(Canceled(self.seq, self.owner.pop(oid), oid[1]))
        for acc in self.accounts.values():
            pos = acc.positions.pop(m.instrument, 0)
            acc.cash += pos * m.value
        self.settled[m.instrument] = m.value
        return out + self._deltas(m.instrument)

    # ---- determinism --------------------------------------------------------------------
    def state_hash(self) -> str:
        h = hashlib.sha256()
        for inst in sorted(self.books):
            b = self.books[inst]
            for oid in sorted(b.orders):
                o = b.orders[oid]
                h.update(f"{inst}|{o.oid}|{o.account}|{int(o.side)}|{o.price}|{o.qty}|{o.seq};".encode())
        for name in sorted(self.accounts):
            a = self.accounts[name]
            pos = ",".join(f"{k}:{v}" for k, v in sorted(a.positions.items()))
            h.update(f"{name}|{a.cash:.6f}|{pos}|{a.killed};".encode())
        h.update(f"{self.seq}|{self.next_oid}".encode())
        return h.hexdigest()

    @classmethod
    def replay(cls, instruments: list[str], journal: list[JournalEntry], limits: RiskLimits | None = None) -> Exchange:
        ex = cls(instruments, limits)
        for e in journal:
            ex.process(e.msg, e.ts)
        return ex

    @staticmethod
    def read_journal(path: Path) -> list[JournalEntry]:
        dec = msgspec.msgpack.Decoder(JournalEntry)
        data, out, i = path.read_bytes(), [], 0
        while i < len(data):
            n = int.from_bytes(data[i : i + 4], "big")
            out.append(dec.decode(data[i + 4 : i + 4 + n]))
            i += 4 + n
        return out

    def close(self) -> None:
        if self._journal_file:
            self._journal_file.close()
