from hypothesis import given, settings
from hypothesis import strategies as st

from courtsider.exchange.engine import (
    Ack,
    Cancel,
    CancelAll,
    Exchange,
    KillSwitch,
    NewOrder,
    Reduce,
    Reject,
    RiskLimits,
    Settle,
    Trade,
)
from courtsider.exchange.marketdata import BookMirror, snapshot
from courtsider.exchange.orderbook import TIF, Side

INSTR = ["HOME", "AWAY"]


def new(acc, side, price, qty, inst="HOME", tif=TIF.GTC):
    return NewOrder(acc, inst, int(side), price, qty, int(tif))


def test_price_time_priority():
    ex = Exchange(INSTR)
    ex.process(new("a", Side.SELL, 50, 5))
    ex.process(new("b", Side.SELL, 50, 5))
    ex.process(new("c", Side.SELL, 49, 5))
    out = ex.process(new("t", Side.BUY, 50, 12))
    fills = [e.fill for e in out if isinstance(e, Trade)]
    assert [(f.maker, f.price, f.qty) for f in fills] == [("c", 49, 5), ("a", 50, 5), ("b", 50, 2)]
    assert ex.position("t", "HOME") == 12
    assert ex.account("t").cash == -(49 * 5 + 50 * 7)


def test_ioc_does_not_rest_and_self_trade_cancels_resting():
    ex = Exchange(INSTR)
    ex.process(new("a", Side.SELL, 60, 3))
    out = ex.process(new("a", Side.BUY, 60, 3, tif=TIF.IOC))
    assert not any(isinstance(e, Trade) for e in out)
    assert ex.books["HOME"].best_ask() is None and ex.books["HOME"].best_bid() is None


def test_risk_checks_and_kill_switch():
    ex = Exchange(INSTR, RiskLimits(max_order_qty=10, max_position=15))
    assert isinstance(ex.process(new("a", Side.BUY, 0, 1))[0], Reject)
    assert isinstance(ex.process(new("a", Side.BUY, 50, 11))[0], Reject)
    ex.process(new("m", Side.SELL, 50, 10))
    ex.process(new("a", Side.BUY, 50, 10))
    assert isinstance(ex.process(new("a", Side.BUY, 50, 6))[0], Reject)  # would breach position 15
    ex.process(new("a", Side.BUY, 40, 2))
    ex.process(KillSwitch("a"))
    assert ex.books["HOME"].best_bid() is None
    assert isinstance(ex.process(new("a", Side.BUY, 40, 1))[0], Reject)


def test_reduce_keeps_priority_and_settlement_pays_out():
    ex = Exchange(INSTR)
    oid = next(e for e in ex.process(new("a", Side.BUY, 40, 10)) if isinstance(e, Ack)).oid
    ex.process(new("b", Side.BUY, 40, 10))
    ex.process(Reduce("a", "HOME", oid, 4))
    out = ex.process(new("s", Side.SELL, 40, 4))
    assert [e.fill.maker for e in out if isinstance(e, Trade)] == ["a"]
    ex.process(Settle("HOME", 100))
    assert ex.account("a").cash == -160 + 400
    assert ex.account("s").cash == 160 - 400
    assert isinstance(ex.process(new("a", Side.BUY, 40, 1))[0], Reject)


def test_journal_file_roundtrip(tmp_path):
    path = tmp_path / "journal.bin"
    ex = Exchange(INSTR, journal_path=path)
    for i in range(50):
        ex.process(new(f"acc{i % 3}", Side.BUY if i % 2 else Side.SELL, 40 + i % 7, 1 + i % 5), ts=float(i))
    ex.close()
    replayed = Exchange.replay(INSTR, Exchange.read_journal(path))
    assert replayed.state_hash() == ex.state_hash()


ops = st.one_of(
    st.tuples(
        st.just("new"),
        st.sampled_from(["a", "b", "c", "d"]),
        st.sampled_from(INSTR),
        st.sampled_from([Side.BUY, Side.SELL]),
        st.integers(1, 99),
        st.integers(1, 50),
        st.sampled_from([TIF.GTC, TIF.IOC]),
    ),
    st.tuples(st.just("cancel"), st.sampled_from(["a", "b", "c", "d"]), st.sampled_from(INSTR), st.integers(1, 200)),
    st.tuples(
        st.just("reduce"),
        st.sampled_from(["a", "b", "c", "d"]),
        st.sampled_from(INSTR),
        st.integers(1, 200),
        st.integers(1, 40),
    ),
    st.tuples(st.just("cancel_all"), st.sampled_from(["a", "b", "c", "d"])),
)


@settings(max_examples=300, deadline=None)
@given(st.lists(ops, min_size=1, max_size=150))
def test_invariants_under_random_order_flow(seq):
    ex = Exchange(INSTR, RiskLimits(max_order_qty=50, max_position=10_000))
    mirrors = {i: BookMirror(i) for i in INSTR}
    for i in INSTR:
        mirrors[i].seed(snapshot(ex, i))
    for op in seq:
        if op[0] == "new":
            _, acc, inst, side, price, qty, tif = op
            out = ex.process(new(acc, side, price, qty, inst, tif))
        elif op[0] == "cancel":
            out = ex.process(Cancel(op[1], op[2], op[3]))
        elif op[0] == "reduce":
            out = ex.process(Reduce(op[1], op[2], op[3], op[4]))
        else:
            out = ex.process(CancelAll(op[1]))
        for e in out:
            if hasattr(e, "md_seq"):
                assert mirrors[e.instrument].apply(e)
        for inst, book in ex.books.items():
            book.check_invariants()
            # positions and cash are zero-sum
            assert sum(a.positions.get(inst, 0) for a in ex.accounts.values()) == 0
            snap = snapshot(ex, inst)
            assert (mirrors[inst].bids, mirrors[inst].asks) == (snap.bids, snap.asks)
    assert abs(sum(a.cash for a in ex.accounts.values())) < 1e-9
    assert Exchange.replay(INSTR, ex.journal, ex.limits).state_hash() == ex.state_hash()


def test_mirror_detects_gap():
    ex = Exchange(INSTR)
    m = BookMirror("HOME")
    m.seed(snapshot(ex, "HOME"))
    d1 = [e for e in ex.process(new("a", Side.BUY, 40, 1)) if hasattr(e, "md_seq")]
    d2 = [e for e in ex.process(new("a", Side.BUY, 41, 1)) if hasattr(e, "md_seq")]
    assert not m.apply(d2[0]) and m.stale and m.gaps == 1
    m.seed(snapshot(ex, "HOME"))
    assert not m.stale and m.best_bid() == 41
    assert d1  # first delta was "lost"
