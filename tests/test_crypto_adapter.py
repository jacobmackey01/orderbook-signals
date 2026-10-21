"""Synthetic checks for the Tardis/Binance adapter: synchronisation, gaps and causality."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lob_research.crypto.adapter import reconstruct, build_session, BOOK_EVENT, TRADE_EVENT
from lob_research.crypto.replication import calendar_folds
from lob_research.crypto.tardis import dataset_url
from lob_research.data import make_observations, walk_forward_folds, FEATURES

DAY = "2026-01-01"
DAY0 = 1767225600_000000  # 2026-01-01T00:00:00Z in microseconds


def rows(ts, snapshot, side, price, amount):
    return {"local_timestamp": ts, "is_snapshot": snapshot, "side": side, "price": price, "amount": amount}


def full_book(mid, levels=12, tick=0.01, size=1.0):
    asks = [(round(mid+tick*(k+1), 2), size+k) for k in range(levels)]
    bids = [(round(mid-tick*(k+1), 2), size+k) for k in range(levels)]
    return asks, bids


def tardis_frame(books, times_us, snapshot_at=(0,)):
    """Encode a sequence of full books as Tardis incremental rows (snapshots or level diffs)."""
    out, previous = [], None
    for k, ((asks, bids), ts) in enumerate(zip(books, times_us)):
        if k in snapshot_at or previous is None:
            out += [rows(ts, True, "ask", p, q) for p, q in asks] + [rows(ts, True, "bid", p, q) for p, q in bids]
        else:
            for side, new, old in [("ask", dict(asks), dict(previous[0])), ("bid", dict(bids), dict(previous[1]))]:
                out += [rows(ts, False, side, p, 0.) for p in old if p not in new]
                out += [rows(ts, False, side, p, q) for p, q in new.items() if old.get(p) != q]
            if out[-1]["local_timestamp"] != ts:  # unchanged book: emit a no-op message so indices align
                out.append(rows(ts, False, "ask", asks[0][0], asks[0][1]))
        previous = (asks, bids)
    return pd.DataFrame(out)


def walk(n, seed=0, start=100.0):
    rng = np.random.default_rng(seed)
    return np.round(start+np.cumsum(rng.choice([-0.01, 0, 0.01], size=n)), 2)


def no_trades():
    return pd.DataFrame({"local_timestamp": np.array([], dtype=np.int64), "side": np.array([], dtype=object),
                         "price": np.array([], dtype=float), "amount": np.array([], dtype=float)})


def test_dataset_url_only_free_first_of_month():
    assert dataset_url("binance", "trades", "2026-10-01", "BTCUSDT") == \
        "https://datasets.tardis.dev/v1/binance/trades/2026/10/01/BTCUSDT.csv.gz"
    with pytest.raises(ValueError):
        dataset_url("binance", "trades", "2026-10-02", "BTCUSDT")


def test_rows_before_first_snapshot_are_skipped_and_messages_read_whole():
    frame = pd.DataFrame([
        rows(1, False, "bid", 99.0, 5.),            # buffered update before any snapshot
        rows(2, True, "ask", 101.0, 1.), rows(2, True, "bid", 100.0, 1.),
        rows(3, False, "ask", 100.5, 2.), rows(3, False, "ask", 101.0, 0.),  # one message, two rows
    ])
    rec = reconstruct(frame.local_timestamp, frame.is_snapshot, frame.side == "bid", frame.price, frame.amount, levels=1)
    assert rec["pre_snapshot_rows_skipped"] == 1
    assert rec["time_us"].tolist() == [2, 3]  # one state per message, none before the snapshot
    assert rec["book"][1].tolist() == [100.5, 2., 100.0, 1.]
    assert rec["reset"].tolist() == [True, False]


def test_new_snapshot_discards_stale_levels_and_missing_deletes_are_counted():
    frame = pd.DataFrame([
        rows(1, True, "ask", 101.0, 1.), rows(1, True, "ask", 102.0, 1.), rows(1, True, "bid", 100.0, 1.),
        rows(2, False, "bid", 95.0, 0.),                     # delete of an absent level
        rows(3, True, "ask", 105.0, 3.), rows(3, True, "bid", 104.0, 3.),  # reconnect snapshot
    ])
    rec = reconstruct(frame.local_timestamp, frame.is_snapshot, frame.side == "bid", frame.price, frame.amount, levels=2)
    assert rec["snapshot_resets"] == 2 and rec["missing_level_deletes"] == 1
    last = rec["book"][-1]
    assert last[0] == 105.0 and np.isnan(last[4])  # 101/102 asks were discarded, not merged
    assert rec["reset"].tolist() == [True, False, True]


def test_two_consecutive_snapshots_with_different_timestamps_reset_twice():
    frame = pd.DataFrame([rows(1, True, "ask", 101.0, 1.), rows(1, True, "bid", 100.0, 1.),
                          rows(2, True, "ask", 103.0, 1.), rows(2, True, "bid", 102.0, 1.)])
    rec = reconstruct(frame.local_timestamp, frame.is_snapshot, frame.side == "bid", frame.price, frame.amount, levels=2)
    assert rec["snapshot_resets"] == 2 and rec["book"][-1][0] == 103.0 and np.isnan(rec["book"][-1][4])


def test_decreasing_local_timestamp_fails():
    frame = pd.DataFrame([rows(2, True, "ask", 101.0, 1.), rows(1, False, "bid", 100.0, 1.)])
    with pytest.raises(ValueError, match="decreases"):
        reconstruct(frame.local_timestamp, frame.is_snapshot, frame.side == "bid", frame.price, frame.amount)


def synthetic_session(n=900, step_us=100_000, snapshot_at=(0,), drop=(), seed=0, trades=None, gap_seconds=2.0):
    mids = walk(n, seed)
    times = DAY0+np.arange(n)*step_us
    keep = np.array([k not in drop for k in range(n)])
    books = [full_book(m) for m in mids]
    books = [b for b, k in zip(books, keep) if k]
    frame = tardis_frame(books, times[keep], snapshot_at=snapshot_at)
    return build_session("TEST", DAY, frame, no_trades() if trades is None else trades, lot_size=0.01,
                         gap_seconds=gap_seconds)


def test_scaling_matches_equity_convention():
    s = synthetic_session(n=20)
    # Price per lot x 10,000: 100.01 USDT x 0.01 lot x 10,000. Size in lots: 1.0 / 0.01.
    first_ask = walk(20)[0]+0.01
    assert s.book[0, 0] == pytest.approx(first_ask*0.01*10000)
    assert s.book[0, 1] == pytest.approx(100.)
    assert s.valid.all() and s.time[0] == 0.0


def test_silence_starts_new_segment_and_stale_trades_are_invalid():
    trades = pd.DataFrame({"local_timestamp": [DAY0+int(30.5e6), DAY0+int(31.5e6)], "side": ["buy", "sell"],
                           "price": [100., 100.], "amount": [0.02, 0.02]})
    s = synthetic_session(n=900, drop=range(300, 330), trades=trades)  # no book messages 30.0-32.9 s
    assert s.quality["silence_gaps"] == 1
    first_after = np.flatnonzero((s.time >= 33.0) & (s.event == BOOK_EVENT))[0]
    assert s.valid[first_after] and s.segment[first_after] > s.segment[first_after-1]
    trade_rows = np.flatnonzero(s.event == TRADE_EVENT)
    assert s.valid[trade_rows].tolist() == [True, False]  # book ages 0.6 s and 1.6 s


def test_trade_merge_tie_rule_and_direction():
    ts = DAY0+int(10e6)
    trades = pd.DataFrame({"local_timestamp": [ts, ts], "side": ["buy", "sell"], "price": [100., 100.],
                           "amount": [0.03, 0.05]})
    s = synthetic_session(n=200, trades=trades)
    at = np.flatnonzero(s.time == 10.0)
    assert s.event[at].tolist() == [BOOK_EVENT, TRADE_EVENT, TRADE_EVENT]
    assert s.direction[at[1:]].tolist() == [-1., 1.]  # resting side opposite the aggressor
    assert s.size[at[1:]] == pytest.approx([3., 5.])
    assert (s.book[at[1]] == s.book[at[0]]).all()  # trades carry the latest consistent book


def test_crossed_book_is_invalid_and_breaks_segment():
    mids = walk(60)
    books = [full_book(m) for m in mids]
    asks, bids = books[30]
    books[30] = ([(bids[0][0], 1.0)]+asks[1:], bids)  # best ask equals best bid: locked
    frame = tardis_frame(books, DAY0+np.arange(60)*100_000)
    s = build_session("TEST", DAY, frame, no_trades(), lot_size=0.01)
    assert not s.valid[30] and s.valid[29] and s.valid[31]
    assert s.segment[31] != s.segment[29]


def event_config():
    path = Path(__file__).resolve().parents[1]/"configs/crypto_binance.json"
    c = json.loads(path.read_text())
    c.update(event_stride=5, horizon_events=20)
    return c


def test_features_never_read_future_rows():
    base = synthetic_session(n=1500, seed=3)
    obs = make_observations(base, event_config())
    cut = int(obs.source_row.iloc[len(obs)//2])
    changed = synthetic_session(n=1500, seed=3)
    rng = np.random.default_rng(9)
    changed.book[cut+1:, 1::4] *= rng.uniform(0.5, 2.0, size=changed.book[cut+1:, 1::4].shape)
    obs2 = make_observations(changed, event_config())
    early = obs.source_row <= cut
    pd.testing.assert_frame_equal(obs.loc[early, FEATURES].reset_index(drop=True),
                                  obs2.loc[obs2.source_row <= cut, FEATURES].reset_index(drop=True))


def test_features_reset_after_reconnect_snapshot():
    """After a mid-day snapshot, no observation uses pre-reset state, so pre-reset data cannot leak."""
    n, reset = 1500, 700
    a = synthetic_session(n=n, seed=4, snapshot_at=(0, reset))
    b = synthetic_session(n=n, seed=4, snapshot_at=(0, reset))
    b.book[:reset, 1::4] *= 3.0  # perturb only pre-reset depth
    b.book[:reset, 0::4] += 50.0
    oa, ob = make_observations(a, event_config()), make_observations(b, event_config())
    reset_time = 70.0
    assert a.time[reset] == reset_time and a.segment[reset] > a.segment[reset-1]
    post_a, post_b = oa.loc[oa.time >= reset_time], ob.loc[ob.time >= reset_time]
    assert len(post_a) and post_a.time.min() >= reset_time+5  # 5 s look-back must sit inside the new segment
    pd.testing.assert_frame_equal(post_a[FEATURES].reset_index(drop=True), post_b[FEATURES].reset_index(drop=True))
    straddle = (oa.time < reset_time) & (oa.target_time >= reset_time)
    assert not straddle.any()  # targets do not cross the reset


def test_calendar_folds_match_full_walk_forward_rule():
    dates = [f"2026-{m:02d}-01" for m in range(1, 13)]
    frame = pd.DataFrame({"date": np.repeat(dates, 3), "time": np.tile([1., 2., 3.], 12)})
    config = event_config()
    expected = [(f, tr.date.unique().tolist(), va.date.unique().tolist(), te.date.unique().tolist())
                for f, tr, va, te, _ in walk_forward_folds(frame, config)]
    got = [(f, tr.date.unique().tolist(), va.date.unique().tolist(), te.date.unique().tolist())
           for f, tr, va, te, _ in calendar_folds(frame, dates, config)]
    assert got == expected and len(got) == 7 and expected[0][1] == dates[:4] and expected[0][2] == [dates[4]]
    # Development phase: only dates up to 2026-08 loaded; folds keep the full-calendar layout.
    dev = frame.loc[frame.date <= dates[7]]
    dev_folds = [(f, te.date.unique().tolist()) for f, _, _, te, _ in calendar_folds(dev, dates, config, test_dates=dates[5:8])]
    assert dev_folds == [(1, [dates[5]]), (2, [dates[6]]), (3, [dates[7]])]
    with pytest.raises(ValueError):
        list(calendar_folds(dev, dates, config, test_dates=[dates[8]]))
