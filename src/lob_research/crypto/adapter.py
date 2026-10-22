"""Map Tardis incremental_book_L2 + trades CSVs onto the equity study's Session.

Reconstruction follows Tardis's documented procedure for incremental_book_L2:
skip rows before the first snapshot; treat rows sharing a local_timestamp as one
WebSocket message and read the book only after the whole message; reset the
book when a snapshot starts (row is a snapshot and the previous row was not, or
belonged to a snapshot with a different local_timestamp); amounts are absolute
level sizes and 0 deletes a level.

Binance's U/u sequence numbers are not in the CSV. Tardis checks them while
recording and reconnects on a gap, which shows up here as a new snapshot. Such
resets, silences longer than ``gap_seconds`` between book messages, crossed or
locked books and shallow books all start a new segment, so the existing feature
and execution code cannot read across them.
"""

from __future__ import annotations

from bisect import bisect_left
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import Session

BOOK_EVENT, TRADE_EVENT = 1, 4


class _Side:
    """One side of an L2 book: level dict plus a sorted key list (asks ascending, bids by -price)."""

    __slots__ = ("sign", "levels", "keys")

    def __init__(self, sign: float):
        self.sign = sign
        self.levels: dict[float, float] = {}
        self.keys: list[float] = []

    def clear(self):
        self.levels.clear()
        self.keys.clear()

    def set(self, price: float, amount: float) -> bool:
        """Apply an absolute level amount. Returns False when deleting a level that is absent."""
        key = self.sign*price
        if amount == 0:
            if key not in self.levels:
                return False
            del self.levels[key]
            del self.keys[bisect_left(self.keys, key)]
            return True
        if key not in self.levels:
            self.keys.insert(bisect_left(self.keys, key), key)
        self.levels[key] = amount
        return True

    def top(self, n: int):
        keys = self.keys[:n]
        return [self.sign*k for k in keys], [self.levels[k] for k in keys]


def reconstruct(local_ts, is_snapshot, is_bid, price, amount, levels: int = 10):
    """Replay incremental L2 rows; return one consistent top-of-book state per message.

    Returns a dict with message timestamps (int microseconds), an (n, 4*levels) book in
    LOBSTER column order (ask price, ask size, bid price, bid size per level, NaN where a
    level is missing), a reset flag per message and counters for the quality report.
    """
    ts = np.asarray(local_ts, dtype=np.int64)
    if len(ts) and np.any(np.diff(ts) < 0):
        raise ValueError("local_timestamp decreases within the file; Tardis guarantees monotonic arrival times")
    snap = np.asarray(is_snapshot, dtype=bool)
    starts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]]) if len(ts) else np.array([], dtype=int)
    ends = np.r_[starts[1:], len(ts)]
    ts_l, snap_l = ts.tolist(), snap.tolist()
    bid_l = np.asarray(is_bid, dtype=bool).tolist()
    price_l = np.asarray(price, dtype=float).tolist()
    amount_l = np.asarray(amount, dtype=float).tolist()
    asks, bids = _Side(1.), _Side(-1.)
    started = False
    prev_snap, prev_ts = False, None
    out_ts, out_reset, rows = [], [], []
    counters = {"level_rows": len(ts), "messages": len(starts), "pre_snapshot_rows_skipped": 0,
                "snapshot_resets": 0, "missing_level_deletes": 0, "negative_amount_rows": 0}
    for start, end in zip(starts.tolist(), ends.tolist()):
        reset = False
        for i in range(start, end):
            if snap_l[i] and (not prev_snap or prev_ts != ts_l[i]):
                asks.clear()
                bids.clear()
                started = True
                reset = True
                counters["snapshot_resets"] += 1
            prev_snap, prev_ts = snap_l[i], ts_l[i]
            if not started:
                counters["pre_snapshot_rows_skipped"] += 1
                continue
            if amount_l[i] < 0:
                counters["negative_amount_rows"] += 1
                continue
            side = bids if bid_l[i] else asks
            if not side.set(price_l[i], amount_l[i]):
                counters["missing_level_deletes"] += 1
        if not started:
            continue
        ap, aq = asks.top(levels)
        bp, bq = bids.top(levels)
        row = np.full(4*levels, np.nan)
        row[0:4*len(ap):4], row[1:4*len(aq):4] = ap, aq
        row[2:4*len(bp):4], row[3:4*len(bq):4] = bp, bq
        rows.append(row)
        out_ts.append(ts_l[start])
        out_reset.append(reset)
    book = np.vstack(rows) if rows else np.empty((0, 4*levels))
    return {"time_us": np.asarray(out_ts, dtype=np.int64), "book": book,
            "reset": np.asarray(out_reset, dtype=bool), **counters}


def book_validity(book: np.ndarray, levels: int = 10) -> np.ndarray:
    ask, aq, bid, bq = book[:, 0::4], book[:, 1::4], book[:, 2::4], book[:, 3::4]
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(book).all(axis=1) & (aq > 0).all(axis=1) & (bq > 0).all(axis=1)
        ok &= (ask > 0).all(axis=1) & (bid > 0).all(axis=1)
        ok &= (np.diff(ask, axis=1) > 0).all(axis=1) & (np.diff(bid, axis=1) < 0).all(axis=1)
        ok &= bid[:, 0] < ask[:, 0]
    return ok


def build_session(symbol: str, date: str, book_frame: pd.DataFrame, trade_frame: pd.DataFrame,
                  lot_size: float, gap_seconds: float = 2.0, levels: int = 10,
                  max_book_age: float = 1.0) -> Session:
    """Merge reconstructed book messages and trades into one causal event stream.

    Times are seconds since 00:00 UTC of ``date`` by Tardis local_timestamp. At equal
    timestamps a book message precedes a trade. Prices are stored as USDT per lot x 10,000
    and sizes in lots so the equity execution code (one share = one lot) applies unchanged.
    """
    day0 = int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp()*1_000_000)
    rec = reconstruct(book_frame.local_timestamp.to_numpy(), book_frame.is_snapshot.to_numpy(),
                      (book_frame.side.to_numpy() == "bid"), book_frame.price.to_numpy(),
                      book_frame.amount.to_numpy(), levels)
    trade_ts = trade_frame.local_timestamp.to_numpy(dtype=np.int64)
    if len(trade_ts) and np.any(np.diff(trade_ts) < 0):
        raise ValueError("Trade local_timestamp decreases within the file")
    msg_ts, book = rec["time_us"], rec["book"]
    if len(msg_ts) == 0:
        raise ValueError(f"No book snapshot in {symbol} {date}")
    # Silence between consecutive book messages suggests missing messages.
    gap = np.r_[False, np.diff(msg_ts) > gap_seconds*1e6]
    book_ok = book_validity(book, levels)
    # Book messages first at equal timestamps: the trade then carries the latest book.
    kind = np.r_[np.zeros(len(msg_ts), dtype=int), np.ones(len(trade_ts), dtype=int)]
    all_ts = np.r_[msg_ts, trade_ts]
    source = np.r_[np.arange(len(msg_ts)), np.arange(len(trade_ts))]
    order = np.lexsort((source, kind, all_ts))
    kind, all_ts, source = kind[order], all_ts[order], source[order]
    book_index = np.searchsorted(msg_ts, all_ts, side="right")-1
    has_book = book_index >= 0
    safe = np.clip(book_index, 0, None)
    age = (all_ts-msg_ts[safe])/1e6
    is_trade = kind == 1
    # A trade row would otherwise make a stale book look fresh to the quote-age checks.
    valid = has_book & book_ok[safe] & ~(is_trade & (age > max_book_age))
    boundary = ~valid | (~is_trade & (rec["reset"][safe] | gap[safe]))
    events = np.where(is_trade, TRADE_EVENT, BOOK_EVENT)
    size = np.zeros(len(all_ts))
    direction = np.ones(len(all_ts))
    t_side = trade_frame.side.to_numpy()
    t_amount = trade_frame.amount.to_numpy(dtype=float)
    ti = source[is_trade]
    size[is_trade] = t_amount[ti]/lot_size
    # Resting-order direction is opposite to the aggressor, as in LOBSTER's convention.
    direction[is_trade] = np.select([t_side[ti] == "buy", t_side[ti] == "sell"], [-1., 1.], 0.)
    scaled = book[safe].copy()
    scaled[:, 0::4] *= lot_size*10000
    scaled[:, 2::4] *= lot_size*10000
    scaled[:, 1::4] /= lot_size
    scaled[:, 3::4] /= lot_size
    scaled[~has_book] = np.nan
    scaled = np.nan_to_num(scaled, nan=0.0)
    time = (all_ts-day0)/1e6
    if len(time) and (time[0] < 0 or time[-1] >= 86400):
        raise ValueError(f"{symbol} {date} contains timestamps outside the UTC day")
    quality = {k: v for k, v in rec.items() if k not in ("time_us", "book", "reset")}
    quality.update({
        "source_rows": int(len(time)), "book_events": int((~is_trade).sum()), "trade_events": int(is_trade.sum()),
        "unknown_side_trades": int((direction[is_trade] == 0).sum()),
        "midday_snapshot_resets": int(max(rec["snapshot_resets"]-1, 0)),
        "silence_gaps": int(gap.sum()), "max_book_gap_seconds": float(np.diff(msg_ts).max()/1e6) if len(msg_ts) > 1 else 0.,
        "crossed_or_locked_messages": int((book[:, 2] >= book[:, 0]).sum()),
        "shallow_messages": int((~np.isfinite(book).all(axis=1)).sum()),
        "invalid_book_rows": int((~valid).sum()), "segments": int(boundary.sum()),
        "first_seconds": float(time[0]), "last_seconds": float(time[-1])})
    return Session(symbol, date, time, events, size, direction, scaled, valid, np.cumsum(boundary), quality)


def read_tardis(book_path: Path, trade_path: Path):
    book = pd.read_csv(book_path, usecols=["local_timestamp", "is_snapshot", "side", "price", "amount"], engine="pyarrow")
    trades = pd.read_csv(trade_path, usecols=["local_timestamp", "side", "price", "amount"], engine="pyarrow")
    if book.is_snapshot.dtype != bool:
        book["is_snapshot"] = book.is_snapshot.astype(str).str.lower().eq("true")
    return book, trades


def load_tardis_session(book_path: Path, trade_path: Path, symbol: str, date: str, lot_size: float,
                        gap_seconds: float = 2.0, levels: int = 10, max_book_age: float = 1.0) -> Session:
    book, trades = read_tardis(book_path, trade_path)
    session = build_session(symbol, date, book, trades, lot_size, gap_seconds, levels, max_book_age)
    for name, path in [("book_sha256", book_path), ("trades_sha256", trade_path)]:
        digest = sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        session.quality[name] = digest.hexdigest()
    return session
