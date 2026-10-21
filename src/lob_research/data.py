"""Source acquisition, paired-file validation and causal snapshot replay."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

SOURCE = "https://php.lobsterdata.com/info/sample/"
SYMBOLS = ("AAPL", "AMZN", "GOOG", "INTC", "MSFT")


def download_samples(root: Path, symbols=SYMBOLS) -> list[dict]:
    """Download official samples, verifying against the committed archive hashes."""
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "data/sample-manifest.json"
    expected = {r["symbol"]: r for r in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    manifest = []
    for symbol in symbols:
        if symbol not in SYMBOLS:
            raise ValueError(f"Unsupported official sample symbol: {symbol}")
        url = SOURCE + f"LOBSTER_SampleFile_{symbol}_2012-06-21_10.zip"
        archive = raw / f"{symbol}.zip"
        blob = archive.read_bytes() if archive.exists() else urlopen(url, timeout=90).read()
        digest = sha256(blob).hexdigest()
        if symbol in expected and digest != expected[symbol]["sha256"]:
            raise ValueError(f"Archive hash mismatch for {symbol}; inspect provider changes before replacing the manifest")
        with ZipFile(BytesIO(blob)) as z:
            members = z.namelist()
            csvs = [n for n in members if n.endswith(".csv")]
            if len(csvs) != 2 or not any("message_10" in n for n in csvs) or not any("orderbook_10" in n for n in csvs):
                raise ValueError(f"Unexpected sample archive layout: {members}")
            for name in members:
                if name.lower().endswith((".csv", ".txt")):
                    # Flatten known archive members; never honour paths supplied by a ZIP.
                    (raw / Path(name).name).write_bytes(z.read(name))
        archive.write_bytes(blob)
        manifest.append({"symbol": symbol, "url": url, "sha256": digest, "bytes": len(blob), "members": members})
    if not expected:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@dataclass
class Session:
    symbol: str
    date: str
    time: np.ndarray
    event: np.ndarray
    size: np.ndarray
    direction: np.ndarray
    book: np.ndarray
    valid: np.ndarray
    segment: np.ndarray
    quality: dict

    def at(self, query: np.ndarray, max_age: float = 1.0):
        query = np.asarray(query, dtype=float)
        index = np.searchsorted(self.time, query, side="right") - 1
        safe = np.clip(index, 0, len(self.time) - 1)
        age = query - self.time[safe]
        ok = (index >= 0) & (age >= 0) & (age <= max_age) & self.valid[safe]
        return safe, ok


def validate_arrays(messages: np.ndarray, book: np.ndarray, levels: int) -> tuple[np.ndarray, dict]:
    if messages.ndim != 2 or messages.shape[1] != 6:
        raise ValueError("Message file must have six columns")
    if book.ndim != 2 or book.shape[1] != 4 * levels or len(book) != len(messages):
        raise ValueError("Paired book/message dimensions do not match")
    if len(book) < 2 or not np.isfinite(messages).all() or not np.isfinite(book).all():
        raise ValueError("Empty, short or non-finite source")
    if np.any(np.diff(messages[:, 0]) < 0):
        raise ValueError("Source timestamps are out of order; sorting would break paired-event semantics")
    if not np.isin(messages[:, 1], np.arange(1, 8)).all():
        raise ValueError("Unknown LOBSTER event type")
    if np.any(messages[:, 3] < 0) or not np.isin(messages[:, 5], [-1, 1]).all():
        raise ValueError("Invalid message size or direction")
    # Columns are ask price, ask size, bid price, bid size, repeated per level.
    ask, aq, bid, bq = book[:, 0::4], book[:, 1::4], book[:, 2::4], book[:, 3::4]
    if np.any(aq < 0) or np.any(bq < 0):
        raise ValueError("Negative displayed depth")
    valid = (ask > 0).all(axis=1) & (ask < 9e9).all(axis=1) & (bid > 0).all(axis=1)
    valid &= (aq > 0).all(axis=1) & (bq > 0).all(axis=1)
    valid &= (np.diff(ask, axis=1) > 0).all(axis=1) & (np.diff(bid, axis=1) < 0).all(axis=1)
    valid &= bid[:, 0] < ask[:, 0]
    halted = np.zeros(len(book), dtype=bool)
    state = False
    for i in np.flatnonzero(messages[:, 1] == 7):
        # Halt (-1), quoting resumes (0), trading resumes (1).
        state = messages[i, 4] != 1
        end = np.searchsorted(np.flatnonzero(messages[:, 1] == 7), i, side="right")
        markers = np.flatnonzero(messages[:, 1] == 7)
        stop = markers[end] if end < len(markers) else len(book)
        if state:
            halted[i:stop] = True
    valid &= ~halted & ~np.isin(messages[:, 1], [6, 7])
    q = {"source_rows": len(book), "invalid_book_rows": int((~valid).sum()),
         "equal_timestamp_pairs": int((np.diff(messages[:, 0]) == 0).sum()),
         "cross_or_halt_events": int(np.isin(messages[:, 1], [6, 7]).sum()),
         "first_seconds": float(messages[0, 0]), "last_seconds": float(messages[-1, 0])}
    return valid, q


def load_session(message_path: Path, levels: int = 10) -> Session:
    book_path = message_path.with_name(message_path.name.replace("_message_", "_orderbook_"))
    messages = pd.read_csv(message_path, header=None).to_numpy(dtype=float)
    book = pd.read_csv(book_path, header=None).to_numpy(dtype=float)
    valid, quality = validate_arrays(messages, book, levels)
    parts = message_path.name.split("_")
    # Any invalid source state breaks the uninterrupted segment, even if a grid misses it.
    boundaries = ~valid | np.isin(messages[:, 1], [6, 7])
    segment = np.cumsum(boundaries)
    quality["message_sha256"] = sha256(message_path.read_bytes()).hexdigest()
    quality["book_sha256"] = sha256(book_path.read_bytes()).hexdigest()
    return Session(parts[0], parts[1], messages[:, 0], messages[:, 1].astype(int),
                   messages[:, 3], messages[:, 5], book, valid, segment, quality)


def ofi_events(book: np.ndarray) -> np.ndarray:
    """Best-quote OFI: price changes reset which displayed queues contribute."""
    ask, aq, bid, bq = book[:, 0], book[:, 1], book[:, 2], book[:, 3]
    result = np.zeros(len(book))
    result[1:] = ((bid[1:] >= bid[:-1]) * bq[1:] - (bid[1:] <= bid[:-1]) * bq[:-1]
                  - (ask[1:] <= ask[:-1]) * aq[1:] + (ask[1:] >= ask[:-1]) * aq[:-1])
    return result


def window_sum(time, values, grid, window, right_indices=None):
    cumulative = np.r_[0.0, np.cumsum(values)]
    right = np.searchsorted(time, grid, side="right") if right_indices is None else np.asarray(right_indices)+1
    left = np.searchsorted(time, grid - window, side="right")
    return cumulative[right] - cumulative[left]


FEATURES = ["spread_bps", "imbalance_1", "imbalance_5", "imbalance_10", "microprice_bps",
            "log_depth", "ofi_500ms", "ofi_5s", "trade_imbalance_5s", "event_rate_5s",
            "return_500ms_bps", "return_5s_bps"]


def make_observations(session: Session, config: dict) -> pd.DataFrame:
    s, c = session, config
    if c.get("sampling", "clock") == "event":
        idx = np.arange(0, len(s.time)-c["horizon_events"], c["event_stride"])
        idx = idx[(s.time[idx] >= c["start_seconds"]) & (s.time[idx] < c["end_seconds"])]
        grid = s.time[idx]
        target_idx = idx+c["horizon_events"]
        target_time = s.time[target_idx]
        ok, target_ok = s.valid[idx], s.valid[target_idx]
    else:
        grid = np.arange(c["start_seconds"], c["end_seconds"], c["grid_seconds"])
        idx, ok = s.at(grid, c["max_quote_age_seconds"])
        target_time = grid+c["horizon_seconds"]
        target_idx, target_ok = s.at(target_time, c["max_quote_age_seconds"])
    old_idx, old_ok = s.at(grid - 5, c["max_quote_age_seconds"])
    lag_idx, lag_ok = s.at(grid - .5, c["max_quote_age_seconds"])
    ask, aq, bid, bq = s.book[idx, 0::4], s.book[idx, 1::4], s.book[idx, 2::4], s.book[idx, 3::4]
    mid_raw = (s.book[:, 0] + s.book[:, 2]) / 2
    mid = mid_raw[idx]
    delta = mid_raw[target_idx] - mid
    out = pd.DataFrame({"symbol": s.symbol, "date": s.date, "time": grid,
                        "source_row": idx, "source_time": s.time[idx], "target_row": target_idx,
                        "target_time": target_time,
                        "target_quote_time": s.time[target_idx], "mid": mid / 10000,
                        "target": np.sign(delta).astype(int) + 1,
                        "future_mid_return_bps": delta / mid * 10000})
    out["spread_bps"] = (ask[:, 0] - bid[:, 0]) / mid * 10000
    for n in (1, 5, 10):
        qb, qa = bq[:, :n].sum(axis=1), aq[:, :n].sum(axis=1)
        out[f"imbalance_{n}"] = np.divide(qb-qa, qb+qa, out=np.zeros_like(qb), where=(qb+qa)>0)
    microprice = (ask[:, 0]*bq[:, 0] + bid[:, 0]*aq[:, 0]) / (bq[:, 0]+aq[:, 0])
    out["microprice_bps"] = (microprice-mid)/mid*10000
    out["log_depth"] = np.log1p(aq.sum(axis=1)+bq.sum(axis=1))
    normalizer = np.maximum(aq[:, 0]+bq[:, 0], 1)
    ofi = ofi_events(s.book)
    ofi[~s.valid] = 0
    out["ofi_500ms"] = window_sum(s.time, ofi, grid, .5, idx)/normalizer
    out["ofi_5s"] = window_sum(s.time, ofi, grid, 5, idx)/normalizer
    executed = np.where(np.isin(s.event, [4, 5]), s.size, 0)
    signed = -s.direction * executed
    total = window_sum(s.time, executed, grid, 5, idx)
    out["trade_imbalance_5s"] = window_sum(s.time, signed, grid, 5, idx)/np.maximum(total, 1)
    out["event_rate_5s"] = window_sum(s.time, np.ones(len(s.time)), grid, 5, idx)/5
    out["return_500ms_bps"] = (mid/mid_raw[lag_idx]-1)*10000
    out["return_5s_bps"] = (mid/mid_raw[old_idx]-1)*10000
    valid = ok & target_ok & old_ok & lag_ok
    valid &= (s.segment[idx] == s.segment[target_idx]) & (s.segment[idx] == s.segment[old_idx])
    valid &= (target_time > grid) & (target_time+.2 < c["end_seconds"])
    valid &= np.isfinite(out[FEATURES]).all(axis=1).to_numpy()
    out = out.loc[valid].reset_index(drop=True)
    s.quality.update({"grid_rows": len(grid), "usable_observations": len(out), "excluded_grid_rows": int((~valid).sum())})
    return out


def split_observations(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, str]:
    out = frame.copy()
    dates = sorted(out.date.unique())
    # Purge through the longest sensitivity latency, not just the primary scenario.
    endpoint = (out.target_time if "target_time" in out else out.time + config["horizon_seconds"]) + .2
    if len(dates) >= 3:
        a = min(max(1, int(len(dates)*config["train_fraction"])), len(dates)-2)
        b = min(max(a+1, int(len(dates)*(config["train_fraction"]+config["validation_fraction"]))), len(dates)-1)
        out["split"] = np.where(out.date < dates[a], "train", np.where(out.date < dates[b], "validation", "test"))
        protocol = "held-out complete trading dates"
    else:
        a = config["start_seconds"] + (config["end_seconds"]-config["start_seconds"])*config["train_fraction"]
        b = config["start_seconds"] + (config["end_seconds"]-config["start_seconds"])*(config["train_fraction"]+config["validation_fraction"])
        out["split"] = np.where(out.time < a, "train", np.where(out.time < b, "validation", "test"))
        out = out.loc[~(((out.split == "train") & (endpoint >= a)) | ((out.split == "validation") & (endpoint >= b)))].copy()
        protocol = "within-session clock-time 60/20/20, purged at boundaries"
    return out.reset_index(drop=True), protocol


def walk_forward_folds(frame: pd.DataFrame, config: dict):
    """Expanding folds, with disjoint tests and validation-only selection."""
    dates = sorted(frame.date.unique())
    if len(dates) >= 5:
        initial = max(2, int(len(dates)*config["walk_forward_initial_train_fraction"]))
        validation_days = max(1, int(len(dates)*config["walk_forward_validation_fraction"]))
        for fold, test_index in enumerate(range(initial+validation_days, len(dates)), 1):
            test_date = dates[test_index]
            val_start = dates[test_index-validation_days]
            train = frame.loc[frame.date < val_start].copy()
            validation = frame.loc[(frame.date >= val_start) & (frame.date < test_date)].copy()
            test = frame.loc[frame.date == test_date].copy()
            yield fold, train, validation, test, "expanding complete-date walk-forward"
    elif len(dates) == 1:
        start, end = config["start_seconds"], config["end_seconds"]
        span = end-start
        initial = config["walk_forward_initial_train_fraction"]
        validation_fraction = config["walk_forward_validation_fraction"]
        test_fraction = config["walk_forward_test_fraction"]
        tests = int(round((1-initial-validation_fraction)/test_fraction))
        endpoint = frame.target_time+.2
        for fold in range(tests):
            val_start = start+span*(initial+fold*test_fraction)
            test_start = start+span*(initial+validation_fraction+fold*test_fraction)
            test_end = min(end, test_start+span*test_fraction)
            train = frame.loc[(frame.time < val_start) & (endpoint < val_start)].copy()
            validation = frame.loc[(frame.time >= val_start) & (frame.time < test_start) & (endpoint < test_start)].copy()
            test = frame.loc[(frame.time >= test_start) & (frame.time < test_end) & (endpoint < test_end)].copy()
            yield fold+1, train, validation, test, "expanding intraday walk-forward (one session)"
    else:
        raise ValueError("Walk-forward study needs one sample date or at least five complete dates; 2–4 dates are insufficient")
