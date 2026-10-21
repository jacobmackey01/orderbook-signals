"""Multi-session crypto replication of the equity walk-forward study.

Reuses the equity feature, model, selection and execution code unchanged
(``make_observations``, ``evaluate_fold``). Only the data adapter, the fixed
calendar of complete dates and the reporting are new. Results are written to
``results/crypto-binance/<phase>/`` and are never pooled with the equity sample.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from ..data import Session, make_observations
from ..evaluation import classification_metrics, information_coefficient, block_ic_interval, paired_block_interval
from ..pipeline import evaluate_fold, validate_config
from .adapter import load_tardis_session
from .tardis import local_path

PHASES = ("development", "heldout")


def calendar_folds(frame: pd.DataFrame, calendar: list[str], config: dict, test_dates=None):
    """The research plan's complete-date expanding rule, laid out on a fixed calendar.

    Identical to ``data.walk_forward_folds`` when every calendar date is loaded, but lets
    the development phase run its folds without loading the held-out dates.
    """
    calendar = sorted(calendar)
    if len(calendar) < 5:
        raise ValueError("Complete-date walk-forward needs at least five dates")
    initial = max(2, int(len(calendar)*config["walk_forward_initial_train_fraction"]))
    validation_days = max(1, int(len(calendar)*config["walk_forward_validation_fraction"]))
    loaded = set(frame.date.unique())
    for fold, test_index in enumerate(range(initial+validation_days, len(calendar)), 1):
        test_date = calendar[test_index]
        if test_dates is not None and test_date not in test_dates:
            continue
        needed = set(calendar[:test_index+1])
        if not needed <= loaded:
            raise ValueError(f"Fold {fold} needs dates that are not loaded: {sorted(needed-loaded)}")
        val_start = calendar[test_index-validation_days]
        train = frame.loc[frame.date < val_start].copy()
        validation = frame.loc[(frame.date >= val_start) & (frame.date < test_date)].copy()
        test = frame.loc[frame.date == test_date].copy()
        yield fold, train, validation, test, "expanding complete-date walk-forward (fixed calendar)"


def require_priced_exits(session: Session, observations: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop decisions whose exit, at any sensitivity latency, falls in another segment or on a stale quote.

    Like the equity study's exclusion of targets that cross halts or invalid books, this is a
    label/exit availability filter applied identically to every model before any prediction. It
    prevents the fail-closed execution check from aborting when a recording gap (reconnect
    snapshot or silence) starts between the target event and target time plus latency.
    """
    keep = np.ones(len(observations), dtype=bool)
    decision_segment = session.segment[observations.source_row.to_numpy(dtype=int)]
    for latency in config.get("latency_sensitivity_ms", [0, 50, 200]):
        index, ok = session.at(observations.target_time.to_numpy()+latency/1000, config["max_quote_age_seconds"])
        keep &= ok & (session.segment[index] == decision_segment)
    session.quality["exit_gap_exclusions"] = int((~keep).sum())
    return observations.loc[keep].reset_index(drop=True)


def _cache_paths(root: Path, symbol: str, date: str):
    base = root/"data/tardis/processed"
    return base/f"{symbol}_{date}_observations.parquet", base/f"{symbol}_{date}_session.npz", base/f"{symbol}_{date}_quality.json"


def prepare(args):
    """Reconstruct one symbol-date, build observations and cache a slim session."""
    root, config, symbol, date = args
    obs_path, session_path, quality_path = _cache_paths(root, symbol, date)
    if obs_path.exists() and session_path.exists() and quality_path.exists():
        return symbol, date
    obs_path.parent.mkdir(parents=True, exist_ok=True)
    session = load_tardis_session(local_path(root, config["exchange"], "incremental_book_L2", date, symbol),
                                  local_path(root, config["exchange"], "trades", date, symbol),
                                  symbol, date, config["lot_size"][symbol], config["gap_seconds"],
                                  config["levels"], config["max_quote_age_seconds"])
    observations = require_priced_exits(session, make_observations(session, config), config)
    observations.to_parquet(obs_path, index=False)
    # Execution needs only level one; keep the cache small.
    np.savez(session_path, time=session.time, event=session.event, size=session.size, direction=session.direction,
             book=session.book[:, :4], valid=session.valid, segment=session.segment)
    quality_path.write_text(json.dumps({"symbol": symbol, "date": date, **session.quality}, indent=2)+"\n")
    print(f"Prepared {symbol} {date}: {session.quality['source_rows']:,} events, {len(observations):,} observations", flush=True)
    return symbol, date


def load_cached(root: Path, symbol: str, date: str):
    obs_path, session_path, quality_path = _cache_paths(root, symbol, date)
    z = np.load(session_path)
    quality = json.loads(quality_path.read_text())
    session = Session(symbol, date, z["time"], z["event"], z["size"], z["direction"], z["book"], z["valid"], z["segment"], quality)
    return pd.read_parquet(obs_path), session, quality


def phase_dates(config: dict, phase: str):
    calendar = sorted(config["dates"])
    heldout = sorted(config["heldout_dates"])
    if heldout != calendar[-len(heldout):]:
        raise ValueError("Held-out dates must be the final calendar dates")
    if phase == "development":
        loaded = calendar[:-len(heldout)]
        return loaded, [d for d in loaded if d in calendar[_first_test_index(config, calendar):]]
    if phase == "heldout":
        return calendar, heldout
    raise ValueError(f"Unknown phase {phase}")


def _first_test_index(config, calendar):
    return max(2, int(len(calendar)*config["walk_forward_initial_train_fraction"])) + \
        max(1, int(len(calendar)*config["walk_forward_validation_fraction"]))


def git_state(root: Path) -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tracked_files": [line for line in run("status", "--porcelain", "--untracked-files=no").splitlines() if line]}


def run(root: Path, config_path: Path | None = None, phase: str = "development", workers: int = 4):
    config_path = config_path or root/"configs/crypto_binance.json"
    config = json.loads(config_path.read_text())
    validate_config(config)
    loaded, test_dates = phase_dates(config, phase)
    out = root/"results/crypto-binance"/phase
    out.mkdir(parents=True, exist_ok=True)
    git = git_state(root)
    if phase == "heldout" and git["dirty_tracked_files"]:
        raise RuntimeError("Held-out evaluation must run from a committed, frozen tree")
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "phase": phase, "git": git,
                "loaded_dates": loaded, "test_dates": test_dates, "config": config,
                "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
                "plan_sha256": sha256((root/"docs/EXTENSION_PLAN.md").read_bytes()).hexdigest(),
                "manifest_sha256": sha256((root/"data/tardis-manifest.json").read_bytes()).hexdigest(),
                "python": platform.python_version(),
                "packages": {n: version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "matplotlib", "pyarrow"]},
                "source_code_sha256": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                                       for p in sorted((root/"src/lob_research").rglob("*.py"))}}
    (out/"run-start.json").write_text(json.dumps(metadata, indent=2)+"\n")
    jobs = [(root, config, s, d) for s in config["symbols"] for d in loaded]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(prepare, jobs))
    names = ["classification", "predictions", "economic", "trades", "decisions", "threshold_search", "confusion", "coefficients", "splits"]
    records = {name: [] for name in names}
    quality, horizons = [], []
    for symbol in config["symbols"]:
        frames, sessions = [], {}
        for date in loaded:
            obs, session, q = load_cached(root, symbol, date)
            frames.append(obs)
            sessions[date] = session
            quality.append(q)
        frame = pd.concat(frames, ignore_index=True).sort_values(["date", "time", "source_row"]).reset_index(drop=True)
        horizons.append(frame.assign(horizon_seconds=frame.target_quote_time-frame.source_time)[["symbol", "date", "horizon_seconds"]])
        for fold, train, val, test, protocol in calendar_folds(frame, config["dates"], config, test_dates):
            evaluate_fold(symbol, sessions, train, val, test, fold, config, records)
        del frames, sessions, frame
    write_outputs(out, config, records, quality, horizons, metadata)
    return metadata


def write_outputs(out: Path, config: dict, records: dict, quality: list, horizons: list, metadata: dict):
    pd.DataFrame(quality).to_csv(out/"data-quality.csv", index=False)
    h = pd.concat(horizons, ignore_index=True)
    h.groupby(["symbol", "date"]).horizon_seconds.describe(percentiles=[.1, .5, .9]).reset_index().to_csv(out/"horizon-seconds.csv", index=False)
    predictions = pd.concat(records["predictions"], ignore_index=True)
    predictions.to_parquet(out/"predictions.parquet", index=False)
    trades = pd.concat(records["trades"], ignore_index=True)
    trades.to_csv(out/"primary-scenario-trades.csv", index=False)
    splits = pd.DataFrame(records["splits"])
    splits.to_csv(out/"split-counts.csv", index=False)
    test_date = splits.loc[splits.split == "test"].set_index(["symbol", "fold"]).first_date
    by_fold = pd.DataFrame(records["classification"])
    by_fold["test_date"] = [test_date[(s, f)] for s, f in zip(by_fold.symbol, by_fold.fold)]
    by_fold.to_csv(out/"classification-by-fold.csv", index=False)
    economic = pd.DataFrame(records["economic"])
    economic["total_net_bps"] = economic.mean_net_bps.fillna(0)*economic.trades
    economic["test_date"] = [test_date[(s, f)] for s, f in zip(economic.symbol, economic.fold)]
    economic.to_csv(out/"execution-sensitivity.csv", index=False)
    for key, filename in [("decisions", "validation-decisions.csv"), ("threshold_search", "threshold-search.csv"),
                          ("confusion", "confusion-matrices.csv"), ("coefficients", "logistic-coefficients.csv")]:
        pd.DataFrame(records[key]).to_csv(out/filename, index=False)

    test = predictions.loc[predictions.split == "test"]
    decisions = pd.DataFrame(records["decisions"])
    selected = decisions.loc[decisions.selected_model].set_index(["symbol", "fold"])
    # By-date summary of the validation-selected series against majority.
    by_date = []
    for (symbol, fold), p in test.loc[test.model == "validation_selected_model"].groupby(["symbol", "fold"]):
        base = test.loc[(test.model == "majority") & (test.symbol == symbol) & (test.fold == fold)]
        model = selected.loc[(symbol, fold)]
        m = classification_metrics(p.target, p.prediction, p[["p_down", "p_flat", "p_up"]].to_numpy(), model.model != "imbalance_rule")
        b = classification_metrics(base.target, base.prediction, base[["p_down", "p_flat", "p_up"]].to_numpy())
        primary = economic.loc[(economic.symbol == symbol) & (economic.fold == fold) & (economic.model == model.model)
                               & (economic.policy == "validation_selected") & (economic.latency_ms == config["latency_ms"])
                               & (economic.fee_bps_per_side == config["fee_bps_per_side"])].iloc[0]
        gross = economic.loc[(economic.symbol == symbol) & (economic.fold == fold) & (economic.model == model.model)
                             & (economic.policy == "validation_selected") & (economic.latency_ms == config["latency_ms"])
                             & (economic.fee_bps_per_side == 0)].iloc[0]
        by_date.append({"symbol": symbol, "fold": fold, "test_date": test_date[(symbol, fold)], "selected_model": model.model,
                        "threshold": model.threshold, "n": m["n"], "macro_f1": m["macro_f1"], "majority_macro_f1": b["macro_f1"],
                        "delta_macro_f1": m["macro_f1"]-b["macro_f1"], "directional_hit_rate": m["directional_hit_rate"],
                        "directional_coverage": m["directional_coverage"],
                        "information_coefficient": information_coefficient(p.score, p.future_mid_return_bps),
                        "flat_share": m["flat_n"]/m["n"],
                        "primary_trades": int(primary.trades), "primary_total_net_bps": float(primary.total_net_bps),
                        "primary_mean_net_bps": primary.mean_net_bps, "spread_only_trades": int(gross.trades),
                        "spread_only_mean_net_bps": gross.mean_net_bps})
    by_date = pd.DataFrame(by_date)
    by_date.to_csv(out/"by-date.csv", index=False)

    aggregate = []
    for (symbol, model), p in test.groupby(["symbol", "model"]):
        p = p.sort_values(["date", "time", "source_row"])
        if p.duplicated(["date", "source_row"]).any():
            raise ValueError("Walk-forward test windows overlap")
        calibrated = model not in ("imbalance_rule",) and not (model == "validation_selected_model" and (p.selected_model == "imbalance_rule").any())
        metric = classification_metrics(p.target, p.prediction, p[["p_down", "p_flat", "p_up"]].to_numpy(), calibrated)
        metric["information_coefficient"] = information_coefficient(p.score, p.future_mid_return_bps)
        if model == "validation_selected_model":
            base = test.loc[(test.symbol == symbol) & (test.model == "majority")].set_index(["date", "source_row"])
            base = base.loc[p.set_index(["date", "source_row"]).index]
            metric.update(block_ic_interval(p.score, p.future_mid_return_bps, p.time, p.date, config))
            metric.update(paired_block_interval(p.target.to_numpy(), p.prediction.to_numpy(), base.prediction.to_numpy(), p.time, p.date, config))
            d = by_date.loc[by_date.symbol == symbol]
            metric.update(dates=len(d), dates_beating_majority=int((d.delta_macro_f1 > 0).sum()),
                          dates_positive_ic=int((d.information_coefficient > 0).sum()))
        aggregate.append({"symbol": symbol, "model": model, "split": "stitched_test", **metric})
    pd.DataFrame(aggregate).to_csv(out/"classification.csv", index=False)
    stitched = economic.loc[economic.policy.isin(["validation_selected", "fixed_0.55_diagnostic"])]
    pnl = stitched.groupby(["symbol", "model", "policy", "latency_ms", "fee_bps_per_side"]).agg(
        trades=("trades", "sum"), total_net_bps=("total_net_bps", "sum"), total_net_usdt_per_lot=("total_net_dollars", "sum")).reset_index()
    sel = []
    for (symbol, fold), model in selected.model.items():
        rows = economic.loc[(economic.symbol == symbol) & (economic.fold == fold) & (economic.model == model)]
        sel.append(rows.assign(model_selected=model))
    sel = pd.concat(sel).groupby(["symbol", "policy", "latency_ms", "fee_bps_per_side"]).agg(
        trades=("trades", "sum"), total_net_bps=("total_net_bps", "sum")).reset_index().assign(model="validation_selected_model")
    pd.concat([pnl, sel], ignore_index=True).to_csv(out/"pnl-aggregate.csv", index=False)
    metadata.update(completed_utc=datetime.now(timezone.utc).isoformat(),
                    total_source_rows=int(sum(q["source_rows"] for q in quality)),
                    total_observations=int(len(h)),
                    outputs_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.csv"))})
    (out/"run.json").write_text(json.dumps(metadata, indent=2, default=str)+"\n")
    from .figures import build_figures
    build_figures(out)
