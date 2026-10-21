"""Decompose the stitched test rank IC of the validation-selected model.

Diagnostic only: it reads saved results and the raw LOBSTER books and changes no
pipeline output. Writes results/diagnostics/ic-decomposition.csv (IC by subset,
with 60-second block-bootstrap intervals) and results/diagnostics/move-composition.csv
(what the future mid-price moves are made of).

Run from the repository root:  ./scripts/python scripts/ic_diagnostics.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lob_research.evaluation import block_ic_interval, information_coefficient

ROOT = Path(__file__).resolve().parents[1]
TICK = 100  # one cent in LOBSTER price units (dollars x 10000)


def raw_top_of_book(root: Path, symbol: str, date: str) -> np.ndarray:
    path = next((root / "data/raw").glob(f"{symbol}_{date}_*_orderbook_10.csv"))
    return pd.read_csv(path, header=None, usecols=[0, 2]).to_numpy(dtype=float)  # ask1, bid1


def raw_times(root: Path, symbol: str, date: str) -> np.ndarray:
    path = next((root / "data/raw").glob(f"{symbol}_{date}_*_message_10.csv"))
    return pd.read_csv(path, header=None, usecols=[0]).to_numpy(dtype=float)[:, 0]


def non_overlapping(frame: pd.DataFrame) -> np.ndarray:
    """Greedy subset whose [source_row, target_row] label paths never overlap."""
    keep, last_target = [], -1
    for i, (src, tgt) in enumerate(zip(frame.source_row.to_numpy(), frame.target_row.to_numpy())):
        if src >= last_target:
            keep.append(i)
            last_target = tgt
    mask = np.zeros(len(frame), dtype=bool)
    mask[keep] = True
    return mask


def main(root: Path = ROOT):
    config = json.loads((root / "configs/study.json").read_text())
    predictions = pd.read_parquet(root / "results/predictions.parquet")
    observations = pd.read_parquet(root / "results/observations.parquet")
    test = predictions.loc[(predictions.split == "test") & (predictions.model == "validation_selected_model")]
    test = test.merge(observations[["symbol", "date", "source_row", "imbalance_1", "imbalance_5", "microprice_bps",
                                    "return_500ms_bps"]], on=["symbol", "date", "source_row"], how="left", validate="1:1")
    ic_rows, composition = [], []
    for (symbol, date), d in test.groupby(["symbol", "date"]):
        d = d.sort_values("source_row").reset_index(drop=True)
        top = raw_top_of_book(root, symbol, date)
        times = raw_times(root, symbol, date)
        # Decision rows followed by another event with the same timestamp (e.g. mid-sweep).
        mid_burst = times[d.source_row + 1] == times[d.source_row]
        ask0, bid0 = top[d.source_row, 0], top[d.source_row, 1]
        ask1, bid1 = top[d.target_row, 0], top[d.target_row, 1]
        spread0, spread1 = (ask0 - bid0) / TICK, (ask1 - bid1) / TICK
        dmid = ((ask1 + bid1) - (ask0 + bid0)) / 2 / TICK  # in ticks
        # Independent reconciliation with the saved target.
        recomputed = dmid * TICK / ((ask0 + bid0) / 2) * 10000
        assert np.allclose(recomputed, d.future_mid_return_bps, atol=1e-9), symbol
        zero = dmid == 0
        half = np.isclose(np.abs(dmid) % 1, .5)
        spread_change = ~np.isclose(spread0, spread1)
        one_tick = np.isclose(spread0, 1)
        both_sides = (ask1 != ask0) & (bid1 != bid0) & (np.sign(ask1 - ask0) == np.sign(bid1 - bid0))
        nonzero = ~zero
        composition.append({
            "symbol": symbol, "n": len(d), "share_zero_return": zero.mean(),
            "share_spread_1_tick_at_decision": one_tick.mean(),
            "median_spread_ticks": float(np.median(spread0)),
            "nonzero_share_half_tick_multiple": half[nonzero].mean(),
            "nonzero_share_spread_changed": spread_change[nonzero].mean(),
            "nonzero_share_both_quotes_shifted_same_way": both_sides[nonzero].mean(),
            "nonzero_median_abs_move_ticks": float(np.median(np.abs(dmid[nonzero]))),
            "share_abs_move_ge_spread": (np.abs(dmid) >= spread0).mean(),
            "nonzero_sign_hit_rate_of_score": float((np.sign(d.score[nonzero]) == np.sign(dmid[nonzero])).mean()),
            "share_decisions_mid_timestamp_burst": float(mid_burst.mean()),
            "share_horizon_over_60s": float(((d.target_time - d.time) > 60).mean()),
        })
        subsets = {
            "all": np.ones(len(d), bool),
            "nonzero_return": nonzero,
            "spread_1_tick": one_tick,
            "spread_wider_than_1_tick": ~one_tick,
            "nonzero_spread_unchanged": nonzero & ~spread_change,
            "nonzero_spread_changed": nonzero & spread_change,
            "non_overlapping_greedy": non_overlapping(d),
            "every_5th_observation": np.arange(len(d)) % 5 == 0,
            "decision_at_end_of_timestamp": ~mid_burst,
            "decision_mid_timestamp_burst": mid_burst,
        }
        scores = {"model_score": d.score.to_numpy(), "microprice_bps_only": d.microprice_bps.to_numpy(),
                  "imbalance_1_only": d.imbalance_1.to_numpy(), "imbalance_5_only": d.imbalance_5.to_numpy(),
                  "return_500ms_only": d.return_500ms_bps.to_numpy()}
        returns = d.future_mid_return_bps.to_numpy()
        for score_name, score in scores.items():
            for name, mask in subsets.items():
                if score_name != "model_score" and name not in ("all", "nonzero_return", "spread_1_tick", "spread_wider_than_1_tick"):
                    continue
                ic = information_coefficient(score[mask], returns[mask])
                interval = block_ic_interval(score[mask], returns[mask], d.time.to_numpy()[mask],
                                             d.date.to_numpy()[mask], config) if mask.sum() >= 30 else {"ic_low": None, "ic_high": None}
                ic_rows.append({"symbol": symbol, "score": score_name, "subset": name, "n": int(mask.sum()),
                                "share_of_rows": float(mask.mean()), "ic": ic, **interval})
        # Same IC with an i.i.d. (one-observation block) bootstrap, to show what the 60 s blocks add.
        iid = block_ic_interval(d.score.to_numpy(), returns, np.arange(len(d), dtype=float), d.date.to_numpy(),
                                {**config, "bootstrap_block_seconds": 1})
        ic_rows.append({"symbol": symbol, "score": "model_score", "subset": "all_iid_bootstrap", "n": len(d),
                        "share_of_rows": 1.0, "ic": information_coefficient(d.score, returns), **iid})
        # Pooled per-fold IC: removes between-fold shifts in score level.
        fold_ic = [information_coefficient(g.score, g.future_mid_return_bps) for _, g in d.groupby("fold")]
        ic_rows.append({"symbol": symbol, "score": "model_score", "subset": "mean_of_per_fold_ic", "n": len(d),
                        "share_of_rows": 1.0, "ic": float(np.nanmean([x for x in fold_ic if x is not None])),
                        "ic_low": None, "ic_high": None})
    out = root / "results/diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    ic = pd.DataFrame(ic_rows)
    ic["ci_width"] = ic.ic_high - ic.ic_low
    ic.to_csv(out / "ic-decomposition.csv", index=False, float_format="%.4f")
    pd.DataFrame(composition).to_csv(out / "move-composition.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(composition).round(3).to_string(index=False))
    print(ic.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
