"""Figures for the crypto replication, in the equity study's visual style."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..presentation import style, SYMBOL_COLORS


def _phases(path: Path):
    """Accept a phase folder or the parent folder holding development/ and heldout/."""
    if (path/"by-date.csv").exists():
        return {path.name: path}
    return {p.name: p for p in sorted(path.iterdir()) if (p/"by-date.csv").exists()}


def build_figures(path: Path):
    style()
    phases = _phases(path)
    by_date = pd.concat([pd.read_csv(p/"by-date.csv").assign(phase=name) for name, p in phases.items()], ignore_index=True)
    by_date = by_date.sort_values(["symbol", "test_date"])
    config = json.loads(next(iter(phases.values())).joinpath("run.json").read_text())["config"]
    out = path/"figures"
    out.mkdir(exist_ok=True)
    symbols = sorted(by_date.symbol.unique())
    dates = sorted(by_date.test_date.unique())
    x = np.arange(len(dates))
    heldout = [i for i, d in enumerate(dates) if d in config["heldout_dates"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, column, label in [(axes[0], "delta_macro_f1", "Macro-F1 minus majority"),
                              (axes[1], "information_coefficient", "Spearman rank IC")]:
        if heldout:
            ax.axvspan(min(heldout)-.5, max(heldout)+.5, color="#e8edf3", zorder=0, label="Held-out dates")
        for i, symbol in enumerate(symbols):
            d = by_date.loc[by_date.symbol == symbol].set_index("test_date").reindex(dates)
            ax.plot(x, d[column], marker=["o", "s"][i % 2], linestyle=["-", "--"][i % 2], color=SYMBOL_COLORS[i], label=symbol)
        ax.axhline(0, color="#454545", lw=1)
        ax.set(xticks=x, xticklabels=[d[2:7] for d in dates], ylabel=label, xlabel="Test date (YY-MM, first of month)")
        ax.grid(axis="y", alpha=.15)
    axes[0].set_title("Validation-selected model vs majority")
    axes[1].set_title("Directional score vs future mid return")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Binance spot via Tardis · one test date per fold · decisions every {config['event_stride']} events, "
                 f"target {config['horizon_events']} events ahead", fontsize=10)
    fig.savefig(out/"performance-by-date.png", dpi=160)
    plt.close(fig)

    trades = pd.concat([pd.read_csv(p/"primary-scenario-trades.csv").assign(phase=name) for name, p in phases.items()
                        if (p/"primary-scenario-trades.csv").exists()], ignore_index=True)
    fig, axes = plt.subplots(1, len(symbols), figsize=(12, 4.6), layout="constrained", squeeze=False)
    for ax, symbol in zip(axes[0], symbols):
        for model, policy, color, label in [("xgboost", "fixed_0.55_diagnostic", "#bd5b32", "XGBoost, fixed 0.55 diagnostic"),
                                            ("logistic", "fixed_0.55_diagnostic", "#2563a6", "Logistic, fixed 0.55 diagnostic")]:
            d = trades.loc[(trades.symbol == symbol) & (trades.model == model) & (trades.policy == policy)].sort_values(["date", "time"])
            if len(d):
                ax.plot(np.arange(1, len(d)+1), d.net_bps.cumsum()/100, color=color, label=label)
        selected = by_date.loc[by_date.symbol == symbol]
        ax.set(title=f"{symbol}: validation-selected policy {int(selected.primary_trades.sum())} trades",
               xlabel="Trade number across test dates", ylabel="Cumulative net % of notional (one position at a time)")
        ax.axhline(0, color="#454545", lw=1)
        ax.grid(alpha=.15)
        ax.legend(frameon=False, loc="lower left")
    fig.suptitle(f"Primary scenario: {config['latency_ms']} ms latency, cross the spread, {config['fee_bps_per_side']:g} bp taker fee per side",
                 fontsize=10)
    fig.savefig(out/"cumulative-net-pnl.png", dpi=160)
    plt.close(fig)
