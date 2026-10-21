"""Figures and a research note generated from the saved walk-forward results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

COLORS = {"majority": "#6b7280", "imbalance_rule": "#b8860b", "logistic": "#2563a6", "xgboost": "#bd5b32", "validation_selected_model": "#202a35"}
LABELS = {"majority": "Majority", "imbalance_rule": "Imbalance rule", "logistic": "Logistic", "xgboost": "XGBoost", "validation_selected_model": "Validation-selected"}
SYMBOL_COLORS = ["#2563a6", "#a67c00", "#bd5b32", "#727d35", "#ad5576"]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.labelcolor": "#202a35", "text.color": "#202a35",
        "axes.titleweight": "bold", "axes.titlesize": 13, "figure.facecolor": "white", "savefig.facecolor": "white"})


def selected_economics(root):
    r = root/"results"
    decisions = pd.read_csv(r/"validation-decisions.csv")
    econ = pd.read_csv(r/"execution-sensitivity.csv")
    return econ.merge(decisions.loc[decisions.selected_model, ["symbol", "fold", "model"]], on=["symbol", "fold", "model"])


def build_figures(root: Path):
    style()
    r = root/"results"
    f = r/"figures"
    f.mkdir(exist_ok=True)
    metrics = pd.read_csv(r/"classification.csv")
    run = json.loads((r/"run.json").read_text())
    config = run["config"]
    period = ", ".join(run["session_dates"]) if len(run["session_dates"]) <= 2 else f"{run['session_dates'][0]} to {run['session_dates'][-1]}"
    sampling = f"decisions every {config['event_stride']} events · target {config['horizon_events']} events ahead" if config["sampling"] == "event" else f"{config['grid_seconds']} s decisions · {config['horizon_seconds']} s target"
    symbols = sorted(metrics.symbol.unique())
    models = ["majority", "imbalance_rule", "logistic", "xgboost"]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    x = np.arange(len(symbols))
    for i, model in enumerate(models):
        values = metrics.loc[metrics.model == model].set_index("symbol").loc[symbols, "macro_f1"]
        ax.bar(x+(i-1.5)*.19, values, .18, label=LABELS[model], color=COLORS[model])
    ax.set(xticks=x, xticklabels=symbols, ylabel="Macro-F1 (down / unchanged / up)", ylim=(0, 1),
           title="Short-horizon prediction: stitched walk-forward test windows")
    ax.legend(ncols=4, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=.15)
    fig.suptitle(f"LOBSTER · {period} · {sampling}", fontsize=10)
    fig.savefig(f/"prediction-comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for i, model in enumerate(["imbalance_rule", "logistic", "xgboost", "validation_selected_model"]):
        d = metrics.loc[metrics.model == model].set_index("symbol").loc[symbols]
        defined = d.information_coefficient.notna()
        positions = x[defined]+(i-1.5)*.13
        values = d.loc[defined, "information_coefficient"]
        low, high = d.loc[defined, "ic_low"], d.loc[defined, "ic_high"]
        ax.errorbar(positions, values, yerr=[np.maximum(values-low, 0), np.maximum(high-values, 0)],
                    fmt=["s", "o", "^", "D"][i], capsize=3, label=LABELS[model], color=COLORS[model])
    ax.axhline(0, color="#454545", lw=1)
    ax.set(xticks=x, xticklabels=symbols, ylabel="Spearman rank IC", title="Directional score versus future mid-price return")
    ax.legend(frameon=False, ncols=2)
    ax.grid(axis="y", alpha=.15)
    fig.suptitle("95% intervals: 60-second block bootstrap within the observed sessions", fontsize=10)
    fig.savefig(f/"information-coefficient.png", dpi=160)
    plt.close(fig)

    folds = pd.read_csv(r/"classification-by-fold.csv")
    folds = folds.loc[(folds.split == "test") & (folds.model.isin(["logistic", "xgboost"]))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, layout="constrained")
    for ax, model in zip(axes, ["logistic", "xgboost"]):
        for i, symbol in enumerate(symbols):
            d = folds.loc[(folds.symbol == symbol) & (folds.model == model)]
            ax.plot(d.fold, d.macro_f1, marker=["o", "s", "^", "D", "v"][i],
                    linestyle=["-", "--", "-.", ":", "-"][i], label=symbol, color=SYMBOL_COLORS[i])
        ax.set(title=LABELS[model], xlabel="Walk-forward test window", xticks=sorted(folds.fold.unique()), ylim=(0, 1))
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Test macro-F1")
    axes[1].legend(frameon=False, ncols=2)
    fig.suptitle("Model performance across expanding walk-forward folds", fontsize=14, fontweight="bold")
    fig.savefig(f/"walk-forward-stability.png", dpi=160)
    plt.close(fig)

    econ = pd.read_csv(r/"execution-sensitivity.csv")
    diagnostic = econ.loc[(econ.policy == "fixed_0.55_diagnostic") & (econ.latency_ms == 50)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True, layout="constrained")
    for ax, model in zip(axes, ["logistic", "xgboost"]):
        for i, symbol in enumerate(symbols):
            d = diagnostic.loc[(diagnostic.model == model) & (diagnostic.symbol == symbol)].groupby("fee_bps_per_side")[["total_net_dollars", "trades"]].sum()
            # Mean bps is weighted by trade count; dollars are not a bps denominator.
            original = diagnostic.loc[(diagnostic.model == model) & (diagnostic.symbol == symbol)].copy()
            original["weighted_bps"] = original.mean_net_bps.fillna(0)*original.trades
            sums = original.groupby("fee_bps_per_side")[["weighted_bps", "trades"]].sum()
            mean = sums.weighted_bps/sums.trades.replace(0, np.nan)
            ax.plot(mean.index, mean, marker=["o", "s", "^", "D", "v"][i], label=symbol, color=SYMBOL_COLORS[i])
        ax.axhline(0, color="#454545", lw=1)
        ax.set(title=LABELS[model], xlabel="Illustrative fee per side (basis points)", xticks=[0, 1, 5])
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Mean net bps per trade, including spread")
    axes[1].legend(frameon=False, ncols=2)
    fig.suptitle("Execution-cost sensitivity · fixed 0.55 confidence diagnostic · 50 ms latency", fontsize=12, fontweight="bold")
    fig.savefig(f/"cost-sensitivity.png", dpi=160)
    plt.close(fig)

    trades = pd.read_csv(r/"primary-scenario-trades.csv")
    decisions = pd.read_csv(r/"validation-decisions.csv")
    chosen = trades.loc[trades.policy == "validation_selected"].merge(
        decisions.loc[decisions.selected_model, ["symbol", "fold", "model"]], on=["symbol", "fold", "model"])
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    held_out = pd.read_parquet(r/"predictions.parquet", columns=["date", "time", "split"])
    held_out = held_out.loc[held_out.split == "test"]
    bounds = pd.to_datetime(held_out.date)+pd.to_timedelta(held_out.time, unit="s")
    start, end = bounds.min(), bounds.max()
    def plot_pnl(d, label, color, linestyle):
        if len(d):
            stamps = pd.to_datetime(d.date)+pd.to_timedelta(d.exit_time, unit="s")
            curve = d.assign(stamp=stamps).groupby("stamp").net_dollars.sum().sort_index().cumsum()
            curve = pd.concat([pd.Series([0.], index=[start]), curve])
            ax.plot(curve.index, curve.values, label=label, color=color, linestyle=linestyle)
        else:
            ax.plot([start, end], [0, 0], color=color, linestyle=linestyle, label=label+" (no trades)")
    plot_pnl(chosen, "Validation-selected policy", COLORS["validation_selected_model"], "-")
    for model, line in [("logistic", "--"), ("xgboost", "-.")]:
        d = trades.loc[(trades.model == model) & (trades.policy == "fixed_0.55_diagnostic")]
        plot_pnl(d, LABELS[model]+" · fixed 0.55 diagnostic", COLORS[model], line)
    ax.axhline(0, color="#454545", lw=1)
    ax.set(xlabel="Exchange-local time", ylabel="Cumulative one-share net P&L (USD, all five stocks)",
           title="Stitched walk-forward P&L after spread and fees", xlim=(start, end))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M" if held_out.date.nunique() == 1 else "%d %b"))
    ax.grid(alpha=.15)
    ax.legend(frameon=False, loc="lower left")
    fig.suptitle("Historical displayed-quote scenario · 50 ms latency · 1 bp fee per side + spread", fontsize=10)
    fig.savefig(f/"walk-forward-pnl.png", dpi=160)
    plt.close(fig)


def markdown_table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"]*len(columns)) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            cells.append("undefined" if pd.isna(v) else f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_findings(root: Path):
    r = root/"results"
    run = json.loads((r/"run.json").read_text())
    metrics = pd.read_csv(r/"classification.csv")
    chosen_metrics = metrics.loc[metrics.model == "validation_selected_model"]
    primary = selected_economics(root)
    primary = primary.loc[(primary.policy == "validation_selected") & (primary.latency_ms == 50) & (primary.fee_bps_per_side == 1)]
    primary_totals = primary.groupby("symbol")[["trades", "total_net_dollars"]].sum().reset_index()
    observation = pd.read_parquet(r/"observations.parquet")
    duration = observation.target_time-observation.time
    positive_improvement = 0
    for _, row in chosen_metrics.iterrows():
        baseline = metrics.loc[(metrics.symbol == row.symbol) & (metrics.model == "majority"), "macro_f1"].iloc[0]
        positive_improvement += int(row.macro_f1 > baseline)
    net = primary_totals.total_net_dollars.sum()
    ntrades = int(primary_totals.trades.sum())
    session_count = len(run['session_dates'])
    period = ', '.join(run['session_dates'])
    sampling_text = (f"makes a decision every {run['config']['event_stride']} supplied events and predicts {run['config']['horizon_events']} events ahead" if run['config']['sampling'] == 'event' else f"makes a decision every {run['config']['grid_seconds']} seconds and predicts {run['config']['horizon_seconds']} seconds ahead")
    text = f'''# Order book signals: walk-forward findings

The validation-selected models beat the training-majority baseline on stitched test macro-F1 for {positive_improvement} of {len(chosen_metrics)} stocks. Under the primary displayed-quote execution scenario, they generated {ntrades:,} one-share trades and a combined net P&L of ${net:,.2f}. These results cover {session_count} historical session(s) and do not establish a persistent trading edge.

## Data and research design

Official LOBSTER ten-level samples for AAPL, AMZN, GOOG, INTC and MSFT on {period} contain {run['total_source_rows']:,} paired book events. The pipeline retained {run['total_observations']:,} decision observations after the fixed filters. The main run {sampling_text}. Median realised horizon: {duration.median():.3f} seconds; 10th–90th percentile: {duration.quantile(.1):.3f}–{duration.quantile(.9):.3f} seconds. These are venue book-update events, not a consolidated tape of every market message.

The run uses {run['folds_per_symbol']} expanding walk-forward folds per stock ({'; '.join(run['protocols'])}). Each fold trains on earlier observations, reserves the next interval for model/threshold selection and tests on the following interval. Test windows are disjoint and purged through target time plus the longest latency scenario. Later folds may train on earlier completed test intervals. The model parameters and candidate thresholds were fixed before real-data performance was inspected. See [research plan](RESEARCH_PLAN.md), [scope](SCOPE.md) and [experiment log](EXPERIMENT_LOG.md).

## Predictive performance

Macro-F1 averages the F1 scores for down, unchanged and up. Hit rate below is directional correctness among predicted up/down observations, counting realised flat moves as misses; coverage prevents comparison with a model that rarely takes a direction from hiding abstention. Rank IC is Spearman correlation between P(up)−P(down) and future mid-price return. Constant scores have undefined IC.

{markdown_table(chosen_metrics, ['symbol','macro_f1','directional_hit_rate','directional_coverage','information_coefficient','ic_low','ic_high'])}

![Prediction comparison](../results/figures/prediction-comparison.png)

![Information coefficient](../results/figures/information-coefficient.png)

IC bounds resample 60-second clock blocks within the observed test series. They describe intraday sampling uncertainty, not robustness across trading days. All models, fold results, confusion matrices and paired model-minus-majority intervals are retained in `results/`. The five-stock comparisons are exploratory.

## P&L after spread and fee costs

The following table stitches the model selected by validation macro-F1 in each fold and that model's validation-selected threshold. Every executed position is one share. The scenario uses 50 ms latency, entry/exit across the observed spread and an illustrative fee of 1 bp per side.

{markdown_table(primary_totals, ['symbol','trades','total_net_dollars'])}

No-trade is a valid outcome of validation selection. A separate fixed 0.55 confidence diagnostic shows what execution costs do to directional trades; it was specified before inspecting test outcomes and is not a strategy selected on test P&L.

![Walk-forward P&L](../results/figures/walk-forward-pnl.png)

![Execution sensitivity](../results/figures/cost-sensitivity.png)

## Interpretation and limits

Prediction quality and economic value are different tests. The spread is paid through actual bid/ask prices in this diagnostic, then fees are deducted from both sides. Latency scenarios use quotes available at the assumed execution times. Trades cannot overlap within a symbol, but displayed historical liquidity does not guarantee an executable fill. Borrow availability, routing, hidden liquidity and impact are not modelled. LOBSTER has already reconstructed the books; this project validates and replays those states rather than claiming a full ITCH or queue reconstruction.

The sample contains five stocks across {session_count} historical session(s). Event dependence, limited regime coverage and historical market structure limit generalisation. Large row counts do not create independent trading days. The fee schedule is illustrative, not a verified account-specific charge. Next evidence should come from several untouched sessions and more realistic execution data, not repeated tuning of this test window.

## Reproduce and inspect

Run `./scripts/python -m lob_research.cli download`, `./scripts/python -m pytest -q` and `./scripts/python -m lob_research.cli run` from the repository root. Exact versions are in `requirements-lock.txt`; data hashes are in `data/sample-manifest.json`; configuration, source-code hashes, run timing and result hashes are in `results/run.json`. The executed notebook reads the pipeline results and checks them against saved predictions. Raw source files, predictions and individual trade records stay local and are excluded from Git.

Sources: [LOBSTER sample downloads](https://php.lobsterdata.com/info/DataSamples.php), [LOBSTER data structure](https://php.lobsterdata.com/info/DataStructure.php). All reported results are generated from the actual downloaded files.
'''
    (root/'docs/FINDINGS.md').write_text(text)
