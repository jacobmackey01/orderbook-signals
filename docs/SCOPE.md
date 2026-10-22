# Project scope: limit order book signals

## Objective

Build directly inspectable evidence of working with real tick-level limit order book data. The project should support a precise discussion of data handling, short-horizon signal research and execution assumptions.

## Research question

Can order-flow and depth imbalance predict the direction and magnitude of short-horizon mid-price moves? Does an XGBoost model add useful information beyond simple baselines, and does that information survive spread, fee and latency assumptions in a walk-forward backtest?

## Data and event resolution

- Five stocks: AAPL, AMZN, GOOG, INTC and MSFT, selected before looking at results.
- Official public LOBSTER ten-level samples for 21 June 2012, obtained directly from the provider. Approximately 2.1 million paired events across the five files.
- Process every supplied book event to calculate order-flow features. Make a decision after every twentieth event and predict 100 supplied book events ahead. This is an event-time horizon, not a fixed number of seconds. Report its realised wall-clock duration.
- Use the exact source row as the information boundary, including events sharing a timestamp. Do not treat events later in the file as available at the decision merely because their timestamp matches.
- Retain an optional fixed-clock sampling mode for comparison, but publish the event-time study as the main result.
- Validate and replay LOBSTER’s already reconstructed snapshots. Do not claim an original NASDAQ ITCH reconstruction engine, individual queue positions or consolidated-market data.

## Signals and models

Features include top-of-book order-flow imbalance, multi-level depth imbalance, microprice displacement, spread, signed execution flow and trailing event intensity. Feature calculations are backward-looking and reset around invalid book states, auction markers and trading halts.

Compare a training-majority classifier, a fixed imbalance rule, standardised logistic regression and XGBoost. Keep model settings fixed. Use validation data to select the model and any trading-confidence threshold; retain a no-trade option.

## Walk-forward evaluation

For the public single-session study, start with 40% of the clock-time interval for training, 10% for validation and the following 10% for test. Expand the training window by 10% and repeat, producing five chronological out-of-sample test windows. Purge labels and execution endpoints that reach the next partition. The first test window starts halfway through the session. Later training may incorporate earlier completed test windows, as in a running research process; no current or future test outcomes enter a fold’s model or threshold choice.

With enough additional sessions, split on complete dates and advance the same expanding train/validation/test structure across days. The bundled data can demonstrate intraday walk-forward behaviour only.

Report:

- Hit rate: directional accuracy on observations where the predicted class is up or down, including flat realised moves as misses; report coverage and sample count. Also retain ordinary three-class accuracy and macro-F1 so abstention and class imbalance remain visible.
- Information coefficient: Spearman rank correlation between the continuous directional score `P(up) - P(down)` and the future mid-price return. For deterministic baselines, use their signed class score; flag constant-score IC as undefined.
- P&L after costs: one-share quote-based long/short backtest, crossing the spread at entry and exit, with no overlapping positions per symbol. Primary scenario: 50 ms latency and illustrative 1 bp fee per side. Report 0/50/200 ms latency and 0/1/5 bp fee sensitivities.
- Results for every model and fold, plus stitched out-of-sample predictions and cumulative one-share P&L. The chosen-model series uses each fold’s validation-selected model, never the best model selected after viewing test results.
- Paired time-block bootstrap uncertainty for model improvement and IC; treat it as within-session uncertainty, not evidence of persistence across days.

## GitHub deliverable

A packaged Python library and CLI; source manifest and file hashes; meaningful unit tests; a fixed research plan and experiment log; an executed notebook with figures; machine-readable metrics; and a concise README explaining the findings and their limits. Raw source data and individual quote/trade records are excluded from Git. The downloader retrieves the public samples separately.

## Boundaries

No live trading, accounts, order submission or paid data purchase is part of this build. Public GitHub publication came later and excludes raw data, row-level outputs and models trained on licensed data. The backtest is an execution scenario using historical displayed quotes, with no guaranteed fills, short availability, market impact or queue model. A positive result would require multi-day and execution follow-up; a negative result is retained rather than tuned away.
