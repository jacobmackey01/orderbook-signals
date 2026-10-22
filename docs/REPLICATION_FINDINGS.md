# Cross-market study: Binance spot (crypto)

**On four held-out dates, BTCUSDT and ETHUSDT show a pattern similar to the equity study: the validation-selected model beats the majority baseline on every date, and validation never selects a trading policy once Binance's 10 bp taker fee is charged.** The headline predictability is materially affected by stale book updates in the 100 ms depth feed, and the results are conditional on complete recording windows (both below). This is a separate cross-market study with a feed-resolution limitation. It supports the equity pattern but does not confirm the equity numbers, and it is not pooled with the equity sample.

Plan: [EXTENSION_PLAN.md](EXTENSION_PLAN.md), committed before any model was fitted. Data, terms and hashes: [DATA_SOURCES.md](DATA_SOURCES.md). Deviations: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md), "Extension". The held-out folds ran once, from commit `39c1fde` (recorded in `results/crypto-binance/heldout/run.json`).

## Design in one paragraph

Tardis.dev free first-of-month files for Binance spot, 2025-11-01 to 2026-10-01 (12 UTC days, 114.1M merged book-message and trade events, 5.7M decisions). The book is rebuilt from `incremental_book_L2`, following Tardis's snapshot-and-update rules. Reconnect snapshots, silences longer than 2 s and invalid books start new segments, and features, targets and exits never cross a segment. The feature set, models, settings, selection rules and execution code are the equity study's, imported unchanged. Decisions are taken every 20 events, with the target 100 events ahead (median realised horizon 1.0 s for BTC and 1.4 s for ETH; the per-date median runs from 0.3 to 6.1 s). Expanding complete-date walk-forward: 4 training dates, 1 validation date, then one test date per fold. Folds testing 2026-04-01 to 06-01 are development folds; 2026-07-01 to 10-01 are held out.

## Results by test date (validation-selected series)

| symbol | test date | phase | model | macro-F1 | majority | hit rate | coverage | IC | IC, book-message rows | trades (primary) |
|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 2026-04-01 | dev | XGBoost | 0.686 | 0.164 | 0.734 | 0.746 | 0.753 | 0.414 | 0 |
| BTCUSDT | 2026-05-01 | dev | XGBoost | 0.718 | 0.220 | 0.699 | 0.547 | 0.757 | 0.399 | 0 |
| BTCUSDT | 2026-06-01 | dev | XGBoost | 0.667 | 0.198 | 0.652 | 0.767 | 0.732 | 0.439 | 0 |
| BTCUSDT | 2026-07-01 | held out | XGBoost | 0.665 | 0.164 | 0.723 | 0.716 | 0.717 | 0.432 | 0 |
| BTCUSDT | 2026-08-01 | held out | XGBoost | 0.744 | 0.269 | 0.687 | 0.324 | 0.695 | 0.312 | 0 |
| BTCUSDT | 2026-09-01 | held out | XGBoost | 0.679 | 0.206 | 0.671 | 0.637 | 0.727 | 0.372 | 0 |
| BTCUSDT | 2026-10-01 | held out | XGBoost | 0.697 | 0.197 | 0.722 | 0.589 | 0.750 | 0.401 | 0 |
| ETHUSDT | 2026-04-01 | dev | Logistic | 0.614 | 0.183 | 0.722 | 0.871 | 0.700 | 0.384 | 0 |
| ETHUSDT | 2026-05-01 | dev | XGBoost | 0.700 | 0.172 | 0.740 | 0.694 | 0.760 | 0.416 | 0 |
| ETHUSDT | 2026-06-01 | dev | XGBoost | 0.653 | 0.176 | 0.731 | 0.852 | 0.739 | 0.444 | 0 |
| ETHUSDT | 2026-07-01 | held out | XGBoost | 0.619 | 0.186 | 0.726 | 0.827 | 0.666 | 0.376 | 0 |
| ETHUSDT | 2026-08-01 | held out | XGBoost | 0.708 | 0.142 | 0.695 | 0.527 | 0.684 | 0.330 | 0 |
| ETHUSDT | 2026-09-01 | held out | XGBoost | 0.669 | 0.172 | 0.747 | 0.742 | 0.729 | 0.432 | 0 |
| ETHUSDT | 2026-10-01 | held out | XGBoost | 0.618 | 0.179 | 0.646 | 0.794 | 0.665 | 0.355 | 0 |

Hit rate counts realised flat moves as misses, and coverage is the share of decisions given a direction. IC is the Spearman correlation of P(up) − P(down) with the future mid return.

## Aggregate over the four held-out dates

| symbol | n | macro-F1 (majority) | Δ macro-F1, 95% block CI | hit rate / coverage | IC, 95% block CI | IC, book-message rows | dates beating majority | dates with IC > 0 |
|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 753,864 | 0.698 (0.199) | [0.495, 0.502] | 0.705 / 0.616 | 0.733 [0.729, 0.737] | 0.388 | 4/4 | 4/4 |
| ETHUSDT | 628,462 | 0.656 (0.175) | [0.478, 0.485] | 0.702 / 0.753 | 0.689 [0.684, 0.694] | 0.373 | 4/4 | 4/4 |

The development folds look the same (BTC 0.693, IC 0.740; ETH 0.656, IC 0.727), so the held-out dates show no sign of degradation. The block intervals resample 60 s blocks within the test dates. They measure intraday sampling noise, not robustness across days. The cross-day evidence is the 7 of 7 test dates per instrument (4 of 4 held out) on which the model beats majority, out of only 12 dates in total.

## Costs

- **Validation-selected policies: 0 trades** on all 14 instrument-folds. Every fold chose the no-trade threshold (1.01) on its validation date, so primary-scenario net P&L is 0.
- **Fixed 0.55 diagnostic, XGBoost, held out, 50 ms latency:**
  - BTC: 59,965 positions at −19.69 bp net each; +0.31 bp before fees.
  - ETH: 61,336 positions at −19.51 bp net each; +0.49 bp before fees.
  - Pre-fee edge falls as latency rises (0 / 50 / 200 ms: BTC +0.41 / +0.31 / +0.26 bp, ETH +0.62 / +0.49 / +0.35 bp).
  - Break-even would need a fee below about 0.15–0.25 bp per side, against 10 bp at VIP 0 and 7.5 bp with the BNB discount. Even the 1 bp sensitivity loses money.
- The spread is about 0.001 bp (one 0.01 USDT tick), so fees, not the spread, drive the result. In the equity study the spread and fees together did the same job.

## Verdict against the plan's criteria (fixed before fitting)

Predictive pattern: the plan's criteria are met for both instruments. Stitched held-out macro-F1 exceeds majority, and IC > 0 on 4 of 4 held-out dates; part of that IC reflects the stale-book effect below. Economic pattern: the plan's criterion is met. Net P&L is ≤ 0 because no trades were taken. The held-out evaluation was run once and is untouched in the sense of the plan. The changes made before it (exit-gap filter, row-type diagnostic) are logged.

## Limits

- **Stale-book artefact.** Binance's depth stream batches level changes every 100 ms, while trades stream one at a time. Most decisions fall on trade rows (74–83% on most test dates; about 49–50% on the quiet 2026-08-01), which carry the previous book message. There, IC is ~0.73. At freshly updated book-message rows it is ~0.37–0.39 on the held-out dates, which is comparable to the equity study's 0.26–0.56. Part of the "prediction" is the next book message catching up with trades already printed. The same staleness makes the displayed-quote execution diagnostic optimistic, so the small positive pre-fee edge should not be read as real.
- **Conditional on complete recording windows.** `require_priced_exits` drops decisions whose exit quote, at any tested latency (0/50/200 ms), would land in a different recording segment or on a quote older than 1 s. This uses knowledge of whether the future exit quote exists, so every result here is conditional on the recording being complete through the exit. It removed 34 decisions in total: 5 on development dates (BTC 3 and ETH 2 on 2026-06-01) and 29 on held-out dates (BTC 23 on 2026-07-01, ETH 6 on 2026-08-01), out of about 1.38 million held-out decisions. The effect on the metrics is negligible, but the simulation is not fully actionable as run: a live system would not know in advance that a feed gap was coming.
- **Not event-level order flow.** There are no individual add/cancel messages, and the exchange gives no cross-channel ordering guarantee (book first at equal timestamps is a fixed convention).
- **Limited evidence.** 12 dates, 4 held out, two correlated instruments on one venue in one year. First-of-month dates mix weekdays and weekends. 2026-06-01 had 18 reconnect snapshots.
- **No execution realism.** Fills are assumed at displayed quotes, with no queue, impact or maker-rebate model. Fees assume VIP 0. Latency is measured from Tardis's Tokyo recorder.
- **Terms.** Raw data, predictions, trade records and fitted-model weights (including the logistic coefficients) stay local under Tardis's licence. Only aggregate metrics and figures are committed.

## Outputs

`results/crypto-binance/{development,heldout}/`:
- `by-date.csv`
- `classification-by-fold.csv` (every model and fold)
- `classification.csv` (stitched; held-out phase = the four held-out dates)
- `pnl-aggregate.csv`, `execution-sensitivity.csv`
- `validation-decisions.csv`, `threshold-search.csv`
- `data-quality.csv`, `horizon-seconds.csv`, `split-counts.csv`, `confusion-matrices.csv`
- `run-start.json`, `run.json` (provenance: commit, config, plan and manifest hashes, package versions, source hashes, output hashes)

Figures for both phases: `results/crypto-binance/figures/performance-by-date.png` and `cumulative-net-pnl.png`.
