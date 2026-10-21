# Extension plan: multi-session crypto replication

Dated 2026-10-22. Written and committed before any model has been fitted to, or any feature computed from, the new data. This is a fixed analysis plan, not an externally registered study. Changes after a look at results go in `EXPERIMENT_LOG.md` under "Extension".

## Question

The equity study (LOBSTER, five NASDAQ stocks, one session, 2012-06-21) found that order-book features improve short-horizon direction prediction over a majority baseline, while the validation-selected trading policies abstain once spread and fees are included. Its main gap is that it rests on one session.

Does the same pattern appear across several independent sessions in a different market: (a) a predictive improvement over the baselines out of sample, on complete held-out dates, and (b) no trading policy that survives the spread plus the venue's documented taker fee?

This is a cross-market replication of the *pattern*, not of the equity numbers. Results are reported separately and never pooled with the equity sample.

## Data

- Source: Tardis.dev downloadable CSV files, which are free without an API key or account for the first day of each month (see `DATA_SOURCES.md` for URLs, terms, hashes). Data types: `incremental_book_L2` and `trades`.
- Venue: Binance spot. Instruments: BTCUSDT and ETHUSDT, chosen before any data were inspected because they are the two most liquid spot pairs and Binance publishes the lowest standard spot taker fee among the large venues, which gives the cost test its best chance.
- Dates (UTC calendar days, the twelve most recent free days): 2025-11-01, 2025-12-01, 2026-01-01, 2026-02-01, 2026-03-01, 2026-04-01, 2026-05-01, 2026-06-01, 2026-07-01, 2026-08-01, 2026-09-01, 2026-10-01.
- Development dates: the first eight (2025-11-01 to 2026-06-01).
- Held-out dates: **2026-07-01, 2026-08-01, 2026-09-01, 2026-10-01**. Their folds are run once, after the code, configuration and this plan are frozen in a commit whose hash is recorded in the held-out run's provenance.
- Disclosure: before this plan was written, the 2026-10-01 BTCUSDT files were downloaded once to measure file size and to count rows, messages and snapshot rows (26.1M level rows, 862k book messages, one snapshot, 3.0M trades). No feature, label or model output was computed from them.

Coinbase data are excluded: Tardis's terms add a Coinbase-specific prohibition on any machine-learning use.

## Cross-market differences (documented, not tuned)

- **Session/day boundary.** Crypto trades 24/7. A session is one UTC calendar day, 00:00:00–24:00:00, by Tardis `local_timestamp` (arrival time at Tardis's Tokyo recorder). Tardis re-subscribes at 00:00 UTC, so every daily file begins with a fresh book snapshot. Decisions are allowed over the whole day; targets and the 200 ms purge must end before 24:00:00. Dates are a month apart, so dates are independent sessions in the calendar sense, though crypto regimes persist.
- **Tick size.** Both pairs have a 0.01 USDT tick (Binance `exchangeInfo`, retrieved 2026-10-22). At BTC ≈ 84,000 USDT this is ~0.001 bp, far below the equity stocks' one-cent tick relative to price. The spread is usually one tick, so the spread feature is nearly constant and spread-crossing cost is negligible; the fee dominates.
- **Fees.** Binance spot Regular user (VIP 0) taker and maker commission: 0.1000% (10 bp) per side, from Binance's public fee schedule (https://www.binance.com/en/fee/schedule; its public data endpoint `bapi/accounts/v1/public/account/trade-level/get`, retrieved 2026-10-22, level 0 `takerCommission: 0.001`). Primary scenario: 10 bp per side. Sensitivity: 0, 1, 5, 7.5 and 10 bp per side (7.5 bp corresponds to Binance's advertised 25% BNB fee discount, not verified here; 0 bp isolates the spread).
- **No halts or auctions.** Continuous trading; the halt/auction logic of the equity study has no counterpart. Recording gaps take its place (below).
- **Event resolution.** The equity study used LOBSTER's per-message book events. Binance's diff-depth stream (`depth@100ms`, as recorded by Tardis) publishes *netted level changes every 100 ms*, not individual order events. Trades are published one per fill. This is therefore **not complete event-level order flow**: individual add/cancel messages within a 100 ms interval are invisible, and cross-channel ordering between book and trade messages is not guaranteed by the exchange. Order-flow imbalance is computed from successive 100 ms book states.
- **Quote currency.** P&L is in USDT, not USD.

## Event definition and book reconstruction

Follow Tardis's documented `incremental_book_L2` reconstruction:

1. Skip rows before the first `is_snapshot=true` row.
2. Rows sharing a `local_timestamp` form one WebSocket message; the book is read as consistent only after the whole group has been applied.
3. A snapshot row following a non-snapshot row, or a snapshot with a different `local_timestamp` from the previous snapshot row, resets the book (all levels removed).
4. `amount` is the new level amount, not a delta; 0 removes the level.

The Binance `U`/`u` sequence numbers are not in the CSV. Tardis validates them during collection and restarts the connection on a missed message, which then appears in the CSV as a new snapshot. Gap handling here:

- **Snapshot reset mid-day** → treated as a recording gap: segment boundary; book rebuilt from the snapshot.
- **Silence**: more than 2.0 s between consecutive book messages → segment boundary at the first message after the silence (suspected gap; both pairs normally update every 100 ms).
- **Crossed or locked book** (best bid ≥ best ask) or fewer than ten levels on either side → invalid row and segment boundary. Crossed books are not repaired.
- Non-increasing `local_timestamp` across messages within a file → the run fails.

Features, targets and executions may not span a segment boundary (the existing `make_observations` requires the decision, the 5 s look-back row and the target row to share a segment, which covers all trailing windows). Gap counts are reported per date.

Events: one event row per book message (event type 1, after its level updates are applied) and one per trade (event type 4, carrying the latest consistent book). Book and trade streams are merged by `local_timestamp`; at equal timestamps the book message precedes the trade (a fixed, documented tie rule). A trade's resting-side direction is the negative of the Tardis aggressor side, so the existing signed-execution convention is unchanged.

Units: to reuse the one-share execution code, quantities are expressed in lots equal to Binance's quantity step (BTC 0.00001, ETH 0.0001), so every displayed level holds at least one lot, and book prices as USDT per lot × 10,000. The simulated position is one lot. Because one lot is economically tiny, P&L is reported as net basis points per trade and as cumulative net USDT per 10,000 USDT of notional per trade (the sum of per-trade net bp); neither depends on the lot.

## Features, target, horizon

The equity study's feature family, unchanged in definition and computed by the existing `make_observations`: spread (bp), 1/5/10-level imbalance, microprice displacement (bp), log depth, 500 ms and 5 s best-quote OFI, 5 s signed trade imbalance and event intensity, 500 ms and 5 s mid returns. Quote age limit 1 s.

Decision every 20 events; target = sign of the mid change 100 events ahead (down/unchanged/up), the same event-time settings as the equity study. The realised wall-clock horizon is reported. It will differ from the equity study's 3.2 s median.

## Walk-forward splits

The research plan's complete-date rule applied to the fixed twelve-date calendar: initial training max(2, floor(0.4 × 12)) = 4 dates, validation max(1, floor(0.1 × 12)) = 1 date, then one complete test date per fold, expanding the training set. Seven folds per instrument, testing 2026-04-01 through 2026-10-01. Folds testing development dates (2026-04-01, 05-01, 06-01) form the development phase; the four held-out folds are run once. Fold *k* trains on all dates before its validation date (including earlier test dates, as in a running process); model choice and the trading threshold use the validation date only. Day files are independent, so no observation's target crosses into another date.

## Models and selection

Unchanged from the equity study: majority, fixed five-level imbalance rule (±0.1), standardised multinomial logistic regression (C=1, max_iter=1000) and XGBoost (120 trees, depth 3, learning rate 0.05, min_child_weight 20, subsample 0.8, colsample 0.8, L2 5, one thread, seed 42). Fitted per instrument and fold. Model chosen by validation macro-F1 (ties to the fixed order). Logistic/XGBoost confidence threshold chosen from 0.45/0.55/0.65/0.75/0.85/1.01 by validation-only net P&L in the primary scenario, at least 30 trades unless no-trade. The existing `choose_threshold` ranks candidates by one-lot net USDT; within a single validation date the price changes by a few percent at most, so this is practically the same ranking as summed net bp. The pre-specified fixed 0.55 diagnostic is also reported.

## Metrics

Per instrument, by date (= by fold) and stitched over test dates: macro-F1 (primary), accuracy, balanced accuracy, directional hit rate with directional coverage and count, Spearman IC of P(up) − P(down) against the future mid return, log loss. Within-sample 60 s block-bootstrap intervals (500 resamples) for the stitched validation-selected series only (IC, and macro-F1 minus majority); these describe intraday sampling noise, not cross-day robustness. The number of test dates on which the model beats majority is reported as the cross-date evidence.

Execution: one-lot, non-overlapping, crossing the displayed best bid/ask at entry and exit, fees on both sides; latency 0/50/200 ms; fees 0/1/5/7.5/10 bp per side. Primary scenario 50 ms and 10 bp. Trades, total net bp (= net USDT per 10,000 USDT notional per trade, summed), mean net bp per trade, winning fraction; cumulative net P&L by date.

## Replication criteria (fixed now)

Evaluated on the held-out dates, per instrument:

- **Predictive pattern replicates** if the stitched validation-selected macro-F1 exceeds majority and the validation-selected IC is positive on at least 3 of the 4 held-out dates.
- **Economic pattern replicates** if the validation-selected policies' primary-scenario net P&L is ≤ 0 (including no trading).

Development-phase results are reported too but do not decide replication.

## Limits stated in advance

Twelve dates (four held out) is still limited evidence. Two instruments on one venue in one market regime. Event resolution is coarser than LOBSTER's. Displayed liquidity is not a guaranteed fill; no queue, impact or passive-fill model. Fees assume a VIP 0 account without discounts. Tardis records from Tokyo; latency scenarios are relative to that recorder's arrival times.
