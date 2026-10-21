# Audit: LOB signal study (2012-06-21 LOBSTER samples)

Independent check of the code, saved results and claims, made before the project is used in applications. Audited commit: `9fbed03`. Scratch checks ran on the raw CSVs. The reproducible part of the IC analysis is in `scripts/ic_diagnostics.py`, which writes `results/diagnostics/ic-decomposition.csv` and `results/diagnostics/move-composition.csv`.

**Bottom line:** I found no material bug, so no pipeline result changed. The headline rank ICs are real out-of-sample associations, but most of the INTC/MSFT IC comes from spread-state mechanics: half-tick mid moves when a one-tick spread opens or closes, predicted by queue imbalance. The large-tick mechanics also dominate the economics. Even with zero fees and zero latency, the fixed 0.55 diagnostic loses about one spread per trade.

## 1. Confirmed issues and practical impact

None of these changes a reported number materially. All are disclosure or wording points for the write-up.

| # | Issue | Impact |
|---|---|---|
| 1 | **Decisions in the middle of an equal-timestamp burst.** Decisions fall every 20 file rows, so 3–11% of them land mid-burst, often partway through a multi-level sweep. Their IC is much higher (INTC 0.70, MSFT 0.76, AAPL 0.43). This follows the stated file-order information boundary, so it is not leakage. It is information nobody could act on, though. | Using only decisions at the last event of their timestamp lowers IC as follows: INTC 0.523→0.489, MSFT 0.561→0.533, AAPL 0.262→0.255, AMZN 0.323→0.315, GOOG 0.266→0.255. It does not affect P&L, because the primary scenario uses 50 ms latency. |
| 2 | **The stitched majority-baseline IC is defined, not "undefined".** The majority score is constant within each fold but differs between folds, so pooling the folds gives a small non-zero IC (−0.011 to 0.035, CIs spanning 0). This contradicts the FINDINGS line saying constant scores have undefined IC. | Cosmetic. Read the value as 0. The mean of the per-fold ICs for the selected model (0.264/0.314/0.256/0.508/0.567) is close to the stitched values, so pooling folds does not inflate the headline. |
| 3 | **Type-5 (hidden) execution sign cannot be verified for INTC/MSFT.** 99% of their hidden executions print exactly at the mid, so the sign rule −direction cannot be checked against price. For AAPL/AMZN/GOOG, 87–91% are on the side of the mid that the rule implies. | This adds noise to `trade_imbalance_5s` only, and hidden prints are about 11% of INTC/MSFT executions. It is not a leakage path. |
| 4 | **The plan says "exceed" the threshold, but the code uses `>=`.** | Immaterial with float probabilities. |

## 2. Checks that passed

- **Row alignment and price scale.** For every type-4 row (7,765–29,798 per stock), the message price equals the *previous* row's best quote on the resting side (100.0%, all five stocks). So each book row is the post-event state, as the code assumes. 100% of type-1 prices appear on the right side of the same row's book. Median mids are $583/$223/$570/$27.0/$30.5, consistent with the 1/10,000 scale.
- **Sample content.** The files contain no type 6 or type 7 rows, no dummy levels and no crossed or locked books. The halt, cross and dummy logic is correct by inspection and covered by unit tests, but the real data never exercises it. Decisions are restricted to 09:35–15:55 and the 5 s lookbacks stay inside that interval, so no window crosses the open or close.
- **Causality.** Features use rows `≤ source_row`: `window_sum` uses `right_indices+1`, and lag quotes use `searchsorted(..., 'right')` at t−0.5 and t−5. All 85,408 targets were recomputed independently from raw rows `source_row` and `target_row` and match exactly. No grid row is dropped by the `target_time > time` filter, so it causes no hidden look-ahead selection. The quote-age filter at t−5 s and t−0.5 s drops about 12% of grid rows using past information only.
- **OFI.** `ofi_events` matches the Cont–Kukanov–Stoikov e_n term by term, including best-price changes. Rows after an invalid state cannot enter a window, because of the segment check.
- **Execution signs.** Trade sign is −direction for types 4 and 5, matching the LOBSTER note that executing a sell limit order is a buyer-initiated trade. Hidden executions never change the visible book (0% of rows), as expected.
- **Splits.** Train, validation and test are purged at `target_time + 0.2 s`, which covers the 200 ms scenario. `split-counts.csv` shows ordered, disjoint windows. Test rows are unique per (date, source_row). The scaler sits inside the sklearn pipeline and is fitted on training rows only.
- **Selection.** The selected model is the first-maximum validation macro-F1 in the fixed order. I re-derived it for all 25 folds; there were no exact ties. Threshold selection uses validation P&L only, with ties broken toward fewer trades. Every fold chose 1.01, i.e. no trading.
- **Execution accounting.** Longs buy at the ask and sell at the bid. Fees are (entry + exit) × bp, and gross P&L = action × (exit − entry); both reconcile exactly in `primary-scenario-trades.csv`. No positions overlap. Exits precede each fold's end. A missing exit fails the run. Trade counts and net dollars reconcile with `execution-sensitivity.csv` for every symbol, fold, model and policy.
- **Saved outputs.** Recomputing macro-F1, hit rate, coverage and IC from `predictions.parquet` reproduces all rows of `classification.csv` and all 200 rows of `classification-by-fold.csv`, which match the FINDINGS table. All 23 unit tests pass.

## 3. IC investigation (selected model, stitched test windows)

| | AAPL | AMZN | GOOG | INTC | MSFT |
|---|---|---|---|---|---|
| Reported IC | 0.262 | 0.323 | 0.266 | 0.523 | 0.561 |
| Share of zero future returns | 3.6% | 5.5% | 2.8% | **75.6%** | **67.7%** |
| IC excluding zero returns | 0.268 | 0.329 | 0.268 | 0.681 | 0.664 |
| Decisions at a 1-tick spread | 0.7% | 0.4% | 0.1% | 79% | 78% |
| IC at a 1-tick spread / wider | n/a | n/a | n/a | 0.506 / 0.494 | 0.528 / 0.540 |
| Non-zero moves that are half-tick multiples | 52% | 53% | 54% | **82%** | **80%** |
| Non-zero moves where both quotes shift the same way | 68% | 61% | 67% | 18% | 20% |
| IC of L1 imbalance alone | 0.071 | 0.167 | 0.119 | **0.464** | **0.488** |
| IC of microprice displacement alone | 0.066 | 0.167 | 0.122 | 0.474 | 0.507 |
| IC of the 500 ms past return alone | 0.180 | 0.242 | 0.178 | 0.187 | 0.181 |
| IC, non-overlapping labels (greedy) / every 5th | 0.240 / 0.264 | 0.295 / 0.319 | 0.234 / 0.343 | 0.514 / 0.531 | 0.555 / 0.569 |
| Sign hit rate on non-zero moves | 0.59 | 0.63 | 0.61 | 0.90 | 0.88 |
| Share of moves with \|Δmid\| ≥ current spread | 16% | 6% | 9% | 4% | 6% |
| Fixed 0.55 diagnostic, 0 fee / 0 latency (mean bp per trade) vs median spread | −1.8 vs 2.6 | −4.7 vs 5.8 | −3.9 vs 5.1 | −3.3 vs 3.7 | −3.1 vs 3.3 |

**What drives it.** For INTC/MSFT the target is mostly a three-state queue outcome. At a one-tick spread, the mid moves only when one best queue depletes: the spread opens to two ticks and the mid moves by half a tick. 80–82% of non-zero moves are of this kind. L1 imbalance or microprice alone recovers about 90% of the model IC (0.46–0.51 vs 0.52–0.56). The model's sign is right on about 89% of non-zero moves. This is the well-known queue-depletion effect, i.e. real and valid predictability. Removing the zero-return rows *raises* IC, so the mass of ties dilutes Spearman rather than inflating it. Overlapping labels do not drive the point estimate: non-overlapping subsets give the same IC. For AAPL/AMZN/GOOG (12–26-tick spreads) almost every move is non-zero, and 94–97% of moves change the spread. Here the IC comes mainly from short-term mid continuation (500 ms return alone: 0.18–0.24) plus L1 imbalance. Again this is a quote inside a wide spread being followed by the other side, not a shift of the whole book.

**Why it is not tradable edge.** Only 4–16% of moves are at least as large as the spread you pay to cross. With no fees and no latency, the confident-signal diagnostic loses roughly 70–92% of the median spread per trade. The predictable part of the mid move is mostly smaller than the half-spread. The write-up should present the IC as "the models predict short-horizon queue and spread dynamics", not as alpha.

**Overlap and uncertainty.** The labels span 100 events and decisions come every 20, so neighbouring labels share about 80% of their path. The 60 s block bootstrap handles this: horizons over 60 s are ≤0.4% of rows. An i.i.d. bootstrap would give CIs about half as wide (e.g. MSFT width 0.025 vs 0.061). The block CI on the full sample is about as wide as the block CI on the non-overlapping subset with 1/5 of the rows (INTC 0.053 vs 0.059; MSFT 0.061 vs 0.068). So the stated intervals do not overstate precision within the session. They say nothing about other days.

## 4. Remaining uncertainties

- The data cover one session, and the halt, cross, dummy and crossed-book paths are untested on real data, because the sample has none of them.
- The direction of midpoint hidden executions cannot be verified (issue 3).
- The execution diagnostic uses venue-only displayed quotes, with no queue position, fill probability or NBBO. A passive or queue-aware strategy is the only plausible way to monetise queue-depletion predictability, and it is outside the current scope.
- The diagnostics use the saved run's predictions. I did not rerun the pipeline, because no fix required it.
