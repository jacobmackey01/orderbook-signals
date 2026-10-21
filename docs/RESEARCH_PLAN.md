# Research plan

Version 2, written before inspecting real-data model performance. This is a fixed analysis plan, not an externally registered study. The user's scope revision replaced the initial single holdout with event-time walk-forward evaluation; see `EXPERIMENT_LOG.md`. Each run records this file's SHA-256 hash, its configuration, code hashes and input provenance.

## Question and population

Do observable LOB features improve prediction of short-horizon mid-price direction relative to majority and simple imbalance baselines? Does a model's predictive value survive a spread-crossing execution diagnostic?

Use all five official ten-level LOBSTER equity samples for 21 June 2012: AAPL, AMZN, GOOG, INTC and MSFT. Restrict decisions to 09:35–15:55 in exchange-local seconds since midnight. No symbols or periods are selected after seeing performance. Books are reconstructed by LOBSTER from NASDAQ TotalView-ITCH; the repository validates and replays these venue states, not consolidated NBBO or full queue priority.

## Observations and features

Process every supplied event. The main study makes a decision every 20 file events, using that exact post-event state, and targets the mid-price 100 file events later. These are book-update events, not necessarily all raw exchange feed messages. Report the distribution of actual wall-clock horizons. Do not read later equal-timestamp events when making the decision. The optional clock mode samples the latest available state every 500 ms with a one-second target.

Features: best-quote spread in basis points; level-one, five-level and ten-level size imbalance; microprice displacement in basis points; log depth; trailing 500 ms and five-second best-quote order-flow imbalance; five-second signed execution imbalance and event intensity; and trailing 500 ms/five-second mid-price returns. OFI handles best-price changes, not just displayed-size differences. Trade direction is the negative of the resting-order direction for execution messages.

Reject mismatched files, non-finite records and out-of-order timestamps. Preserve equal-timestamp events and file pairing. Exclude invalid/locked/crossed books, missing ten-level depth, stale quotes and targets or feature windows crossing invalid, auction or halt segments. Current features contain no future data. Targets are down/unchanged/up based on the exact future mid versus the decision mid, without future smoothing.

## Expanding walk-forward design

On one session, first train on the earliest 40% of the fixed interval, validate on the next 10% and test on the following 10%. Expand the training interval by 10% and repeat for five disjoint test windows. Purge train, validation and test observations whose target time plus 200 ms reaches the next partition boundary. Historical feature lookbacks may cross a boundary, as in a running system. Later folds may include earlier completed test windows in training. Model settings are never changed using test results.

For five or more additional common dates, initial training covers max(2, floor(40% of dates)), validation covers max(1, floor(10% of dates)), and one full date is tested at each successive step. Two to four dates are rejected rather than silently presenting an inadequate day-level walk-forward study. The bundled run provides only within-session evidence.

## Models and selection

Fit separately per stock and fold:

- Majority class, with class frequencies estimated on training.
- Fixed five-level imbalance rule: down below -0.1, unchanged from -0.1 to 0.1, up above 0.1.
- Standardised multinomial logistic regression: C=1, max_iter=1000; scaler fitted only on training.
- XGBoost: 120 trees, depth 3, learning rate 0.05, min_child_weight 20, subsample 0.8, colsample_bytree 0.8, L2=5, one thread, seed 42. No early stopping or hyperparameter search.

Select each fold's candidate model by validation macro-F1; break ties by the fixed model order above. Publish all model results as well as the stitched validation-selected series. For logistic and XGBoost, select the trade-confidence threshold from 0.45/0.55/0.65/0.75/0.85/1.01 using validation-only total one-share net dollars, at least 30 trades except the explicit no-trade candidate. Break ties towards fewer trades. A directional probability must exceed both the threshold and the flat probability. Baselines have fixed actions.

## Evaluation metrics

Primary predictive metric is macro-F1 over all three classes. Also report ordinary accuracy, balanced accuracy, class counts, confusion matrices and log loss for probabilistic models. Directional hit rate counts correct up/down predictions among all directional predictions; realised unchanged moves count as misses. Report directional coverage and count.

Information coefficient is Spearman rank correlation between P(up)-P(down) and future mid-price return; use the signed prediction for the deterministic imbalance rule. Constant-score IC is undefined. Report 95% intervals from 500 paired resamples of non-overlapping 60-second clock-time blocks for model-minus-majority macro-F1 and 500 block resamples for IC. These intervals quantify within-session sampling uncertainty; no cross-day or family-wise significance claim follows. Report every stock and test window.

## Quote-based execution diagnostic

One share per long or short position. Enter at the best ask for longs or bid for shorts at decision time plus latency. Exit on the opposite side at target-event time plus latency. At zero latency use the exact source and target file rows, not later events sharing those timestamps. Require displayed quantity of at least one share, quote age at most one second and an uninterrupted segment. Within each symbol, decisions at or before the previous exit time are skipped; this conservative tie rule avoids assuming a within-timestamp order of fills and new decisions. All test-fold exits precede that fold's end.

Primary scenario: 50 ms latency and illustrative fees of 1 bp per side, in addition to the observed spread. Sensitivities: 0/50/200 ms latency and 0/1/5 bp fee per side, using the same validation-selected threshold rather than choosing test winners. Also publish a fixed 0.55 directional-confidence diagnostic for logistic and XGBoost, specified before the first real-data run, so abstention does not hide execution costs. It is not a strategy selected on test outcomes.

Report trade count, coverage, mean net basis points per trade, one-share net dollars and winning-trade fraction. Stitch fold P&L using each fold's validation-selected model and threshold. No annualised Sharpe, leverage, market-impact, borrow, passive fill or queue model. Historical displayed liquidity is not a guaranteed executable quote. Fees are assumptions, not verified venue schedules.

## Interpretation and deliverables

Ship code, meaningful correctness tests, real-data outputs, a rerunnable executed notebook, figures and a concise findings note. Preserve negative results and no-trade policies. Keep raw data and row-level derived records out of Git; fetch samples separately. One historical session cannot establish persistent alpha or replace professional LOB experience. Expansion needs multiple untouched dates and stronger execution evidence. No live orders or data purchases are authorised.
