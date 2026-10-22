# Validation record

The final real-data run completed for five stocks and five expanding walk-forward windows per stock. The configuration, research plan, Python versions, source-code hashes and CSV hashes match `results/run.json`.

## Executed checks

- 23 unit tests passed. They cover paired-file failures, out-of-order data, equal-timestamp events, dummy/crossed depth, halts, OFI price resets, causal feature windows, resting-order execution signs, split purging, disjoint expanding test windows, train-only scaling, exact event horizons, entry/exit prices, spread and fee arithmetic, no overlapping positions, abstention, missing exits and undefined IC.
- The full pipeline processed 2,110,860 real source rows and retained 85,408 decision observations.
- An independent raw-CSV calculation reconciled all 85,408 current mids, future returns and target classes to the exact source/target book rows.
- The final notebook executed top to bottom without cell errors. It independently recomputes macro-F1 and IC from predictions, checks probability sums and disjoint test rows, reconciles selected-policy P&L to individual trades and checks non-overlap and fee accounting.
- All five exported figures were inspected for visible values, labels, units and layout. The P&L figure includes the no-trade selected policy and the separately specified fixed-threshold losses.
- The HTML notebook export was generated successfully with embedded figures. A full browser layout inspection was unavailable because the in-app browser rejected the local-file URL protocol. Figure inspection and notebook execution do not constitute a full-page HTML layout check.

## Independent audit (2026-10-22)

See `docs/AUDIT.md`. These checks were re-run against the raw CSVs and the saved outputs:
- Message/book alignment: each type-4 price equals the previous row's best quote on the resting side (100%), and the 1/10,000 price scale was confirmed. The sample has no type 6 or type 7 rows, no dummy levels and no crossed or locked books.
- All 85,408 targets recomputed from raw source and target rows. `classification.csv` (all rows) and `classification-by-fold.csv` (200 rows) reproduced from `predictions.parquet`. Trades reconcile with `execution-sensitivity.csv`, and fee, spread and non-overlap arithmetic were checked per trade.
- Validation-only model and threshold selection was re-derived for all 25 folds. There were no exact macro-F1 ties.
- IC decomposition: `./scripts/python scripts/ic_diagnostics.py` writes `results/diagnostics/`.

## Interpretation

Validation establishes that the implemented sample study runs and its reported results reconcile. It does not establish current-market performance, executable fills or persistent trading alpha. The data contain a single historical trading date. No model hyperparameters or thresholds were retuned on test outcomes after the first run.

GitHub Actions is provided as a workflow file. It has not run remotely because the repository has not been published.

## Continuation checks (2026-10-22, branch `claude/lob-continuation`)

- The audit, economics and crypto-replication branches were merged. All 44 tests pass: the 23 original tests, 8 economics cases and 13 crypto adapter tests.
- The equity pipeline was rerun on the merged code. Every committed result table and figure reproduced byte for byte; only the timestamps and source hashes in `run.json` changed. `lob_research.economics` and `scripts/ic_diagnostics.py` also reproduced their committed outputs exactly.
- The notebook was rebuilt and executed top to bottom with no cell errors. The HTML export was regenerated, but its full-page layout was not inspected in a browser.
- Headline README numbers were recomputed from `results/classification.csv`, `results/validation-decisions.csv` and `results/crypto-binance/heldout/classification.csv`.
- The crypto held-out phase was run once by the replication work, from commit `39c1fde`. It was not rerun during integration, because the merge changed no crypto code. The raw files (about 12 GB with caches) are in the `wt-multisession` worktree under `data/tardis/`.
- GitHub Actions has still not run remotely.
