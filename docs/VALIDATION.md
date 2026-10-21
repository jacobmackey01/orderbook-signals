# Validation record

The final real-data run completed for five stocks and five expanding walk-forward windows per stock. The configuration, research plan, Python versions, source-code hashes and CSV hashes match `results/run.json`.

## Executed checks

- 23 unit tests passed. They cover paired-file failures, out-of-order data, equal-timestamp events, dummy/crossed depth, halts, OFI price resets, causal feature windows, resting-order execution signs, split purging, disjoint expanding test windows, train-only scaling, exact event horizons, entry/exit prices, spread and fee arithmetic, no overlapping positions, abstention, missing exits and undefined IC.
- The full pipeline processed 2,110,860 real source rows and retained 85,408 decision observations.
- An independent raw-CSV calculation reconciled all 85,408 current mids, future returns and target classes to the exact source/target book rows.
- The final notebook executed top to bottom without cell errors. It independently recomputes macro-F1 and IC from predictions, checks probability sums and disjoint test rows, reconciles selected-policy P&L to individual trades and checks non-overlap and fee accounting.
- All five exported figures were inspected for visible values, labels, units and layout. The P&L figure includes the no-trade selected policy and the separately specified fixed-threshold losses.
- The HTML notebook export was generated successfully with embedded figures. A full browser layout inspection was unavailable because the in-app browser rejected the local-file URL protocol. Figure inspection and notebook execution do not constitute a full-page HTML layout check.

## Interpretation

Validation establishes that the implemented sample study runs and its reported results reconcile. It does not establish current-market performance, executable fills or persistent trading alpha. The data contain a single historical trading date. No model hyperparameters or thresholds were retuned on test outcomes after the first run.

GitHub Actions is provided as a workflow file. It has not run remotely because the repository has not been published.
