# Economics diagnostic plan

Written 2026-10-22, before any of the analyses below were computed. The predictive and execution results in `docs/FINDINGS.md` had already been seen: models rank future mid moves (rank IC 0.26–0.56), every validation-selected threshold abstains, and the fixed 0.55 diagnostic loses money. This plan asks **why**, using controlled comparisons on the same trades. It is not a search for a profitable rule.

## Status of the evidence

Every number is computed on the **existing walk-forward test windows of one session (21 June 2012)**, which have already been inspected. The work is therefore **diagnostic**, not out-of-sample confirmation. Execution uses historical displayed quotes only: one share, taker entry and exit across the spread, no passive fills, no queue positions, no impact, no annualisation. Displayed-quote outcomes are simulations, not realised profits.

## Questions and fixed analyses

1. **Which cost dominates?** For the same non-overlapping trades as the existing diagnostic (fixed 0.55 confidence for logistic and XGBoost, plus every directional argmax prediction), split each trade's net P&L exactly into
   `net = signal (decision mid → target mid) + latency (mid drift caused by trading at t+L instead of t) − half-spread at entry − half-spread at exit − fees`.
   Report per stock in bp per trade and ticks per trade (1 tick = $0.01) at 0/50/200 ms and 0/1/5 bp per side. The components must sum exactly to the existing simulator's net P&L, trade by trade.
2. **Is the expected move ever larger than the spread?** Group every test decision of the validation-selected model by score decile (score = P(up) − P(down)). Compare the conditional mean future mid return with the decision spread and with spread + 2 × fee, in bp and in ticks.
3. **Is confidence the same as expected return?** Group decisions by directional confidence max(P(up), P(down)) and report hit rate, mean favourable mid move and mean net bp per decision for a taker trade in the predicted direction (overlapping decisions; a description, not a strategy).
4. **What is the ceiling?** With perfect foresight of the executable quotes, what fraction of decisions could have been profitable after crossing the spread and paying fees? Also report the non-overlapping perfect-foresight total as an explicit look-ahead upper bound.

## Pre-specified cost-aware rule (question 5)

Written question: *if the model's probabilities are converted into an expected mid move and compared with the observed round-trip cost, does the model ever expect to clear that cost?*

Rule, fixed here and not revised after seeing test P&L:

- For each stock, fold and model (logistic, XGBoost), estimate on that fold's **training rows only** `m_up = mean(future mid return bp | realised up)` and `m_down = mean(|future mid return bp| | realised down)`.
- At each decision, `E[r] = P(up)·m_up − P(down)·m_down` in bp, using the fold model's probabilities.
- Round-trip cost at the decision = observed spread in bp (two half-spreads, assuming the spread at exit equals the spread at entry) + 2 × fee per side (1 bp in the primary scenario).
- Go long if `E[r] > cost`, short if `−E[r] > cost`, otherwise abstain. The break-even margin is zero, so there is **no free threshold**; validation is used only to report the rule's validation result before the single test evaluation.
- Evaluate once with the existing simulator at 50 ms latency and 1 bp per side. Report validation and test trade counts and net P&L, labelled as assessed on already-seen windows.

If the rule trades almost never, that is the answer to the written question and it is reported as such.
