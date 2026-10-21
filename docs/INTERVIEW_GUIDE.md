# Interview guide: limit order book signals

This guide is for understanding the project well enough to explain it, defend it and extend it. Every number comes from the saved run or the audit and economics diagnostics in this repository. Work through the exercises yourself rather than memorising the answers.

Run the exercises from the repository root after `download` and `run`, using `./scripts/python`. Each snippet reads only files the pipeline already writes.

---

## 1. What a limit order book represents

A limit order book is the set of resting buy (bid) and sell (ask) limit orders at one venue, grouped by price level. Each level has a price and a total displayed size. The best bid and best ask make up the top of book. Their difference is the spread, and their average is the mid.

LOBSTER pairs every message (a new order, cancellation, execution or hidden execution) with the 10-level book state *after* that message. This project replays those already-reconstructed NASDAQ states. It does not reconstruct the ITCH feed, track queue positions or see other venues. NASDAQ was one of several US venues in 2012, so this is a venue book, not the consolidated NBBO.

**Exercise 1: read one row yourself.** Pick an MSFT decision, read its raw book row and recompute four features.

```python
import pandas as pd
obs = pd.read_parquet("results/observations.parquet")
r = obs[obs.symbol == "MSFT"].iloc[1000]
book = pd.read_csv("data/raw/MSFT_2012-06-21_34200000_57600000_orderbook_10.csv", header=None)
row = book.iloc[int(r.source_row)]
ask, aq, bid, bq = row[0] / 1e4, row[1], row[2] / 1e4, row[3]   # prices are USD x 10,000
mid = (ask + bid) / 2
print(ask, aq, bid, bq)
print("spread bp", (ask - bid) / mid * 1e4, "vs", r.spread_bps)
print("L1 imbalance", (bq - aq) / (bq + aq), "vs", r.imbalance_1)
print("microprice bp", ((ask * bq + bid * aq) / (bq + aq) - mid) / mid * 1e4, "vs", r.microprice_bps)
t = book.iloc[int(r.target_row)]
print("future mid bp", ((t[0] + t[2]) / 2e4 - mid) / mid * 1e4, "vs", r.future_mid_return_bps)
```

This decision has bid 30.96 × 2,836 and ask 30.98 × 6,079: a two-tick spread (6.5 bp), more size on the ask, imbalance −0.36, microprice −1.17 bp below the mid. The mid 100 events later is 0.5 cent lower (−1.6 bp). That is a half-tick move: one side of the quote moved and the other did not.

## 2. Depth imbalance versus order-flow imbalance

**Depth imbalance** is a snapshot: `(bid size − ask size) / (bid size + ask size)`, computed here over 1, 5 and 10 levels. It describes the queues as they stand.

**Order-flow imbalance (OFI)** is a flow. It measures how the best queues *changed* between consecutive events (Cont, Kukanov and Stoikov). With best bid `b`, bid size `qb`, best ask `a` and ask size `qa`, each event contributes

```
e_n = 1[b_n ≥ b_{n-1}]·qb_n − 1[b_n ≤ b_{n-1}]·qb_{n-1} − 1[a_n ≤ a_{n-1}]·qa_n + 1[a_n ≥ a_{n-1}]·qa_{n-1}
```

When a best price changes, the whole old queue leaves the calculation and the whole new one enters it. Price changes are why OFI is not just a difference in displayed sizes. The project sums `e_n` over the trailing 500 ms and 5 s and divides by the current top-of-book size (`src/lob_research/data.py`, `ofi_events`).

**Worked example. The two can disagree.**

| event | bid | bid size | ask | ask size | e_n | why |
|---|---|---|---|---|---|---|
| t0 | 30.00 | 500 | 30.01 | 300 | | |
| t1 | 30.00 | 700 | 30.01 | 300 | +200 | 200 shares added at the bid |
| t2 | 30.00 | 700 | 30.02 | 400 | +300 | ask queue at 30.01 cleared; the 300 departed shares count as buying pressure |
| t3 | 30.01 | 100 | 30.02 | 400 | +100 | new, higher bid of 100 |

OFI over the window is +600, strong buying pressure. Yet at t3 the L1 depth imbalance is (100 − 400)/500 = **−0.6**, which points the other way. The flow says buyers have been pushing. The snapshot says the new bid queue is thin. Models get both features because they answer different questions.

**Exercise 2.** Add the row `t4: bid 30.01 × 100, ask 30.01 × 50` (a locked book). Why does the pipeline mark such states invalid rather than computing `e_n`? (Hint: crossed and locked books are excluded in `validate_arrays`, and windows never cross an invalid segment.)

## 3. Microprice and its assumptions

The weighted mid used here is

```
microprice = (ask · bid size + bid · ask size) / (bid size + ask size)
```

Example: bid 30.00 × 900, ask 30.01 × 100. Mid = 30.005. Microprice = (30.01·900 + 30.00·100)/1000 = **30.009**, displaced +1.33 bp. The thin ask queue is more likely to empty first, which would move the mid up.

Assumptions to state if asked:
- Only level 1 matters, and queue size is a proxy for the probability that each side depletes first. Hidden liquidity, cancellations and arriving orders are ignored.
- It is a linear interpolation. Stoikov's (2018) "micro-price" is a different, model-based estimate of the expected future mid. Do not call this feature that.
- It is a forecast of where the mid goes, not a price anyone can trade at.

In this data the microprice alone has rank IC 0.47 (INTC) and 0.51 (MSFT), about 90% of the full model's IC. It has only 0.07–0.17 for the high-priced stocks (`results/diagnostics/ic-decomposition.csv`).

## 4. Event time versus clock time

A decision is made every 20 book events, and the target is the mid 100 events later. Event time adapts to activity: busy markets reach 100 events in milliseconds, quiet ones take seconds. In this run the median realised horizon is 3.18 s, and the 10th–90th percentile range is 0.003–12.97 s.

Why it matters:
- The "same" horizon covers very different calendar durations, so latency of 50 ms is a big share of a 3 ms horizon and irrelevant for a 13 s one.
- Bursts with equal timestamps exist. The project uses the exact source row as the information boundary, so later rows with the same timestamp are not used. The audit found 3–11% of decisions fall *inside* such a burst, often partway through a sweep. That follows the rule, but nobody could act at that point. Dropping them lowers IC only slightly (INTC 0.52 → 0.49, MSFT 0.56 → 0.53).
- A clock-time mode (500 ms grid, 1 s target) is still in the code as an option for comparison.

**Exercise 3.** Compute the horizon distribution yourself:

```python
import pandas as pd
o = pd.read_parquet("results/observations.parquet")
h = o.target_quote_time - o.source_time
print(h.describe(percentiles=[.1, .5, .9]))
print(h.groupby(o.symbol).median())
```

Which stock has the shortest median horizon, and why might that be?

## 5. Why overlapping labels matter

With a decision every 20 events and a 100-event target, consecutive labels share about 80% of their path. Each label overlaps four or five neighbours. That has three consequences:

1. **Leakage across splits.** A training label can reach into the validation or test window. The pipeline purges any observation whose target time plus 200 ms (the largest latency) reaches the next partition boundary.
2. **Overstated precision.** 85,408 observations are not 85,408 independent draws. An i.i.d. bootstrap would make the IC interval for MSFT about half as wide (0.025 vs 0.061) as the 60-second block bootstrap the project uses.
3. **Check, don't assume.** The audit recomputed IC on non-overlapping subsets (every 5th decision, and a greedy non-overlapping set). The point estimates barely moved (MSFT 0.555–0.569 vs 0.561), and the block interval on the full sample is about as wide as on the 1-in-5 subset. So the block bootstrap handles the overlap.

The bigger dependence problem is not visible inside one session: all observations share one day's regime. The block bootstrap says nothing about other days.

## 6. How the chronological evaluation prevents leakage

The session interval is 09:35–15:55 (22,800 s). Each 10% slice is 38 minutes.

| fold | train | validation | test |
|---|---|---|---|
| 0 | 0–40% | 40–50% | 50–60% (from 12:45) |
| 1 | 0–50% | 50–60% | 60–70% |
| … | expands by 10% | | |
| 4 | 0–80% | 80–90% | 90–100% |

- The feature scaler sits inside the sklearn pipeline, so it is fitted on training rows only.
- Model settings were fixed before any real-data result. Each fold picks its model by validation macro-F1 and its trading threshold by validation net P&L.
- Test windows are disjoint. Later folds train on earlier test windows. That copies a running research process, but it means those windows are not untouched for later folds.
- Feature lookbacks (up to 5 s) may cross a boundary. That is fine because they use only past data, as a live system would.

**Exercise 4.** Open `results/split-counts.csv` and confirm that the test windows do not overlap and that each fold's validation precedes its test. Then explain why choosing the threshold on *test* P&L would be leakage even though the model was fitted on training data.

## 7. Why directional hit rate needs coverage

Hit rate = correct up/down calls ÷ all up/down calls, counting realised "unchanged" outcomes as misses. Coverage = the share of decisions where the model calls a direction at all.

| | hit rate | coverage |
|---|---|---|
| AAPL | 57.3% | 100% |
| INTC | 58.5% | 12.8% |
| MSFT | 59.3% | 26.6% |

For INTC, 75.6% of future mid changes are exactly zero, so the model mostly predicts "unchanged" and calls a direction only when it is fairly sure. A 58% hit rate on 13% of decisions is a different object from 57% on all of them. Without coverage, a model could look accurate by abstaining.

**Exercise 5.** Reproduce the table and the zero-return share:

```python
import pandas as pd
p = pd.read_parquet("results/predictions.parquet")
t = p[(p.split == "test") & (p.model == "validation_selected_model")]
for s, g in t.groupby("symbol"):
    d = g[g.prediction != 1]                      # classes: 0 down, 1 unchanged, 2 up
    print(s, "hit", round((d.prediction == d.target).mean(), 4),
          "coverage", round(len(d) / len(g), 4),
          "zero-return share", round((g.future_mid_return_bps == 0).mean(), 3))
```

Then compute the hit rate of a rule that always predicts "up" for INTC. Why is it so low?

## 8. Why positive IC can coexist with negative net P&L

IC (Spearman correlation between `P(up) − P(down)` and the future mid return) measures **ranking**. P&L measures whether the **size** of a move beats the **cost** of trading it. These are different tests.

What the diagnostics show (all on already-seen test windows, so explanatory only):
- **Where the IC comes from.** For INTC/MSFT, 80–82% of non-zero moves are half-tick moves: a one-tick spread opens because one best queue empties. Queue imbalance predicts which side empties, so the model gets the sign of a non-zero move right about 89% of the time. For AAPL/AMZN/GOOG (spreads of 12–26 one-cent ticks) the IC comes mostly from short-term mid continuation and imbalance inside a wide spread.
- **Where the money goes.** For XGBoost at the fixed 0.55 confidence diagnostic (50 ms, 1 bp per side), the mid-to-mid signal earns +0.4 to +2.0 bp per trade, while crossing the spread costs 2.3–5.5 bp. That makes the spread 2–8 times the signal. With zero fees every stock still loses 2–5 bp per trade.
- **The ceiling.** Even with perfect foresight, taker trades clear the spread on 0% (INTC, MSFT) to 9.7% (AAPL) of decisions at zero fees, and on at most 0.5% at 1 bp per side. Over the 100-event horizon the INTC/MSFT mid never moved more than one tick.

**Worked example ($30 stock, 1-cent tick).** Bid 30.00, ask 30.01, mid 30.005. The model correctly predicts the ask lifting to 30.02, so the mid rises half a tick (+1.7 bp). A taker buys at 30.01 and later sells at the bid, now 30.01 at best: gross $0.00, fees −$0.006. If only the ask moved, the bid is still 30.00: gross −$0.01. **A correct prediction still loses.** The mid has to move by more than the whole spread before a taker breaks even.

The classifier is trained on class probabilities, not on whether a move exceeds cost. Higher confidence raises the hit rate (AMZN 0.47 → 0.81 across confidence deciles), but net return per decision stays flat and negative in every decile. A pre-specified cost-aware rule (trade only when the model's expected return exceeds spread plus fees) never fired: the largest expected return on any decision was 1.9 bp, against costs of 4.3–7.5 bp.

**Exercise 6.** Using the worked example, find the smallest favourable mid move (in ticks) at which a one-share taker round trip breaks even at 0 bp and at 1 bp fees per side. Then check `results/economics/mid-move-distribution.csv` for how often INTC moved that far.

## 9. What the execution diagnostic assumes

- One share, a market (taker) order in and out: buy at the ask, sell at the bid, or the reverse for shorts.
- Entry at decision time + latency; exit at target-event time + latency, using the quote displayed at that moment. The primary case is 50 ms and an illustrative 1 bp per side; the sensitivities are 0/50/200 ms and 0/1/5 bp.
- Positions never overlap within a stock. A missing exit quote fails the run rather than being dropped.
- Not modelled: queue position, passive fills, other venues, hidden liquidity, market impact, short borrow, actual NASDAQ fee and rebate tiers. A displayed quote does not guarantee a fill, so the results are simulations, not realised profits.

Be ready to say why passive execution is the natural next idea and why it is hard. Queue-depletion signals are exactly what market makers use to decide when to cancel. A passive model needs queue position, fill probability and adverse selection, and the samples can't support claims about those.

## 10. Why the single-session evidence is limited

- One date (21 June 2012), five stocks, chosen before results. 85,408 decision observations, but only **one** independent day for regime purposes.
- The block bootstrap measures within-day sampling noise, not persistence. A news day, a different volatility regime or a tick-size change could move everything.
- Market structure has changed since 2012 (more fragmentation, different fee tiers, odd-lot quoting), so absolute numbers do not transfer.
- Several real-data paths (halts, auction crosses, crossed books, empty levels) never occur in this sample. They are covered only by unit tests.

The independent-session extension (see the README and `docs/REPLICATION_FINDINGS.md` if present) is a separate replication in a different market. It shows whether the *pattern* recurs across dates, not that the equity numbers persist.

## 11. What to test next with better data

In order of value:
1. **Multiple untouched equity sessions**, splitting on complete dates (the pipeline already supports five or more dates), with results reported per date.
2. **Consolidated quotes (NBBO) and trades**, so execution uses the best price across venues.
3. **A target aligned with the economics**: predict whether the move exceeds the spread plus fees, or the signed move in ticks, rather than a three-class direction.
4. **Passive execution with an explicit fill model**: LOBSTER order IDs allow queue tracking at displayed levels. This needs a fill and adverse-selection model, validated before any P&L claim.
5. Longer horizons, where the move distribution is wider relative to the spread, with the same walk-forward discipline.

Not worth doing: more trees, deep models or a hyperparameter search on this session. The constraint is economics and data, not model capacity.

---

## Quick self-test

1. Why does a 1-cent tick on a $27 stock make a profitable taker strategy over a 3-second horizon nearly impossible?
2. A colleague reports a 0.56 IC. What two numbers do you ask for next? (Suggested: coverage and the share of zero-return outcomes; then net bp per trade after crossing.)
3. Why do later folds training on earlier test windows not invalidate those later folds' test results?
4. What would convince you the INTC/MSFT signal is *not* just spread-state mechanics?
5. The validation-selected policy never trades. Is that a failure of the project?
