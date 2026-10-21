# Limit order book signals

**Models using order-book features improve short-horizon prediction in these equity samples, but the validation-selected trading policies abstain once spread and fee costs are included.** This project tests that distinction using real LOBSTER data, XGBoost and an expanding walk-forward backtest.

The sample contains **2,110,860 paired book updates** across AAPL, AMZN, GOOG, INTC and MSFT on **21 June 2012**. It is a study of one historical session, not evidence of persistent alpha.

![Walk-forward prediction comparison](results/figures/prediction-comparison.png)

## Research question

Can order-flow and depth imbalance predict short-horizon mid-price moves out of sample? Does a tree-based model improve on simple baselines, and does that predictive information survive trading costs?

Every source event contributes to the features. A decision is made after every 20 supplied events, with a target 100 events ahead. The realised median prediction horizon is about 3.18 seconds in the bundled run; event time varies with activity. The pipeline retains 85,408 decision observations after filtering.

## What the project does

- Validates paired message/book files, price scaling, occupied levels, chronological ordering and interrupted sessions. Equal-timestamp events keep their source-file order.
- Builds order-flow imbalance, multi-level depth imbalance, microprice, spread and trailing execution-flow features without reading past the decision's exact event row.
- Compares majority, an imbalance rule, logistic regression and XGBoost over **five expanding intraday walk-forward folds per stock**. Model and trading thresholds use only earlier training/validation data.
- Reports macro-F1, directional hit rate with coverage, Spearman information coefficient, time-block uncertainty and P&L after spread and illustrative fees.
- Tests latency and fee assumptions, using a one-share, non-overlapping displayed-quote execution scenario. It does not assume passive fills.

LOBSTER supplies already reconstructed books. This repository validates and replays those snapshots; it does not claim to reconstruct a full ITCH feed or model queue priority.

## Findings

The validation-selected series improves test macro-F1 over the majority baseline for all five stocks. Directional hit rates are approximately 57–60%, with materially different directional coverage across stocks. Read the coverage alongside the hit rate: a model that mostly predicts no move is a different strategy from one that always takes a direction.

At 50 ms latency and an illustrative fee of 1 bp per side, every fold's validation-selected logistic/XGBoost threshold chooses **no trading**. A separately fixed 0.55-confidence diagnostic does trade and loses money after costs. The useful finding is the gap between predictive association and an economically viable execution rule.

See the [findings note](docs/FINDINGS.md) for exact metrics, uncertainty and limitations. The [executed notebook](notebooks/orderbook-study.ipynb) includes independent reconciliations, and the [scope](docs/SCOPE.md) maps the work to the intended research evidence.

## Reproduce

Use Python 3.12 or later. The tested environment uses Python 3.12. Exact package versions are retained in `requirements-lock.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
./scripts/python -m lob_research.cli download
./scripts/python -m pytest -q
./scripts/python -m lob_research.cli run
```

`scripts/python` behaves like `python3`. On macOS it exposes the OpenMP runtime bundled in the official scikit-learn wheel to XGBoost; if your wheel does not include that runtime, install `libomp` through your normal package manager. On Linux it simply invokes Python. Set `PYTHON_BIN` to select a particular interpreter.

The download step retrieves five public ZIP files (about 32 MB compressed), verifies the committed SHA-256 hashes and extracts only the expected file types. Raw data stay local. The research run regenerates the metrics, figures and findings note. It makes no data purchases, submits no orders and requires no API key.

To rebuild and execute the companion notebook after a run:

```bash
./scripts/python scripts/build_notebook.py
./scripts/python -m jupyter nbconvert --execute --to notebook --inplace notebooks/orderbook-study.ipynb
```

## Evaluation design

The first fold uses 40% of the fixed intraday interval for training, 10% for validation and the next 10% for test. Training expands by 10% at each step. Test windows never overlap. Labels and execution endpoints are purged at partition boundaries, including the longest latency scenario. Later folds may learn from earlier completed test intervals.

The target has three classes: down, unchanged and up. Hit rate is directional correctness among predicted up/down observations, counting unchanged realised outcomes as misses. IC is the Spearman correlation of `P(up) - P(down)` with the future mid-price return. Bootstrap intervals resample 60-second clock-time blocks; they do not establish robustness across days.

The execution diagnostic enters and exits across the best displayed quotes, adds fees on both sides and skips new decisions until the previous position exits. At zero latency it uses the exact event row rather than a later event with the same timestamp. If an entered position has an unpriced exit, the run fails rather than deleting that trade with hindsight. The full design and fixed parameters are in the [research plan](docs/RESEARCH_PLAN.md).

## Repository

```text
src/lob_research/     acquisition, event features, models, evaluation and CLI
configs/study.json   fixed universe, horizons, folds and cost assumptions
tests/               causality, chronology, trade accounting and boundary checks
notebooks/           executed research companion
results/             aggregate metrics, figures and run provenance
docs/                scope, research plan, findings and experiment log
data/                archive manifest; ignored raw inputs downloaded separately
```

`results/predictions.parquet`, `results/observations.parquet` and individual trade records are local audit inputs, excluded from Git. The notebook's displayed outputs remain readable without them; rerunning its checks requires downloading the samples and running the pipeline first.

## Extend to additional sessions

Place matching ten-level LOBSTER `*_message_10.csv` and `*_orderbook_10.csv` files in `data/raw`, using the provider's original naming convention. Supply the same dates for every configured symbol. With five or more dates, the pipeline switches explicitly to expanding complete-date folds; two to four dates are rejected. Use a separate result location or copy of the repository to retain the bundled baseline. Do not repeatedly tune on this sample's test windows.

The next substantive test is performance across multiple untouched sessions with a better execution model. Displayed quotes do not guarantee fills; routing, borrow availability and market impact are absent. The illustrative fees are not an account-specific NASDAQ schedule.

## Sources and attribution

- [Official LOBSTER sample downloads](https://php.lobsterdata.com/info/DataSamples.php).
- [LOBSTER output specification](https://php.lobsterdata.com/info/DataStructure.php), including the resting-order direction convention and 1/10,000 USD price scale.
- [Data manifest](data/sample-manifest.json), with exact URLs and archive hashes. Raw files and provider documentation remain subject to the provider's rights and are not relicensed with this code.

Developed with Codex assistance. The research conclusions follow from the saved run; anyone presenting the project should be able to explain and reproduce the methodology.
