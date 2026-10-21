"""Create the portable research companion with nbformat, then execute separately."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
md("""# Limit order book signals

**Research objective:** test whether order-flow and depth features predict short-horizon mid-price moves, then distinguish prediction from net trading value. This notebook is the executed companion to a packaged Python pipeline. The source is the official LOBSTER sample for five NASDAQ stocks on 21 June 2012.

## tl;dr

The cells below compute the summary from the saved run. One historical session provides evidence of implementation and intraday evaluation, rather than persistent alpha."""),
code("""from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from IPython.display import display, Markdown, Image

ROOT = Path.cwd()
if not (ROOT / 'configs/study.json').exists():
    ROOT = ROOT.parent
assert (ROOT / 'configs/study.json').exists(), 'Open the notebook from the repository or notebooks folder'
R = ROOT / 'results'
run = json.loads((R / 'run.json').read_text())
metrics = pd.read_csv(R / 'classification.csv')
decisions = pd.read_csv(R / 'validation-decisions.csv')
chosen = metrics.loc[metrics.model == 'validation_selected_model']
display(Markdown(f"**{run['total_source_rows']:,} source events; {run['total_observations']:,} usable decision observations; {run['folds_per_symbol']} walk-forward windows per stock.**"))
chosen[['symbol', 'macro_f1', 'directional_hit_rate', 'directional_coverage', 'information_coefficient']].round(4)
"""),
md("""## Context & Methods

Decisions follow every twentieth supplied book event. The target is the mid-price direction 100 events ahead. Features use the exact source row, including file ordering for equal timestamps. The first fold trains on 40% of the session, validates on the next 10% and tests on the next 10%; training expands by 10% for each subsequent fold. Target and execution endpoints are purged at boundaries.

**Key assumptions:** LOBSTER has already reconstructed the venue book. The price scale is 1/10,000 USD. There is no consolidated NBBO or queue model. Predictions refer to event time; execution scenarios add wall-clock latency. A displayed quote is not a guaranteed fill. Fees of 0/1/5 bp per side are sensitivity assumptions.

### Reproduce the full computation

From the repository root, create a Python 3.12+ environment, install the project, then run `./scripts/python -m lob_research.cli download` and `./scripts/python -m lob_research.cli run`. The wrapper resolves the OpenMP library on macOS using the official scikit-learn wheel where available. This notebook reads the generated outputs, independently reconciles key measures and displays the saved figures. It does not silently substitute synthetic or missing data."""),
code("""config = json.loads((ROOT / 'configs/study.json').read_text())
assert hashlib.sha256((ROOT / 'configs/study.json').read_bytes()).hexdigest() == run['config_sha256']
assert hashlib.sha256((ROOT / 'docs/RESEARCH_PLAN.md').read_bytes()).hexdigest() == run['plan_sha256']
for filename, digest in run['outputs_sha256'].items():
    assert hashlib.sha256((R / filename).read_bytes()).hexdigest() == digest, filename
display(Markdown('Configuration, research-plan hash and saved CSV hashes match the executed run.'))
"""),
md("""## Data

The provider's [sample page](https://php.lobsterdata.com/info/DataSamples.php) supplies paired message and ten-level book files. The [data specification](https://php.lobsterdata.com/info/DataStructure.php) defines event types, occupied levels, price scaling and execution direction. Raw files are downloaded separately and excluded from Git.

### Source quality and coverage"""),
code("""quality = pd.read_csv(R / 'data-quality.csv')
display(quality[['symbol', 'date', 'source_rows', 'invalid_book_rows', 'equal_timestamp_pairs', 'usable_observations', 'excluded_grid_rows']])
assert quality.source_rows.sum() == run['total_source_rows']
manifest = json.loads((ROOT / 'data/sample-manifest.json').read_text())
for item in manifest:
    archive = ROOT / 'data/raw' / (item['symbol'] + '.zip')
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == item['sha256']
observations = pd.read_parquet(R / 'observations.parquet')
assert len(observations) == run['total_observations']
assert ((observations.target_row - observations.source_row) == config['horizon_events']).all()
assert (observations.target_time > observations.time).all()
duration = observations.target_time - observations.time
display(duration.quantile([.1, .5, .9]).rename('100-event horizon in seconds').to_frame().round(3))
"""),
md("""### Walk-forward partitions

The test windows are sequential and disjoint. The table shows actual retained endpoints after purging; a validation or training target cannot reach the next partition."""),
code("""splits = pd.read_csv(R / 'split-counts.csv')
display(splits.loc[splits.symbol == 'AAPL', ['fold', 'split', 'n', 'first_seconds', 'last_seconds', 'max_target_seconds']].round(3))
for (symbol, fold), group in splits.groupby(['symbol', 'fold']):
    a = group.set_index('split')
    if len(run['session_dates']) == 1:
        assert a.loc['train', 'max_target_seconds'] + .2 < a.loc['validation', 'first_seconds']
        assert a.loc['validation', 'max_target_seconds'] + .2 < a.loc['test', 'first_seconds']
    else:
        assert a.loc['train', 'last_date'] < a.loc['validation', 'first_date']
        assert a.loc['validation', 'last_date'] < a.loc['test', 'first_date']
display(Markdown('All retained training and validation endpoints precede the next partition, including the 200 ms sensitivity latency.'))
"""),
md("""## Results

### Predictive performance

Macro-F1 weights down, unchanged and up equally. Directional hit rate counts flat outcomes as misses and is shown alongside coverage. Rank IC compares the continuous directional score with future mid-price returns. The model-selection series changes only according to earlier validation performance."""),
code("""display(metrics.pivot(index='symbol', columns='model', values='macro_f1').round(4))
display(Image(filename=str(R / 'figures/prediction-comparison.png'), alt='Macro-F1 comparison across five stocks and four models'))
"""),
md("""The comparison includes the simple imbalance rule and majority baseline. Improvements are scoped to these five intraday test intervals; the number of events does not represent independent days."""),
code("""display(Image(filename=str(R / 'figures/information-coefficient.png'), alt='Information coefficients with block-bootstrap intervals'))
display(chosen[['symbol', 'information_coefficient', 'ic_low', 'ic_high', 'directional_hit_rate', 'directional_coverage']].round(4))
"""),
md("""The 95% bounds resample 60-second clock-time blocks within the observed test series. They do not quantify uncertainty over future trading days or correct for choosing among five stocks."""),
code("""display(Image(filename=str(R / 'figures/walk-forward-stability.png'), alt='Model performance across five walk-forward windows'))
"""),
md("""### Spread, fees and net P&L

Each trade is one share, enters and exits across the spread and waits until the previous position is closed. The primary scenario adds 50 ms latency and 1 bp fees per side. A no-trade threshold can be selected on validation. We retain a pre-specified 0.55-confidence diagnostic to show execution economics even when the primary policy abstains."""),
code("""economic = pd.read_csv(R / 'execution-sensitivity.csv')
selected_keys = decisions.loc[decisions.selected_model, ['symbol', 'fold', 'model']]
selected_economic = economic.merge(selected_keys, on=['symbol', 'fold', 'model'])
primary = selected_economic.loc[(selected_economic.policy == 'validation_selected') & (selected_economic.latency_ms == 50) & (selected_economic.fee_bps_per_side == 1)]
totals = primary.groupby('symbol')[['trades', 'total_net_dollars']].sum()
display(totals.round(3))
display(Markdown(f"**Primary scenario: {int(totals.trades.sum()):,} one-share trades; combined net P&L ${totals.total_net_dollars.sum():,.2f}.**"))
display(Image(filename=str(R / 'figures/walk-forward-pnl.png'), alt='Stitched net P and L for validation-selected and fixed-threshold policies'))
"""),
code("""display(Image(filename=str(R / 'figures/cost-sensitivity.png'), alt='Net basis points per trade under three fee scenarios'))
diagnostic = economic.loc[(economic.policy == 'fixed_0.55_diagnostic') & (economic.latency_ms == 50) & (economic.fee_bps_per_side == 1)]
display(diagnostic.groupby(['symbol', 'model'])[['trades', 'total_net_dollars']].sum().round(3))
"""),
md("""### Independent reconciliations

Recompute macro-F1 and rank IC directly from the row-level predictions, rather than accepting the summary CSV. Check probability sums, disjoint test rows, non-overlapping executions and trade cash accounting."""),
code("""predictions = pd.read_parquet(R / 'predictions.parquet')
test = predictions.loc[predictions.split == 'test']
for (symbol, model), group in test.groupby(['symbol', 'model']):
    assert not group.duplicated(['date', 'source_row']).any()
    np.testing.assert_allclose(group[['p_down', 'p_flat', 'p_up']].sum(axis=1), 1, atol=1e-6)
    counts = np.zeros((3, 3), dtype=int)
    np.add.at(counts, (group.target.to_numpy(), group.prediction.to_numpy()), 1)
    denom = counts.sum(axis=0) + counts.sum(axis=1)
    f1 = np.divide(2*np.diag(counts), denom, out=np.zeros(3), where=denom > 0).mean()
    reported = metrics.loc[(metrics.symbol == symbol) & (metrics.model == model)].iloc[0]
    np.testing.assert_allclose(f1, reported.macro_f1)
    if group.score.nunique() > 1 and group.future_mid_return_bps.nunique() > 1:
        np.testing.assert_allclose(spearmanr(group.score, group.future_mid_return_bps).statistic, reported.information_coefficient)
trades = pd.read_csv(R / 'primary-scenario-trades.csv')
np.testing.assert_allclose(trades.net_dollars, trades.gross_dollars - trades.fees_dollars, atol=1e-10)
selected_trades = trades.loc[trades.policy == 'validation_selected'].merge(selected_keys, on=['symbol', 'fold', 'model'])
for (symbol, date), group in selected_trades.groupby(['symbol', 'date']):
    group = group.sort_values('time')
    assert (group.time.to_numpy()[1:] > group.exit_time.to_numpy()[:-1]).all()
np.testing.assert_allclose(selected_trades.net_dollars.sum(), totals.total_net_dollars.sum(), atol=1e-8)
display(Markdown('Prediction metrics, disjoint test rows, trade accounting and non-overlap checks reconciled.'))
"""),
md("""## Takeaways

This project demonstrates event-level LOB processing, causal feature construction, fixed-model walk-forward testing and the distinction between signal prediction and net trading value. It does not establish professional trading experience or a persistent executable edge.

The main limitations are a single historical session, one venue book, illustrative fees and displayed-quote execution without routing, impact or borrow constraints. The next useful extension is multiple untouched trading dates with the same frozen pipeline. See `docs/FINDINGS.md` for the result-specific narrative, and `docs/RESEARCH_PLAN.md` for exact definitions.

The implementation was developed with Codex assistance. Anyone presenting it should be able to explain the feature formulas, event ordering, fold boundaries and execution assumptions."""),
]
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.12"}}
path = root/'notebooks/orderbook-study.ipynb'
path.parent.mkdir(exist_ok=True)
nbf.validate(nb)
nbf.write(nb, path)
print(path)
