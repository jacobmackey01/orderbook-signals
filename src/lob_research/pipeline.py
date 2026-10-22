"""Fixed walk-forward study with inspectable prediction and execution evidence."""
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .data import FEATURES, load_session, make_observations, walk_forward_folds
from .evaluation import (classification_metrics, paired_block_interval, choose_threshold,
                         direction_from_probabilities, simulate_dates, trade_metrics,
                         information_coefficient, block_ic_interval)
from .models import fit_models


def validate_config(c):
    if c['levels'] != 10:
        raise ValueError('This study requires exactly ten book levels')
    for name in ['grid_seconds', 'horizon_seconds', 'max_quote_age_seconds']:
        if c[name] <= 0:
            raise ValueError('Time intervals must be positive')
    if c.get('sampling', 'clock') not in ['event', 'clock']:
        raise ValueError('Sampling must be event or clock')
    for name in ['event_stride', 'horizon_events']:
        if not isinstance(c[name], int) or c[name] < 1:
            raise ValueError('Event intervals must be positive integers')
    if c['start_seconds'] >= c['end_seconds'] or not c['symbols']:
        raise ValueError('Invalid study interval or empty universe')
    if c['latency_ms'] not in [0, 50, 200] or c['fee_bps_per_side'] < 0:
        raise ValueError('Primary latency must be covered by the sensitivity purge')
    if not any(t > 1 for t in c['confidence_thresholds']):
        raise ValueError('Threshold grid must include an abstention candidate')
    a, b, step = (c['walk_forward_initial_train_fraction'], c['walk_forward_validation_fraction'], c['walk_forward_test_fraction'])
    if min(a, b, step) <= 0 or a+b+step > 1 or not np.isclose((1-a-b)/step, round((1-a-b)/step)):
        raise ValueError('Walk-forward fractions must leave an integral number of test windows')


def evaluate_fold(symbol, sessions, train, validation, test, fold, config, records):
    if min(len(train), len(validation), len(test)) < 100:
        raise ValueError(f'Insufficient data for {symbol} fold {fold}')
    models = fit_models(train[FEATURES], train.target, config['seed'])
    validation_scores, fold_predictions = [], {}
    for model_name, model in models.items():
        val_p = model.predict_proba(validation[FEATURES])
        val_y = model.predict(validation[FEATURES]).astype(int)
        vm = classification_metrics(validation.target, val_y, val_p, model_name != 'imbalance_rule')
        validation_scores.append((model_name, vm['macro_f1']))
        if model_name in ['logistic', 'xgboost']:
            chosen, candidates = choose_threshold(sessions, validation, val_p, config)
            threshold = chosen['threshold']
            records['threshold_search'].extend({'symbol': symbol, 'fold': fold, 'model': model_name, **r} for r in candidates)
        else:
            threshold = 0.
        records['decisions'].append({'symbol': symbol, 'fold': fold, 'model': model_name,
                                    'validation_macro_f1': vm['macro_f1'], 'threshold': threshold})
        for split, d, probability, labels in [
            ('validation', validation, val_p, val_y),
            ('test', test, model.predict_proba(test[FEATURES]), model.predict(test[FEATURES]).astype(int))]:
            metric = classification_metrics(d.target, labels, probability, model_name != 'imbalance_rule')
            score = labels-1 if model_name == 'imbalance_rule' else probability[:, 2]-probability[:, 0]
            records['classification'].append({'symbol': symbol, 'fold': fold, 'model': model_name,
                'split': split, 'information_coefficient': information_coefficient(score, d.future_mid_return_bps), **metric})
            p = d[['symbol', 'date', 'time', 'target_time', 'source_row', 'target_row', 'target', 'future_mid_return_bps']].copy()
            p['split'], p['fold'], p['model'], p['prediction'], p['score'] = split, fold, model_name, labels, score
            p[['p_down', 'p_flat', 'p_up']] = probability
            records['predictions'].append(p)
            if split != 'test':
                continue
            fold_predictions[model_name] = p
            matrix = confusion_matrix(d.target, labels, labels=[0, 1, 2])
            for actual in range(3):
                for predicted in range(3):
                    records['confusion'].append({'symbol': symbol, 'fold': fold, 'model': model_name,
                        'actual': actual, 'predicted': predicted, 'count': int(matrix[actual, predicted])})
            policies = {'validation_selected': threshold}
            if model_name in ['logistic', 'xgboost']:
                policies['fixed_0.55_diagnostic'] = .55
            for policy, gate in policies.items():
                actions = labels-1 if model_name in ['majority', 'imbalance_rule'] else direction_from_probabilities(probability, gate)
                for latency in config.get('latency_sensitivity_ms', [0, 50, 200]):
                    for fee in config.get('fee_sensitivity_bps', [0., 1., 5.]):
                        trades = simulate_dates(sessions, d, actions, latency_ms=latency, fee_bps=fee,
                            horizon=config['horizon_seconds'], max_age=config['max_quote_age_seconds'])
                        records['economic'].append({'symbol': symbol, 'fold': fold, 'model': model_name,
                            'policy': policy, 'threshold': gate, 'latency_ms': latency, 'fee_bps_per_side': fee,
                            **trade_metrics(trades, len(d))})
                        if latency == config['latency_ms'] and fee == config['fee_bps_per_side']:
                            trades['symbol'], trades['fold'], trades['model'], trades['policy'] = symbol, fold, model_name, policy
                            records['trades'].append(trades)
        if model_name == 'logistic':
            for target, coefs in zip(model[-1].classes_, model[-1].coef_):
                records['coefficients'].extend({'symbol': symbol, 'fold': fold, 'class': int(target),
                    'feature': feature, 'standardised_coefficient': float(value)} for feature, value in zip(FEATURES, coefs))
    winner = max(validation_scores, key=lambda r: r[1])[0]
    for r in records['decisions']:
        if r['symbol'] == symbol and r['fold'] == fold:
            r['selected_model'] = r['model'] == winner
    chosen_prediction = fold_predictions[winner].copy()
    chosen_prediction['selected_model'] = winner
    chosen_prediction['model'] = 'validation_selected_model'
    records['predictions'].append(chosen_prediction)
    for split, frame in [('train', train), ('validation', validation), ('test', test)]:
        records['splits'].append({'symbol': symbol, 'fold': fold, 'split': split, 'n': len(frame),
            'first_date': frame.date.min(), 'last_date': frame.date.max(), 'first_seconds': frame.time.min(),
            'last_seconds': frame.time.max(), 'max_target_seconds': frame.target_time.max()})
    print(f'{symbol} fold {fold}: validation-selected {winner}; train/validation/test {len(train)}/{len(validation)}/{len(test)}', flush=True)


def run(root: Path, config_path: Path | None = None):
    config_path = config_path or root/'configs/study.json'
    config = json.loads(config_path.read_text())
    validate_config(config)
    result = root/'results'
    result.mkdir(exist_ok=True)
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
        'config_sha256': sha256(config_path.read_bytes()).hexdigest(),
        'plan_sha256': sha256((root/'docs/RESEARCH_PLAN.md').read_bytes()).hexdigest(),
        'python': platform.python_version(),
        'packages': {n: version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'xgboost', 'matplotlib', 'pyarrow']},
        'source_code_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted((root/'src/lob_research').glob('*.py'))}}
    (result/'run-start.json').write_text(json.dumps(metadata, indent=2)+'\n')
    sessions, observations, quality = {}, [], []
    for symbol in config['symbols']:
        paths = sorted((root/'data/raw').glob(f'{symbol}_*_message_10.csv'))
        if not paths:
            raise FileNotFoundError(f'No LOBSTER message files for {symbol}; run lob-research download first')
        sessions[symbol] = {}
        for path in paths:
            session = load_session(path, config['levels'])
            if session.date in sessions[symbol]:
                raise ValueError(f'Duplicate/overlapping source session for {symbol} {session.date}')
            sessions[symbol][session.date] = session
            observations.append(make_observations(session, config))
            quality.append({'symbol': symbol, 'date': session.date, **session.quality})
            print(f'Validated {symbol} {session.date}: {session.quality["source_rows"]:,} source rows', flush=True)
    if len({tuple(sorted(v)) for v in sessions.values()}) != 1:
        raise ValueError('All symbols must have the same date universe')
    frame = pd.concat(observations, ignore_index=True).sort_values(['symbol', 'date', 'time', 'source_row']).reset_index(drop=True)
    frame.to_parquet(result/'observations.parquet', index=False)
    pd.DataFrame(quality).to_csv(result/'data-quality.csv', index=False)
    names = ['classification', 'predictions', 'economic', 'trades', 'decisions', 'threshold_search', 'confusion', 'coefficients', 'splits']
    records = {name: [] for name in names}
    protocols = set()
    for symbol in config['symbols']:
        sample = frame.loc[frame.symbol == symbol].copy()
        for fold, train, val, test, protocol in walk_forward_folds(sample, config):
            protocols.add(protocol)
            evaluate_fold(symbol, sessions[symbol], train, val, test, fold, config, records)
    predictions = pd.concat(records['predictions'], ignore_index=True)
    predictions.to_parquet(result/'predictions.parquet', index=False)
    pd.concat(records['trades'], ignore_index=True).to_csv(result/'primary-scenario-trades.csv', index=False)
    mapping = {'classification': 'classification-by-fold.csv', 'economic': 'execution-sensitivity.csv',
        'decisions': 'validation-decisions.csv', 'threshold_search': 'threshold-search.csv',
        'confusion': 'confusion-matrices.csv', 'coefficients': 'logistic-coefficients.csv', 'splits': 'split-counts.csv'}
    for key, filename in mapping.items():
        pd.DataFrame(records[key]).to_csv(result/filename, index=False)
    aggregate = []
    for (symbol, model), p in predictions.loc[predictions.split == 'test'].groupby(['symbol', 'model']):
        p = p.sort_values(['date', 'time', 'source_row'])
        if p.duplicated(['date', 'source_row']).any():
            raise ValueError('Walk-forward test windows overlap')
        base = predictions.loc[(predictions.symbol == symbol) & (predictions.model == 'majority') & (predictions.split == 'test')]
        base = base.set_index(['date', 'source_row']).loc[p.set_index(['date', 'source_row']).index]
        metric = classification_metrics(p.target, p.prediction, p[['p_down', 'p_flat', 'p_up']].to_numpy(), model != 'imbalance_rule')
        metric['information_coefficient'] = information_coefficient(p.score, p.future_mid_return_bps)
        metric.update(block_ic_interval(p.score, p.future_mid_return_bps, p.time, p.date, config))
        metric.update(paired_block_interval(p.target.to_numpy(), p.prediction.to_numpy(), base.prediction.to_numpy(), p.time, p.date, config))
        aggregate.append({'symbol': symbol, 'model': model, 'split': 'stitched_test', **metric})
    pd.DataFrame(aggregate).to_csv(result/'classification.csv', index=False)
    metadata.update(completed_utc=datetime.now(timezone.utc).isoformat(), protocols=sorted(protocols),
        total_source_rows=sum(q['source_rows'] for q in quality), total_observations=len(frame),
        session_dates=sorted(frame.date.unique()), folds_per_symbol=int(pd.DataFrame(records['decisions']).fold.nunique()))
    metadata['outputs_sha256'] = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(result.glob('*.csv'))}
    (result/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    from .presentation import build_figures, write_findings
    build_figures(root)
    write_findings(root)
    return metadata
