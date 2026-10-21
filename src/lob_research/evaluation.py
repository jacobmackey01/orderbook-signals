"""Prediction metrics, paired block intervals and non-overlapping quote scenarios."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, log_loss


def macro_f1_from_matrix(matrix):
    matrix = np.asarray(matrix)
    tp = np.diagonal(matrix, axis1=-2, axis2=-1)
    denominator = matrix.sum(axis=-1) + matrix.sum(axis=-2)
    f1 = np.divide(2*tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator>0)
    return f1.mean(axis=-1)


def classification_metrics(y, prediction, probability, calibrated=True):
    probability = np.asarray(probability, dtype=float)
    totals = probability.sum(axis=1, keepdims=True)
    if np.any(probability < 0) or not np.allclose(totals, 1, atol=1e-5):
        raise ValueError("Invalid class probabilities")
    probability = probability/totals  # Normalise float32 tree outputs before float64 log loss.
    matrix = confusion_matrix(y, prediction, labels=[0, 1, 2])
    supports = matrix.sum(axis=1)
    recalls = np.divide(np.diag(matrix), supports, out=np.zeros(3), where=supports>0)
    directional = np.asarray(prediction) != 1
    return {"n": len(y), "macro_f1": float(macro_f1_from_matrix(matrix)),
            "balanced_accuracy": float(recalls.mean()), "accuracy": float(np.trace(matrix)/len(y)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1, 2])) if calibrated else None,
            "down_n": int(supports[0]), "flat_n": int(supports[1]), "up_n": int(supports[2]),
            "directional_hit_rate": float((np.asarray(y)[directional] == np.asarray(prediction)[directional]).mean()) if directional.any() else None,
            "directional_predictions": int(directional.sum()), "directional_coverage": float(directional.mean())}


def information_coefficient(score, returns):
    score, returns = np.asarray(score), np.asarray(returns)
    if len(score) < 3 or np.ptp(score) == 0 or np.ptp(returns) == 0:
        return None
    return float(spearmanr(score, returns).statistic)


def block_ic_interval(score, returns, time, dates, config):
    if information_coefficient(score, returns) is None:
        return {"ic_low": None, "ic_high": None}
    block = pd.factorize(pd.MultiIndex.from_arrays([dates, (np.asarray(time)//config["bootstrap_block_seconds"]).astype(int)]))[0]
    groups = [np.flatnonzero(block == k) for k in range(block.max()+1)]
    rng = np.random.default_rng(config["seed"])
    score, returns = np.asarray(score), np.asarray(returns)
    estimates = []
    for _ in range(config["bootstrap_repetitions"]):
        index = np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])
        estimate = information_coefficient(score[index], returns[index])
        if estimate is not None:
            estimates.append(estimate)
    low, high = np.quantile(estimates, [.025, .975])
    return {"ic_low": float(low), "ic_high": float(high)}


def paired_block_interval(y, prediction, baseline, time, dates, config):
    blocks = pd.factorize(pd.MultiIndex.from_arrays([dates, (np.asarray(time)//config["bootstrap_block_seconds"]).astype(int)]))[0]
    matrices = np.zeros((blocks.max()+1, 3, 3), dtype=int)
    controls = np.zeros_like(matrices)
    np.add.at(matrices, (blocks, np.asarray(y), np.asarray(prediction)), 1)
    np.add.at(controls, (blocks, np.asarray(y), np.asarray(baseline)), 1)
    rng = np.random.default_rng(config["seed"])
    sampled = rng.integers(len(matrices), size=(config["bootstrap_repetitions"], len(matrices)))
    delta = macro_f1_from_matrix(matrices[sampled].sum(axis=1)) - macro_f1_from_matrix(controls[sampled].sum(axis=1))
    low, high = np.quantile(delta, [.025, .975])
    return {"delta_macro_f1_low": float(low), "delta_macro_f1_high": float(high), "bootstrap_blocks": len(matrices)}


def direction_from_probabilities(probability, threshold):
    probability = np.asarray(probability)
    up, down = probability[:, 2], probability[:, 0]
    best = np.maximum(up, down)
    # Flat must not be the most probable outcome. Exact directional ties abstain.
    return np.where((best >= threshold) & (best > probability[:, 1]), np.sign(up-down), 0).astype(int)


def simulate(session, observations, actions, latency_ms=50, fee_bps=1., horizon=1., max_age=1.):
    if len(observations) != len(actions):
        raise ValueError("Observation/action lengths differ")
    if not observations.time.is_monotonic_increasing:
        raise ValueError("Execution decisions must be ordered")
    if not np.isin(actions, [-1, 0, 1]).all():
        raise ValueError("Actions must be -1, 0 or 1")
    t = observations.time.to_numpy()
    entry_time = t + latency_ms/1000
    target_time = observations.target_time.to_numpy() if "target_time" in observations else t+horizon
    exit_time = target_time + latency_ms/1000
    ei, eok = session.at(entry_time, max_age)
    xi, xok = session.at(exit_time, max_age)
    di, dok = session.at(t, max_age)
    if "source_row" in observations:
        di = observations.source_row.to_numpy(dtype=int)
        dok = session.valid[di]
        if latency_ms == 0:
            ei, eok = di.copy(), dok.copy()
    if latency_ms == 0 and "target_row" in observations:
        xi = observations.target_row.to_numpy(dtype=int)
        xok = session.valid[xi]
    entry_ok = eok & dok & (session.segment[di] == session.segment[ei])
    exit_ok = xok & (session.segment[ei] == session.segment[xi])
    rows, available_after = [], -np.inf
    for i in np.flatnonzero(entry_ok & (np.asarray(actions) != 0)):
        if t[i] <= available_after:
            continue
        if not exit_ok[i]:
            raise ValueError("Unpriced or interrupted exit after an eligible entry; cannot discard the trade using future information")
        action = int(actions[i])
        entry_col, exit_col = (0, 2) if action == 1 else (2, 0)
        if session.book[ei[i], entry_col+1] < 1:
            continue
        if session.book[xi[i], exit_col+1] < 1:
            raise ValueError("Exit lacks one-share displayed liquidity; no complete P&L can be reported")
        entry = session.book[ei[i], entry_col]/10000
        exit_ = session.book[xi[i], exit_col]/10000
        gross = action*(exit_-entry)
        fees = (entry+exit_)*fee_bps/10000
        rows.append({"time": t[i], "entry_time": entry_time[i], "exit_time": exit_time[i],
                     "entry_quote_time": session.time[ei[i]], "exit_quote_time": session.time[xi[i]],
                     "action": action, "entry": entry, "exit": exit_, "gross_dollars": gross,
                     "fees_dollars": fees, "net_dollars": gross-fees, "net_bps": (gross-fees)/entry*10000})
        available_after = exit_time[i]
    columns = ["time", "entry_time", "exit_time", "entry_quote_time", "exit_quote_time", "action", "entry", "exit", "gross_dollars", "fees_dollars", "net_dollars", "net_bps"]
    return pd.DataFrame(rows, columns=columns)


def trade_metrics(trades, n_decisions):
    return {"trades": len(trades), "total_net_dollars": float(trades.net_dollars.sum()),
            "mean_net_bps": float(trades.net_bps.mean()) if len(trades) else None,
            "participation": float(len(trades)/n_decisions) if n_decisions else 0.,
            "winning_trade_fraction": float((trades.net_dollars > 0).mean()) if len(trades) else None}


def simulate_dates(sessions, observations, actions, **kwargs):
    results = []
    for date in sorted(observations.date.unique()):
        mask = (observations.date == date).to_numpy()
        trades = simulate(sessions[date], observations.loc[mask], np.asarray(actions)[mask], **kwargs)
        trades["date"] = date
        results.append(trades)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def choose_threshold(sessions, validation, probabilities, config):
    choices = []
    for threshold in config["confidence_thresholds"]:
        actions = direction_from_probabilities(probabilities, threshold)
        trades = simulate_dates(sessions, validation, actions, latency_ms=config["latency_ms"], fee_bps=config["fee_bps_per_side"],
                                horizon=config["horizon_seconds"], max_age=config["max_quote_age_seconds"])
        metric = trade_metrics(trades, len(validation))
        if metric["trades"] >= config["min_validation_trades"] or threshold > 1:
            choices.append({"threshold": threshold, **metric})
    if not choices:
        raise ValueError("No abstention candidate in threshold grid")
    chosen = max(choices, key=lambda r: (r["total_net_dollars"], -r["trades"], r["threshold"]))
    return chosen, choices
