import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lob_research.data import (Session, FEATURES, validate_arrays, ofi_events, make_observations,
                               split_observations, window_sum, walk_forward_folds)
from lob_research.evaluation import (simulate, direction_from_probabilities, choose_threshold,
                                     macro_f1_from_matrix, paired_block_interval, information_coefficient)
from lob_research.models import Majority, fit_models
from lob_research.pipeline import validate_config


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1]/"configs/study.json"
    c = json.loads(path.read_text())
    c.update(start_seconds=2., end_seconds=18., sampling="clock")
    return c


def arrays(n=220):
    m = np.column_stack([np.arange(n)*.1, np.ones(n), np.arange(n), np.full(n, 10), np.full(n, 1000000), np.ones(n)])
    b = np.zeros((n, 40))
    for level in range(10):
        b[:, 4*level] = 1000200+level*100
        b[:, 4*level+1] = 100
        b[:, 4*level+2] = 1000000-level*100
        b[:, 4*level+3] = 100
    return m, b


def fixture_session(m=None, b=None):
    if m is None:
        m, b = arrays()
    valid, q = validate_arrays(m, b, 10)
    return Session("TEST", "2020-01-01", m[:, 0], m[:, 1].astype(int), m[:, 3], m[:, 5],
                   b, valid, np.cumsum(~valid | np.isin(m[:, 1], [6, 7])), q)


def test_paired_lengths_and_out_of_order_fail():
    m, b = arrays()
    with pytest.raises(ValueError, match="dimensions"):
        validate_arrays(m[:-1], b, 10)
    m[5, 0] = 0
    with pytest.raises(ValueError, match="out of order"):
        validate_arrays(m, b, 10)


def test_equal_timestamp_events_preserve_last_state():
    m, b = arrays()
    m[1, 0] = m[0, 0]
    b[1, 1] = 77
    s = fixture_session(m, b)
    idx, ok = s.at(np.array([0., -.1, 100.]))
    assert idx[0] == 1 and ok.tolist() == [True, False, False]
    assert s.book[idx[0], 1] == 77


def test_crossed_and_dummy_depth_are_excluded():
    m, b = arrays()
    b[3, 0] = b[3, 2]
    b[4, 36] = 9999999999
    b[4, 37] = 0
    valid, _ = validate_arrays(m, b, 10)
    assert not valid[3] and not valid[4] and valid[5]


def test_halt_until_resume_is_excluded():
    m, b = arrays()
    m[[20, 30, 40], 1] = 7
    m[[20, 30, 40], 4] = [-1, 0, 1]
    valid, _ = validate_arrays(m, b, 10)
    assert not valid[20:41].any()
    assert valid[41]


def test_ofi_sign_and_price_reset():
    b = np.array([[102, 10, 100, 10], [102, 10, 100, 15], [102, 10, 101, 7], [102, 10, 100, 20]], float)
    np.testing.assert_equal(ofi_events(b), [0, 5, 7, -7])


def test_window_is_left_open_and_right_closed():
    np.testing.assert_equal(window_sum(np.array([1, 2, 3]), np.array([10, 20, 30]), np.array([2]), 1), [20])


def test_future_changes_do_not_change_past_features(config):
    m, b = arrays()
    original = make_observations(fixture_session(m, b), config)
    changed = b.copy()
    changed[m[:, 0] > 10] += np.tile([100, 0, 100, 0], 10)
    perturbed = make_observations(fixture_session(m, changed), config)
    left = original.loc[original.time <= 10, FEATURES]
    right = perturbed.loc[perturbed.time <= 10, FEATURES]
    np.testing.assert_allclose(left, right)
    assert (original.loc[original.time == 9.5, "target"].iloc[0] != perturbed.loc[perturbed.time == 9.5, "target"].iloc[0])


def test_execution_resting_direction_is_inverted(config):
    m, b = arrays()
    m[80:100, 1] = 4
    m[80:100, 5] = -1
    d = make_observations(fixture_session(m, b), config)
    assert d.loc[d.time == 10, "trade_imbalance_5s"].iloc[0] == 1


def test_clock_split_purges_label_and_execution_endpoints(config):
    frame = pd.DataFrame({"date": "2020-01-01", "time": np.arange(2, 18, .1)})
    out, protocol = split_observations(frame, config)
    a, b = 11.6, 14.8
    assert (out.loc[out.split == "train", "time"] + 1.2 < a).all()
    assert (out.loc[out.split == "validation", "time"] + 1.2 < b).all()
    assert "within-session" in protocol


def test_multi_day_protocol_keeps_whole_dates(config):
    frame = pd.DataFrame({"date": np.repeat(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"], 10), "time": np.tile(np.arange(10), 5)})
    out, protocol = split_observations(frame, config)
    assert out.groupby("date").split.nunique().max() == 1
    assert set(out.split) == {"train", "validation", "test"}
    assert "complete" in protocol


def test_hand_calculated_long_short_and_fees():
    s = fixture_session()
    obs = pd.DataFrame({"time": [5., 5.5, 6., 7.]})
    trades = simulate(s, obs, np.array([1, 1, 1, -1]), latency_ms=0, fee_bps=1)
    # Constant two-cent spread; exact-exit-time decisions are skipped.
    assert trades.time.tolist() == [5., 7.]
    assert trades.entry.tolist() == [100.02, 100.]
    np.testing.assert_allclose(trades.gross_dollars, [-.02, -.02])
    np.testing.assert_allclose(trades.fees_dollars, [.020002, .020002])
    np.testing.assert_allclose(trades.net_dollars, [-.040002, -.040002])


def test_execution_uses_future_observable_quotes():
    m, b = arrays()
    b[m[:, 0] >= 5.1] += np.tile([100, 0, 100, 0], 10)
    trades = simulate(fixture_session(m, b), pd.DataFrame({"time": [5.]}), np.array([1]), latency_ms=200, fee_bps=0)
    assert trades.entry.iloc[0] == 100.03
    assert (trades.entry_quote_time <= trades.entry_time).all()
    assert (trades.exit_quote_time <= trades.exit_time).all()


def test_abstention_threshold_chosen_when_all_trades_lose(config):
    s = fixture_session()
    validation = pd.DataFrame({"date": s.date, "time": np.arange(5, 15, .5)})
    p = np.tile([.01, .01, .98], (len(validation), 1))
    config["min_validation_trades"] = 1
    chosen, _ = choose_threshold({s.date: s}, validation, p, config)
    assert chosen["threshold"] > 1 and chosen["trades"] == 0


def test_probability_gate_rejects_flat_winner_and_direction_ties():
    p = np.array([[.1, .7, .2], [.4, .2, .4], [.1, .2, .7]])
    np.testing.assert_equal(direction_from_probabilities(p, .3), [0, 0, 1])


def test_block_bootstrap_identical_prediction_delta_zero(config):
    y = np.tile(np.arange(3), 100)
    result = paired_block_interval(y, y, y, np.arange(300), np.repeat("2020-01-01", 300), config)
    assert result["delta_macro_f1_low"] == result["delta_macro_f1_high"] == 0
    assert macro_f1_from_matrix(np.eye(3)) == 1


def test_standard_scaler_is_fitted_only_on_training():
    x = pd.DataFrame(np.tile(np.arange(90)[:, None], (1, len(FEATURES))), columns=FEATURES)
    y = np.tile(np.arange(3), 30)
    models = fit_models(x.iloc[:60], y[:60])
    scaler = models["logistic"][0]
    np.testing.assert_allclose(scaler.mean_, np.repeat(29.5, len(FEATURES)))
    models["logistic"].predict(x.iloc[60:])
    np.testing.assert_allclose(scaler.mean_, np.repeat(29.5, len(FEATURES)))


def test_invalid_config_rejected(config):
    config["confidence_thresholds"] = [.5]
    with pytest.raises(ValueError, match="abstention"):
        validate_config(config)


def test_walk_forward_test_windows_are_disjoint_and_purged(config):
    frame = pd.DataFrame({"date": "2020-01-01", "time": np.arange(2, 18, .01)})
    frame["target_time"] = frame.time+.3
    folds = list(walk_forward_folds(frame, config))
    assert len(folds) == 5
    test_indices = []
    for _, train, val, test, _ in folds:
        assert train.target_time.max()+.2 < val.time.min()
        assert val.target_time.max()+.2 < test.time.min()
        test_indices.extend(test.index)
    assert len(test_indices) == len(set(test_indices))
    assert len(folds[-1][1]) > len(folds[0][1])


def test_event_features_do_not_read_later_equal_timestamp_events(config):
    config.update(sampling="event", event_stride=1, horizon_events=10)
    m, b = arrays()
    m[101, 0] = m[100, 0]
    changed = b.copy()
    changed[101, 3] = 99999
    original = make_observations(fixture_session(m, b), config)
    altered = make_observations(fixture_session(m, changed), config)
    np.testing.assert_allclose(original.loc[original.source_row == 100, FEATURES], altered.loc[altered.source_row == 100, FEATURES])
    assert not np.allclose(original.loc[original.source_row == 101, FEATURES], altered.loc[altered.source_row == 101, FEATURES])


def test_event_target_is_exact_source_event_horizon(config):
    config.update(sampling="event", event_stride=3, horizon_events=10)
    d = make_observations(fixture_session(), config)
    assert ((d.target_row-d.source_row) == 10).all()


def test_zero_latency_does_not_read_later_equal_timestamp_quote():
    m, b = arrays()
    m[51, 0] = m[50, 0]
    b[51] += np.tile([100, 0, 100, 0], 10)
    s = fixture_session(m, b)
    d = pd.DataFrame({"time": [5.], "target_time": [6.], "source_row": [50], "target_row": [60]})
    trade = simulate(s, d, np.array([1]), latency_ms=0, fee_bps=0)
    assert trade.entry.iloc[0] == 100.02


def test_rank_ic_and_undefined_constant_score():
    assert information_coefficient([1, 2, 3], [10, 20, 30]) == 1
    assert information_coefficient([1, 1, 1], [10, 20, 30]) is None


def test_missing_exit_cannot_silently_remove_a_loss():
    s = fixture_session()
    d = pd.DataFrame({"time": [5.], "target_time": [100.]})
    with pytest.raises(ValueError, match="cannot discard"):
        simulate(s, d, np.array([1]), latency_ms=0)
