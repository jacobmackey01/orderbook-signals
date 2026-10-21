import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lob_research.data import Session, validate_arrays
from lob_research.economics import (TICK, components_raw, cost_aware_actions, decision_outcomes,
                                    decompose_trades, expected_return_bps, fee_dollars,
                                    fold_class_moves, class_conditional_moves, summarise_decomposition)
from lob_research.evaluation import simulate


def session_from_quotes(ask_cents, bid_cents, dt=.1):
    """Valid ten-level book whose best quotes follow the given paths (prices in cents)."""
    ask, bid = np.asarray(ask_cents, float)*100, np.asarray(bid_cents, float)*100
    n = len(ask)
    m = np.column_stack([np.arange(n)*dt, np.ones(n), np.arange(n), np.full(n, 10), bid, np.ones(n)])
    b = np.zeros((n, 40))
    for level in range(10):
        b[:, 4*level], b[:, 4*level+1] = ask+level*100, 100
        b[:, 4*level+2], b[:, 4*level+3] = bid-level*100, 100
    valid, q = validate_arrays(m, b, 10)
    return Session("TEST", "2020-01-01", m[:, 0], m[:, 1].astype(int), m[:, 3], m[:, 5],
                   b, valid, np.cumsum(~valid | np.isin(m[:, 1], [6, 7])), q)


def observations_for(session, rows, horizon_rows):
    rows = np.asarray(rows)
    return pd.DataFrame({"date": session.date, "time": session.time[rows], "source_row": rows,
                         "target_row": rows+horizon_rows, "target_time": session.time[rows+horizon_rows]})


def test_fee_arithmetic_is_per_side_on_traded_prices():
    assert fee_dollars(100., 101., 1.) == pytest.approx(.0201)
    assert fee_dollars(30., 30., 5.) == pytest.approx(.03)


def test_worked_example_half_tick_move_on_one_tick_spread():
    # $30.00 bid / $30.01 ask; the ask ticks up so the mid rises half a tick.
    s = session_from_quotes([3001, 3001, 3002], [3000, 3000, 3000])
    c = components_raw(s.book, np.array([1.]), np.array([0]), np.array([2]), np.array([0]), np.array([2]))
    assert c["signal"][0]/1e4 == pytest.approx(.005)        # +0.5 tick predicted correctly
    assert c["spread"][0]/1e4 == pytest.approx(-.015)       # half-spread in (0.5 tick) + out (now 1 tick)
    assert c["gross"][0]/1e4 == pytest.approx(-.01)         # buy 30.01, sell 30.00
    fee = fee_dollars(30.01, 30.00, 1.)
    assert (c["gross"][0]/1e4-fee)/TICK == pytest.approx(-1.6001)
    # Even a full one-tick mid move only breaks even before fees on a one-tick spread.
    s = session_from_quotes([3001, 3002], [3000, 3001])
    c = components_raw(s.book, np.array([1.]), np.array([0]), np.array([1]), np.array([0]), np.array([1]))
    assert c["gross"][0] == 0 and c["signal"][0] == 100


@pytest.mark.parametrize("latency", [0, 50, 200])
def test_decomposition_sums_exactly_to_simulator_net(latency):
    rng = np.random.default_rng(0)
    mid2 = 2*10000+np.cumsum(rng.integers(-1, 2, 400))   # mid in half-cents
    half = rng.integers(1, 4, 400)                         # half-spread in half-cents
    ask, bid = (mid2+half)/2, (mid2-half)/2
    s = session_from_quotes(ask, bid)
    obs = observations_for(s, np.arange(5, 380, 3), 10)
    actions = rng.choice([-1, 0, 1], len(obs))
    tr = decompose_trades(s, obs, actions, latency_ms=latency, fee_bps=1.)
    sim = simulate(s, obs, actions, latency_ms=latency, fee_bps=1.)
    assert len(tr) == len(sim) > 10
    total = tr.signal_dollars+tr.latency_dollars+tr.spread_dollars+tr.fee_1bp_dollars
    np.testing.assert_allclose(total, sim.net_dollars, rtol=0, atol=1e-12)
    np.testing.assert_allclose(tr.signal_dollars+tr.latency_dollars+tr.spread_dollars, sim.gross_dollars, rtol=0, atol=1e-12)
    if latency == 0:
        assert (tr.latency_dollars == 0).all()
    summary = summarise_decomposition(tr)
    parts = summary["signal_bps"]+summary["latency_bps"]+summary["spread_bps"]+summary["fee_1bp_bps"]
    assert parts == pytest.approx(summary["net_1bp_bps"], abs=1e-9)
    assert summary["net_1bp_bps"] == pytest.approx(sim.net_bps.mean(), abs=1e-9)
    assert summary["net_1bp_total_dollars"] == pytest.approx(sim.net_dollars.sum(), abs=1e-9)


def test_decision_outcome_matches_single_simulated_trade():
    s = session_from_quotes(np.r_[np.full(10, 3001), np.full(10, 3004)], np.r_[np.full(10, 3000), np.full(10, 3003)])
    obs = observations_for(s, [2], 10)
    o = decision_outcomes(s, obs, latency_ms=50, fee_bps=1.)
    long = simulate(s, obs, np.array([1]), latency_ms=50, fee_bps=1.)
    short = simulate(s, obs, np.array([-1]), latency_ms=50, fee_bps=1.)
    mid = 30.005
    assert o.long_net_bps.iloc[0] == pytest.approx(long.net_dollars.iloc[0]/mid*1e4)
    assert o.short_net_bps.iloc[0] == pytest.approx(short.net_dollars.iloc[0]/mid*1e4)
    assert o.mid_move_ticks.iloc[0] == pytest.approx(3.)


def test_cost_aware_rule_trades_only_when_expected_move_beats_cost():
    moves = {"m_up": 4., "m_down": 2.}
    p = np.array([[.1, .1, .8], [.8, .1, .1], [.0, .0, 1.], [.45, .1, .45]])
    np.testing.assert_allclose(expected_return_bps(p, moves), [3., -1.2, 4., .9])
    np.testing.assert_equal(cost_aware_actions(p, [2.9, 1., 3.9, .1], moves, 0.), [1, -1, 1, 1])
    np.testing.assert_equal(cost_aware_actions(p, [2.9, 1., 3.9, .1], moves, 1.), [0, 0, 0, 0])


def test_class_moves_use_training_rows_only():
    config = json.loads((Path(__file__).resolve().parents[1]/"configs/study.json").read_text())
    config.update(start_seconds=0., end_seconds=100.)
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"date": "2020-01-01", "time": np.arange(0, 100, .05)})
    frame["target_time"] = frame.time+.3
    frame["future_mid_return_bps"] = rng.normal(0, 2, len(frame))
    frame["target"] = np.sign(frame.future_mid_return_bps).astype(int)+1
    original = fold_class_moves(frame, config)
    for fold, moves in original.items():
        val_start = 100*(config["walk_forward_initial_train_fraction"]+(fold-1)*config["walk_forward_test_fraction"])
        changed = frame.copy()
        later = changed.time >= val_start
        changed.loc[later, "future_mid_return_bps"] *= 50   # future rows change drastically
        assert fold_class_moves(changed, config)[fold] == moves
    with pytest.raises(ValueError):
        class_conditional_moves(frame.assign(target=1))
