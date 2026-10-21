"""Why positive IC coexists with negative net P&L: a diagnostic cost decomposition.

Everything here re-reads the saved walk-forward test windows of one session, so the
results are DIAGNOSTIC (already-seen windows), not out-of-sample confirmation.
Execution uses historical displayed quotes only: one share, taker entry and exit
across the spread, no passive fills, no queue model, no annualisation.

Run:  ./scripts/python -m lob_research.economics [--root .]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_session, walk_forward_folds
from .evaluation import direction_from_probabilities, simulate

TICK = 0.01           # US equity tick in dollars (prices here are all above $1)
PRICE_SCALE = 10000   # LOBSTER prices are integer dollars x 10,000
FEES = (0., 1., 5.)
LATENCIES = (0, 50, 200)
PROB = ["p_down", "p_flat", "p_up"]


# --------------------------------------------------------------------------- quotes
def quote_indices(session, observations, latency_ms, horizon=1., max_age=1.):
    """Decision, entry and exit book rows exactly as ``evaluation.simulate`` chooses them.

    ``decompose_trades`` re-runs ``simulate`` and asserts that every trade matches,
    so a change to the simulator's quote rules cannot silently diverge from this.
    """
    t = observations.time.to_numpy()
    target_time = observations.target_time.to_numpy() if "target_time" in observations else t+horizon
    ei, eok = session.at(t+latency_ms/1000, max_age)
    xi, xok = session.at(target_time+latency_ms/1000, max_age)
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
    return di, ei, xi, entry_ok, exit_ok


def components_raw(book, action, di, ti, ei, xi):
    """Exact P&L components in raw LOBSTER price units (half-integers, exact in float).

    gross = a*(exit - entry) = signal + latency + spread, where
      signal  = a*(mid[target] - mid[decision])            pure mid-to-mid prediction
      latency = a*((mid[exit] - mid[entry]) - (mid[target] - mid[decision]))
      spread  = -(half_spread[entry] + half_spread[exit])  always a cost for a taker
    """
    a = np.asarray(action, dtype=float)
    ask, bid = book[:, 0], book[:, 2]
    mid = (ask+bid)/2
    half = (ask-bid)/2
    signal = a*(mid[ti]-mid[di])
    latency = a*((mid[xi]-mid[ei])-(mid[ti]-mid[di]))
    spread = -(half[ei]+half[xi])
    entry = np.where(a > 0, ask[ei], bid[ei])
    exit_ = np.where(a > 0, bid[xi], ask[xi])
    return {"signal": signal, "latency": latency, "spread": spread, "entry": entry, "exit": exit_,
            "gross": a*(exit_-entry)}


def fee_dollars(entry_dollars, exit_dollars, fee_bps):
    """Proportional fee charged on both sides, as in ``evaluation.simulate``."""
    return (np.asarray(entry_dollars)+np.asarray(exit_dollars))*fee_bps/10000


# ---------------------------------------------------------------- trade decomposition
def decompose_trades(session, observations, actions, latency_ms=50, fee_bps=1., horizon=1., max_age=1.):
    """Split each non-overlapping simulator trade into signal, latency, spread and fees.

    Trades are those of ``evaluation.simulate`` (called here); the function fails if
    its own replay of the selection disagrees with the simulator on any trade.
    """
    trades = simulate(session, observations, actions, latency_ms=latency_ms, fee_bps=fee_bps,
                      horizon=horizon, max_age=max_age)
    di, ei, xi, entry_ok, _ = quote_indices(session, observations, latency_ms, horizon, max_age)
    t = observations.time.to_numpy()
    target_time = observations.target_time.to_numpy() if "target_time" in observations else t+horizon
    exit_time = target_time+latency_ms/1000
    actions = np.asarray(actions)
    chosen, available_after = [], -np.inf
    for i in np.flatnonzero(entry_ok & (actions != 0)):
        if t[i] <= available_after:
            continue
        entry_col = 0 if actions[i] == 1 else 2
        if session.book[ei[i], entry_col+1] < 1:
            continue
        chosen.append(i)
        available_after = exit_time[i]
    chosen = np.asarray(chosen, dtype=int)
    if len(chosen) != len(trades):
        raise AssertionError(f"Trade replay found {len(chosen)} trades; simulator found {len(trades)}")
    ti = observations.target_row.to_numpy(dtype=int)
    c = components_raw(session.book, actions[chosen], di[chosen], ti[chosen], ei[chosen], xi[chosen])
    out = pd.DataFrame({"time": t[chosen], "action": actions[chosen].astype(int),
                        "entry": c["entry"]/PRICE_SCALE, "exit": c["exit"]/PRICE_SCALE,
                        "decision_mid": (session.book[di[chosen], 0]+session.book[di[chosen], 2])/2/PRICE_SCALE})
    for name in ["signal", "latency", "spread", "gross"]:
        out[f"{name}_dollars"] = c[name]/PRICE_SCALE
    for fee in FEES:
        out[f"fee_{fee:g}bp_dollars"] = -fee_dollars(out.entry, out.exit, fee)
    out["net_dollars"] = out.gross_dollars-fee_dollars(out.entry, out.exit, fee_bps)
    if len(trades):
        checks = {"time": (out.time, trades.time), "entry": (out.entry, trades.entry),
                  "exit": (out.exit, trades.exit), "net": (out.net_dollars, trades.net_dollars)}
        for name, (mine, theirs) in checks.items():
            if not np.allclose(np.asarray(mine), np.asarray(theirs), rtol=0, atol=1e-9):
                raise AssertionError(f"Decomposition disagrees with simulator on {name}")
    out["simulator_net_dollars"] = trades.net_dollars.to_numpy() if len(trades) else np.array([])
    return out


def summarise_decomposition(trades):
    """Mean per trade in bp of entry price (as the simulator's net_bps) and in ticks."""
    parts = ["signal", "latency", "spread"]+[f"fee_{f:g}bp" for f in FEES]
    row = {"trades": len(trades)}
    if not len(trades):
        return row
    for p in parts:
        d = trades[f"{p}_dollars"]
        row[f"{p}_bps"] = float((d/trades.entry*10000).mean())
        row[f"{p}_ticks"] = float((d/TICK).mean())
    gross = trades.signal_dollars+trades.latency_dollars+trades.spread_dollars
    for f in FEES:
        net = gross+trades[f"fee_{f:g}bp_dollars"]
        row[f"net_{f:g}bp_bps"] = float((net/trades.entry*10000).mean())
        row[f"net_{f:g}bp_ticks"] = float((net/TICK).mean())
        row[f"net_{f:g}bp_total_dollars"] = float(net.sum())
    row["signal_total_dollars"] = float(trades.signal_dollars.sum())
    row["mean_entry_price"] = float(trades.entry.mean())
    costs = {"spread": -row["spread_bps"], "latency": -row["latency_bps"], "fee_1bp": -row["fee_1bp_bps"]}
    row["largest_cost_at_1bp"] = max(costs, key=costs.get)
    return row


# --------------------------------------------------------------- per-decision outcomes
def decision_outcomes(session, observations, latency_ms=50, fee_bps=1., horizon=1., max_age=1.):
    """Taker long and short outcomes for EVERY decision (overlapping; descriptive only).

    bp are relative to the decision mid. ``valid`` requires priced, uninterrupted
    entry and exit quotes with at least one displayed share on the side crossed.
    """
    di, ei, xi, entry_ok, exit_ok = quote_indices(session, observations, latency_ms, horizon, max_age)
    book = session.book
    ti = observations.target_row.to_numpy(dtype=int)
    long = components_raw(book, np.ones(len(di)), di, ti, ei, xi)
    short = components_raw(book, -np.ones(len(di)), di, ti, ei, xi)
    mid = (book[di, 0]+book[di, 2])/2
    valid = entry_ok & exit_ok & (book[ei, 1] >= 1) & (book[ei, 3] >= 1) & (book[xi, 1] >= 1) & (book[xi, 3] >= 1)
    out = pd.DataFrame(index=observations.index)
    out["valid"] = valid
    out["decision_mid"] = mid/PRICE_SCALE
    out["spread_ticks"] = (book[di, 0]-book[di, 2])/PRICE_SCALE/TICK
    out["mid_move_ticks"] = long["signal"]/PRICE_SCALE/TICK
    for name, c in [("long", long), ("short", short)]:
        net = c["gross"]-fee_dollars(c["entry"], c["exit"], fee_bps)
        out[f"{name}_net_bps"] = net/mid*10000
    return out


# ----------------------------------------------------------------- cost-aware rule
def class_conditional_moves(train):
    """Mean |future mid return| (bp) given realised up / down, from TRAINING rows only."""
    r, y = train.future_mid_return_bps.to_numpy(), train.target.to_numpy()
    if not (y == 2).any() or not (y == 0).any():
        raise ValueError("Training rows must contain realised up and down moves")
    return {"m_up": float(r[y == 2].mean()), "m_down": float(np.abs(r[y == 0]).mean())}


def expected_return_bps(probability, moves):
    probability = np.asarray(probability, dtype=float)
    return probability[:, 2]*moves["m_up"]-probability[:, 0]*moves["m_down"]


def cost_aware_actions(probability, spread_bps, moves, fee_bps):
    """Trade only if the model-implied mid move exceeds spread + two fees (break-even margin 0)."""
    expected = expected_return_bps(probability, moves)
    cost = np.asarray(spread_bps, dtype=float)+2*fee_bps
    return np.where(expected > cost, 1, np.where(-expected > cost, -1, 0)).astype(int)


def fold_class_moves(frame, config):
    """Per-fold training estimates; only rows inside each fold's training set are read."""
    return {fold: class_conditional_moves(train) for fold, train, *_ in walk_forward_folds(frame, config)}


# ------------------------------------------------------------------------ analysis
def _decile(values):
    return pd.qcut(pd.Series(values).rank(method="first"), 10, labels=False)+1


def run_economics(root: Path):
    config = json.loads((root/"configs/study.json").read_text())
    res = root/"results"
    out = res/"economics"
    out.mkdir(parents=True, exist_ok=True)
    obs = pd.read_parquet(res/"observations.parquet")
    pred = pd.read_parquet(res/"predictions.parquet")
    pred = pred.merge(obs[["symbol", "date", "source_row", "spread_bps", "mid"]], on=["symbol", "date", "source_row"],
                      how="left", validate="many_to_one")
    if pred.spread_bps.isna().any():
        raise ValueError("Predictions do not match observations")
    sessions = {}
    for symbol in config["symbols"]:
        for path in sorted((root/"data/raw").glob(f"{symbol}_*_message_10.csv")):
            s = load_session(path, config["levels"])
            sessions[(symbol, s.date)] = s
    kw = dict(horizon=config["horizon_seconds"], max_age=config["max_quote_age_seconds"])
    order = ["date", "time", "source_row"]

    def groups(frame):
        for (symbol, fold, date), g in frame.groupby(["symbol", "fold", "date"], sort=True):
            yield symbol, fold, sessions[(symbol, date)], g.sort_values(order).reset_index(drop=True)

    test = pred.loc[pred.split == "test"]
    # 1. Decomposition of the same non-overlapping trades as the simulator.
    rows, recon, trade_frames = [], [], []
    policies = {"fixed_0.55_diagnostic": lambda g: direction_from_probabilities(g[PROB].to_numpy(), .55),
                "every_directional_prediction": lambda g: g.prediction.to_numpy()-1}
    for model in ["logistic", "xgboost", "validation_selected_model"]:
        for policy, rule in policies.items():
            for latency in LATENCIES:
                per_symbol = {}
                for symbol, fold, session, g in groups(test.loc[test.model == model]):
                    tr = decompose_trades(session, g, rule(g), latency_ms=latency, fee_bps=config["fee_bps_per_side"], **kw)
                    tr["symbol"], tr["fold"] = symbol, fold
                    per_symbol.setdefault(symbol, []).append(tr)
                    recon.append({"symbol": symbol, "fold": fold, "model": model, "policy": policy, "latency_ms": latency,
                                  "trades": len(tr), "max_abs_component_minus_simulator_dollars":
                                  float(np.abs(tr.signal_dollars+tr.latency_dollars+tr.spread_dollars
                                               + tr["fee_1bp_dollars"]-tr.simulator_net_dollars).max()) if len(tr) else 0.})
                for symbol, frames in per_symbol.items():
                    tr = pd.concat(frames, ignore_index=True)
                    rows.append({"symbol": symbol, "model": model, "policy": policy, "latency_ms": latency,
                                 **summarise_decomposition(tr)})
                    if latency == config["latency_ms"]:
                        trade_frames.append(tr.assign(model=model, policy=policy))
    decomposition = pd.DataFrame(rows)
    decomposition.to_csv(out/"decomposition.csv", index=False)
    recon = pd.DataFrame(recon)
    saved = pd.read_csv(res/"execution-sensitivity.csv")
    saved = saved.loc[(saved.policy == "fixed_0.55_diagnostic") & (saved.fee_bps_per_side == config["fee_bps_per_side"])]
    mine = pd.concat(trade_frames).groupby(["symbol", "fold", "model", "policy"]).simulator_net_dollars.sum().reset_index()
    check = mine.merge(saved.assign(latency_ms=saved.latency_ms), on=["symbol", "fold", "model", "policy"])
    check = check.loc[check.latency_ms == config["latency_ms"]]
    recon.attrs["saved_run_max_abs_diff"] = float((check.simulator_net_dollars-check.total_net_dollars).abs().max())
    recon.to_csv(out/"reconciliation.csv", index=False)

    # Per-decision outcomes for every test decision (overlapping, descriptive).
    base = test.loc[test.model == "validation_selected_model"]
    outcome_frames = {}
    for latency in [0, config["latency_ms"]]:
        for fee in FEES:
            parts = []
            for symbol, fold, session, g in groups(base):
                o = decision_outcomes(session, g, latency_ms=latency, fee_bps=fee, **kw)
                parts.append(pd.concat([g[["symbol", "fold", "date", "source_row"]], o], axis=1))
            outcome_frames[(latency, fee)] = pd.concat(parts, ignore_index=True)
    primary = outcome_frames[(config["latency_ms"], config["fee_bps_per_side"])]

    # 2. Signal magnitude by score decile versus the round-trip cost.
    sel = base.sort_values(["symbol"]+order).reset_index(drop=True)
    sel = sel.merge(primary[["symbol", "date", "source_row", "spread_ticks", "mid_move_ticks"]], on=["symbol", "date", "source_row"])
    dec_rows = []
    for symbol, g in sel.groupby("symbol"):
        g = g.assign(decile=_decile(g.score.to_numpy()).to_numpy())
        for d, h in g.groupby("decile"):
            r = h.future_mid_return_bps
            dec_rows.append({"symbol": symbol, "score_decile": int(d), "n": len(h), "mean_score": h.score.mean(),
                             "mean_future_mid_return_bps": r.mean(), "mean_future_mid_move_ticks": h.mid_move_ticks.mean(),
                             "mean_abs_mid_move_ticks": h.mid_move_ticks.abs().mean(),
                             "mean_spread_bps": h.spread_bps.mean(), "mean_spread_ticks": h.spread_ticks.mean(),
                             "round_trip_cost_bps_1bp_fee": h.spread_bps.mean()+2*config["fee_bps_per_side"],
                             "abs_mean_move_exceeds_spread": bool(abs(r.mean()) > h.spread_bps.mean()),
                             "frac_decisions_abs_move_gt_spread": float((r.abs() > h.spread_bps).mean())})
    pd.DataFrame(dec_rows).to_csv(out/"signal-vs-spread-by-decile.csv", index=False)
    moves = sel.mid_move_ticks.abs()
    dist = sel.assign(bucket=np.select([moves == 0, moves == .5, moves == 1, moves > 1], ["0", "0.5", "1", ">1"], "other"))
    dist = dist.groupby("symbol").bucket.value_counts(normalize=True).unstack(fill_value=0).reset_index()
    dist = dist.merge(sel.groupby("symbol").agg(median_spread_ticks=("spread_ticks", "median"),
                                                median_mid=("mid", "median"), median_spread_bps=("spread_bps", "median")).reset_index())
    dist.to_csv(out/"mid-move-distribution.csv", index=False)

    # 3. Confidence versus expected monetary return (per decision, overlapping).
    conf_rows = []
    for model in ["logistic", "xgboost"]:
        m = test.loc[test.model == model, ["symbol", "date", "source_row", "target", "future_mid_return_bps"]+PROB]
        m = m.merge(primary, on=["symbol", "date", "source_row"]).loc[lambda x: x.valid]
        direction = np.sign(m.p_up-m.p_down).to_numpy()
        m = m.assign(confidence=np.maximum(m.p_up, m.p_down), direction=direction,
                     hit=(m.target.to_numpy()-1) == direction,
                     favourable_ticks=direction*m.mid_move_ticks,
                     net_bps=np.where(direction > 0, m.long_net_bps, np.where(direction < 0, m.short_net_bps, np.nan)))
        m = m.loc[m.direction != 0]
        for symbol, g in m.groupby("symbol"):
            g = g.assign(decile=_decile(g.confidence.to_numpy()).to_numpy())
            for d, h in g.groupby("decile"):
                conf_rows.append({"symbol": symbol, "model": model, "confidence_decile": int(d), "n": len(h),
                                  "mean_confidence": h.confidence.mean(), "hit_rate": h.hit.mean(),
                                  "mean_favourable_mid_move_ticks": h.favourable_ticks.mean(),
                                  "mean_favourable_mid_move_bps": (h.direction*h.future_mid_return_bps).mean(),
                                  "mean_spread_ticks": h.spread_ticks.mean(), "mean_net_bps_50ms_1bp": h.net_bps.mean(),
                                  "frac_net_positive": float((h.net_bps > 0).mean())})
    pd.DataFrame(conf_rows).to_csv(out/"confidence-vs-return.csv", index=False)

    # 4. Perfect-foresight ceiling (explicit look-ahead).
    pf_rows = []
    for (latency, fee), o in outcome_frames.items():
        best = np.maximum(o.long_net_bps, o.short_net_bps)
        o = o.assign(best=best)
        for symbol, g in o.loc[o.valid].groupby("symbol"):
            pos = g.best > 0
            pf_rows.append({"symbol": symbol, "latency_ms": latency, "fee_bps_per_side": fee, "decisions": len(g),
                            "frac_mid_moved": float((g.mid_move_ticks != 0).mean()),
                            "frac_profitable_with_foresight": float(pos.mean()),
                            "mean_best_net_bps_if_profitable": float(g.best[pos].mean()) if pos.any() else np.nan})
    pf = pd.DataFrame(pf_rows)
    oracle = []
    for symbol, fold, session, g in groups(base):
        o = decision_outcomes(session, g, latency_ms=config["latency_ms"], fee_bps=config["fee_bps_per_side"], **kw)
        act = np.where(o.valid & (o.long_net_bps > 0), 1, np.where(o.valid & (o.short_net_bps > 0), -1, 0))
        tr = simulate(session, g, act, latency_ms=config["latency_ms"], fee_bps=config["fee_bps_per_side"], **kw)
        oracle.append({"symbol": symbol, "trades": len(tr), "net": tr.net_dollars.sum()})
    oracle = pd.DataFrame(oracle).groupby("symbol").sum().reset_index()
    pf = pf.merge(oracle.rename(columns={"trades": "oracle_nonoverlap_trades_50ms_1bp",
                                         "net": "oracle_nonoverlap_net_dollars_50ms_1bp"}), on="symbol")
    pf.to_csv(out/"perfect-foresight.csv", index=False)

    # 5. Pre-specified cost-aware rule (docs/ECONOMICS_PLAN.md).
    rule_rows, move_rows = [], []
    for symbol in config["symbols"]:
        frame = obs.loc[obs.symbol == symbol].sort_values(order).reset_index(drop=True)
        for fold, moves in fold_class_moves(frame, config).items():
            move_rows.append({"symbol": symbol, "fold": fold, **moves})
            for model in ["logistic", "xgboost"]:
                for split in ["validation", "test"]:
                    g = pred.loc[(pred.symbol == symbol) & (pred.fold == fold) & (pred.model == model) & (pred.split == split)]
                    g = g.sort_values(order).reset_index(drop=True)
                    prob = g[PROB].to_numpy()
                    act = cost_aware_actions(prob, g.spread_bps, moves, config["fee_bps_per_side"])
                    expected = expected_return_bps(prob, moves)
                    session = sessions[(symbol, g.date.iloc[0])]
                    tr = simulate(session, g, act, latency_ms=config["latency_ms"], fee_bps=config["fee_bps_per_side"], **kw)
                    rule_rows.append({"symbol": symbol, "fold": fold, "model": model, "split": split, "decisions": len(g),
                                      "frac_signal_exceeds_cost": float((act != 0).mean()),
                                      "mean_abs_expected_bps": float(np.abs(expected).mean()),
                                      "max_abs_expected_bps": float(np.abs(expected).max()),
                                      "mean_cost_bps": float(g.spread_bps.mean()+2*config["fee_bps_per_side"]),
                                      "trades": len(tr), "net_dollars": float(tr.net_dollars.sum()),
                                      "mean_net_bps": float(tr.net_bps.mean()) if len(tr) else np.nan})
    pd.DataFrame(move_rows).to_csv(out/"cost-aware-class-moves.csv", index=False)
    pd.DataFrame(rule_rows).to_csv(out/"cost-aware-rule.csv", index=False)
    build_figures(out)
    summary = {"diagnostic": "computed on already-seen walk-forward test windows of one session",
               "max_abs_component_reconciliation_dollars": float(recon.max_abs_component_minus_simulator_dollars.max()),
               "max_abs_diff_vs_saved_execution_sensitivity_dollars": recon.attrs["saved_run_max_abs_diff"]}
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


# ------------------------------------------------------------------------- figures
COLORS = {"signal": "#2a78d6", "spread": "#eb6834", "latency": "#1baf7a", "fee": "#eda100"}


def build_figures(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
    d = pd.read_csv(out/"decomposition.csv")
    d = d.loc[(d.model == "xgboost") & (d.latency_ms == 50)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False, layout="constrained")
    for ax, unit in zip(axes, ["bps", "ticks"]):
        for k, policy in enumerate(["fixed_0.55_diagnostic", "every_directional_prediction"]):
            p = d.loc[d.policy == policy].set_index("symbol")
            x = np.arange(len(p))+(k-.5)*.4
            pos = np.zeros(len(p)); neg = np.zeros(len(p))
            for part, col in [("signal", "signal"), ("latency", "latency"), ("spread", "spread"), ("fee_1bp", "fee")]:
                v = p[f"{part}_{unit}"].to_numpy()
                bottom = np.where(v >= 0, pos, neg)
                ax.bar(x, v, .36, bottom=bottom, color=COLORS[col], edgecolor="white", linewidth=1,
                       label=(part.replace("fee_1bp", "fees (1 bp/side)") if k == 0 else None),
                       hatch=None if k == 0 else "//")
                pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
            ax.scatter(x, p[f"net_1bp_{unit}"], color="black", s=22, zorder=3, label="net" if k == 0 else None)
        ax.set_xticks(np.arange(len(p)), p.index)
        ax.axhline(0, color="black", lw=.8)
        ax.set_ylabel(f"mean per trade ({'bp of entry price' if unit == 'bps' else 'ticks, $0.01'})")
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    axes[1].text(.99, .02, "left bar: fixed 0.55 confidence\nright bar (hatched): every directional prediction",
                 transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8)
    fig.suptitle("Where the P&L goes · XGBoost · 50 ms latency · same trades as the simulator (diagnostic, seen test windows)",
                 fontsize=11)
    fig.savefig(out/"decomposition.png", dpi=150)
    plt.close(fig)

    s = pd.read_csv(out/"signal-vs-spread-by-decile.csv")
    syms = list(s.symbol.unique())
    fig, axes = plt.subplots(1, len(syms), figsize=(14, 3.8), sharex=True, layout="constrained")
    for ax, sym in zip(axes, syms):
        g = s.loc[s.symbol == sym]
        ax.bar(g.score_decile, g.mean_future_mid_move_ticks, color=COLORS["signal"], width=.75, label="mean mid move")
        half = g.mean_spread_ticks.mean()
        ax.axhline(half, color=COLORS["spread"], ls="--", lw=1.5, label="spread (round-trip crossing cost)")
        ax.axhline(-half, color=COLORS["spread"], ls="--", lw=1.5)
        ax.axhline(0, color="black", lw=.8)
        ax.set_title(f"{sym}  (spread {g.mean_spread_bps.mean():.1f} bp)", fontsize=10)
        ax.set_xticks([1, 5, 10])
        ax.set_xlabel("score decile")
    axes[0].set_ylabel("ticks ($0.01)")
    axes[0].legend(loc="center left", fontsize=7, frameon=False)
    fig.suptitle("Expected mid move by model-score decile vs the spread a taker pays · validation-selected model, test windows",
                 fontsize=11)
    fig.savefig(out/"move-vs-spread-by-decile.png", dpi=150)
    plt.close(fig)

    c = pd.read_csv(out/"confidence-vs-return.csv")
    c = c.loc[c.model == "xgboost"]
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for color, sym in zip(palette, syms):
        g = c.loc[c.symbol == sym]
        axes[0].plot(g.confidence_decile, g.hit_rate, marker="o", ms=4, lw=2, color=color, label=sym)
        axes[1].plot(g.confidence_decile, g.mean_net_bps_50ms_1bp, marker="o", ms=4, lw=2, color=color, label=sym)
    axes[0].set_ylabel("directional hit rate"); axes[1].set_ylabel("mean net bp per decision (50 ms, 1 bp/side)")
    axes[1].axhline(0, color="black", lw=.8)
    for ax in axes:
        ax.set_xlabel("confidence decile, max(P(up), P(down))")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Higher confidence raises the hit rate but net return stays negative · XGBoost, every test decision (overlapping)",
                 fontsize=11)
    fig.savefig(out/"confidence-vs-net.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Diagnostic economics of the LOB signal study")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.figures_only:
        build_figures(root/"results/economics")
    else:
        print(json.dumps(run_economics(root), indent=2))


if __name__ == "__main__":
    main()
