"""Simple backtesting tools for the STOXX 600 project.

The first implementation is deliberately transparent: it builds positions from
slow momentum, fast reversion and optional CPD scores, then aggregates daily
stock PnL into an equal-weight portfolio.
"""

from __future__ import annotations

import sys
from typing import Callable, Iterable

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd

from src.metrics import (
    annual_return,
    annual_volatility,
    calmar,
    hit_ratio,
    max_drawdown,
    sharpe,
    sortino,
)


def squash_signal(values, scale=0.20):
    """Bound noisy return signals into the interval [-1, 1]."""
    values = pd.to_numeric(values, errors="coerce")
    return np.tanh(values / scale)


def add_rule_based_signals(
    panel,
    slow_col="252d_arith_ret",
    fast_col="21d_arith_ret",
    cpd_col="ensemble_score",
    signal_shift=1,
):
    """Add slow momentum, fast reversion and CPD-adjusted position signals."""
    out = panel.sort_values(["ticker", "date"]).copy()

    out["slow_momentum_signal"] = squash_signal(out[slow_col], scale=0.25)
    out["fast_reversion_signal"] = -squash_signal(out[fast_col], scale=0.08)

    if cpd_col in out.columns:
        cpd = pd.to_numeric(out[cpd_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        cpd = 0.0

    out["slow_fast_signal"] = (
        0.70 * out["slow_momentum_signal"]
        + 0.30 * out["fast_reversion_signal"]
    )
    out["cpd_adjusted_signal"] = (
        (1.0 - cpd) * out["slow_momentum_signal"]
        + cpd * out["fast_reversion_signal"]
    )

    signal_cols = [
        "slow_momentum_signal",
        "slow_fast_signal",
        "cpd_adjusted_signal",
    ]
    for col in signal_cols:
        position_col = col.replace("_signal", "_position")
        out[position_col] = (
            out.groupby("ticker", sort=False)[col]
            .shift(signal_shift)
            .clip(-1.0, 1.0)
        )

    return out


def portfolio_returns(
    panel,
    position_col,
    return_col="1d_arith_ret",
    cost_bps=0.0,
    target_vol=0.15,
    vol_window=60,
    max_leverage=3.0,
):
    """Aggregate stock-level positions into daily gross and net returns."""
    data = panel[["date", "ticker", position_col, return_col]].copy()
    data = data.dropna(subset=[position_col, return_col])
    data = data.sort_values(["ticker", "date"])
    data["stock_pnl"] = data[position_col] * data[return_col]
    data["turnover"] = (
        data.groupby("ticker", sort=False)[position_col]
        .diff()
        .abs()
        .fillna(data[position_col].abs())
    )

    daily = (
        data.groupby("date", sort=True)
        .agg(
            gross_return=("stock_pnl", "mean"),
            turnover=("turnover", "mean"),
            n_assets=("ticker", "nunique"),
        )
        .reset_index()
    )
    daily["cost"] = daily["turnover"] * cost_bps / 10000.0
    daily["net_return_unscaled"] = daily["gross_return"] - daily["cost"]

    realised_vol = (
        daily["net_return_unscaled"]
        .rolling(vol_window, min_periods=max(20, vol_window // 2))
        .std()
        * np.sqrt(252)
    )
    leverage = (target_vol / realised_vol).replace([np.inf, -np.inf], np.nan)
    daily["leverage"] = leverage.shift(1).clip(upper=max_leverage).fillna(1.0)
    daily["net_return"] = daily["net_return_unscaled"] * daily["leverage"]
    daily["strategy"] = position_col.replace("_position", "")
    return daily


def performance_summary(returns_by_strategy):
    """Create numeric performance metrics for each strategy."""
    rows = []
    for strategy, frame in returns_by_strategy.groupby("strategy"):
        returns = frame["net_return"].dropna()
        if returns.empty:
            continue
        rows.append({
            "strategy": strategy,
            "start_date": frame["date"].min(),
            "end_date": frame["date"].max(),
            "n_days": int(len(returns)),
            "ann_return": annual_return(returns),
            "ann_vol": annual_volatility(returns),
            "sharpe": sharpe(returns),
            "sortino": sortino(returns),
            "calmar": calmar(returns),
            "max_drawdown": max_drawdown(returns),
            "hit_ratio": hit_ratio(returns),
            "avg_assets": float(frame["n_assets"].mean()),
            "avg_turnover": float(frame["turnover"].mean()),
        })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def run_rule_based_backtest(
    panel,
    cost_bps=1.0,
    target_vol=0.15,
    start_date=None,
    end_date=None,
):
    """Run the transparent slow-momentum / fast-reversion backtest."""
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"])
    if start_date is not None:
        data = data.loc[data["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        data = data.loc[data["date"] <= pd.Timestamp(end_date)]

    data = add_rule_based_signals(data)
    position_cols = [
        "slow_momentum_position",
        "slow_fast_position",
        "cpd_adjusted_position",
    ]
    returns = [
        portfolio_returns(
            data,
            position_col=position_col,
            cost_bps=cost_bps,
            target_vol=target_vol,
        )
        for position_col in position_cols
    ]
    returns_by_strategy = pd.concat(returns, ignore_index=True)
    summary = performance_summary(returns_by_strategy)
    return returns_by_strategy, summary


def expanding_window_backtest(data, model_fn: Callable, folds: Iterable):
    """Run a generic expanding-window backtest for later model experiments."""
    results = []
    for fold_id, fold in enumerate(folds):
        train_start, train_end, test_start, test_end = fold
        train = data.loc[(data["date"] >= train_start) & (data["date"] <= train_end)]
        test = data.loc[(data["date"] >= test_start) & (data["date"] <= test_end)]
        model = model_fn()
        model.fit(train)
        fold_result = model.predict(test)
        fold_result["fold"] = fold_id
        results.append(fold_result)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
