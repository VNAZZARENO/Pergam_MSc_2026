"""Risk-adjusted performance metrics for strategy evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe(returns: pd.Series | np.ndarray, annualization: int = 252) -> float:
    """Annualized Sharpe ratio (assumes zero risk-free rate)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return r.mean() / r.std(ddof=1) * np.sqrt(annualization)


def sortino(returns: pd.Series | np.ndarray, annualization: int = 252) -> float:
    """Annualized Sortino ratio (downside deviation denominator)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    downside = r[r < 0]
    if len(downside) < 2:
        return np.nan
    dd = downside.std(ddof=1)
    if dd == 0:
        return np.nan
    return r.mean() / dd * np.sqrt(annualization)


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    r = np.asarray(returns, dtype=float)
    cumulative = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / peak
    return float(dd.min())


def calmar(returns: pd.Series | np.ndarray, annualization: int = 252) -> float:
    """Calmar ratio: annualized CAGR divided by absolute max drawdown."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return np.nan
    cagr = (np.prod(1.0 + r) ** (annualization / n)) - 1.0
    mdd  = abs(max_drawdown(r))
    return cagr / mdd if mdd > 0 else np.nan


def hit_ratio(returns: pd.Series | np.ndarray) -> float:
    """Fraction of positive daily returns."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    return float((r > 0).mean()) if len(r) > 0 else np.nan


def compute_metrics(
    returns: pd.Series | np.ndarray,
    annualization: int = 252,
) -> dict[str, float]:
    """Compute the full set of metrics for a daily return series.

    Returns
    -------
    dict with keys: sharpe, sortino, calmar, ann_return, ann_vol,
                    max_drawdown, hit_ratio.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)

    ann_ret = (np.prod(1.0 + r) ** (annualization / n)) - 1.0 if n > 0 else np.nan
    ann_vol = r.std(ddof=1) * np.sqrt(annualization) if n > 1 else np.nan

    return {
        "sharpe":       sharpe(r, annualization),
        "sortino":      sortino(r, annualization),
        "calmar":       calmar(r, annualization),
        "ann_return":   ann_ret,
        "ann_vol":      ann_vol,
        "max_drawdown": max_drawdown(r),
        "hit_ratio":    hit_ratio(r),
    }
