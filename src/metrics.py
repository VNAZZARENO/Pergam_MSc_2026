"""Risk-adjusted performance metrics for strategy evaluation.

Two compute_metrics variants are provided:
    compute_metrics()         — programmatic dict, geometric compounding.
                                Used by scripts and NB03.
    compute_display_metrics() — display dict with pretty keys, arithmetic.
                                Used by NB04 to build the performance table.
"""

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


# ---------------------------------------------------------------------------
# Display metrics  (NB04 — pretty keys, arithmetic annualisation)
# ---------------------------------------------------------------------------


def compute_display_metrics(returns: pd.Series) -> dict:
    """Performance metrics with display-friendly keys for NB04 tables.

    Uses simple arithmetic annualisation (mean * 252) consistent with the
    paper's Exhibit 3 presentation.  For programmatic use, prefer
    ``compute_metrics()`` which uses geometric compounding.

    Parameters
    ----------
    returns : daily return series.

    Returns
    -------
    dict with keys: Returns, Vol, Sharpe, Downside Dev, Sortino,
                    MDD, Calmar, % +ve, Avg P / Avg L.
    """
    _nan_keys = ["Returns", "Vol", "Sharpe", "Downside Dev", "Sortino",
                 "MDD", "Calmar", "% +ve", "Avg P / Avg L"]
    r = returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in _nan_keys}

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe_ = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0]
    dd_dev   = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    sortino_ = ann_ret / dd_dev if dd_dev and dd_dev > 0 else np.nan

    cum    = (1 + r).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax()
    mdd    = dd.min()
    calmar_= ann_ret / abs(mdd) if mdd != 0 else np.nan

    pct_pos = float((r > 0).mean())
    avg_p   = r[r > 0].mean() if (r > 0).any() else np.nan
    avg_l   = abs(r[r < 0].mean()) if (r < 0).any() else np.nan
    p_to_l  = avg_p / avg_l if avg_l and avg_l > 0 else np.nan

    return {
        "Returns":      ann_ret,
        "Vol":          ann_vol,
        "Sharpe":       sharpe_,
        "Downside Dev": dd_dev,
        "Sortino":      sortino_,
        "MDD":          mdd,
        "Calmar":       calmar_,
        "% +ve":        pct_pos,
        "Avg P / Avg L": p_to_l,
    }


def format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Pretty-print formatting for the NB04 metrics table.

    Parameters
    ----------
    df : metrics DataFrame with display-friendly column names.

    Returns
    -------
    DataFrame with values formatted as strings.
    """
    fmt = df.copy()
    for col in ["Returns", "Vol", "Downside Dev", "MDD"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    for col in ["Sharpe", "Sortino", "Calmar", "Avg P / Avg L"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
    if "% +ve" in fmt.columns:
        fmt["% +ve"] = fmt["% +ve"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    return fmt


# ---------------------------------------------------------------------------
# Rolling and yearly Sharpe  (NB04 time-series diagnostics)
# ---------------------------------------------------------------------------


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Trailing ``window``-day annualised Sharpe ratio.

    Parameters
    ----------
    returns : daily return series.
    window  : rolling window in trading days (default 252 = 1 year).

    Returns
    -------
    pd.Series of rolling Sharpe values.
    """
    r = returns.dropna()
    return (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252)


def yearly_sharpe(returns: pd.Series) -> pd.Series:
    """Annualised Sharpe ratio broken down by calendar year.

    Parameters
    ----------
    returns : daily return series with a DatetimeIndex.

    Returns
    -------
    pd.Series indexed by year (int).
    """
    r = returns.dropna()
    return r.groupby(r.index.year).apply(
        lambda y: float((y.mean() / y.std()) * np.sqrt(252)) if y.std() > 0 else 0.0
    )
