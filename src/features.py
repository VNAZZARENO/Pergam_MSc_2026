"""Feature engineering for STOXX 600 panel data.

Computes per-ticker log and arithmetic returns over multiple horizons and
annualized realized volatility over rolling windows. Operates on the long
(date, ticker, price, ...) DataFrame produced by :mod:`data_loader`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


RETURN_HORIZONS: tuple[tuple[int, str], ...] = (
    (1, "1d"),
    (20, "20d"),
    (252, "252d"),
)

VOL_WINDOWS: tuple[tuple[int, str], ...] = (
    (20, "20d"),
    (60, "60d"),
    (252, "252d"),
)

FINAL_COLS: tuple[str, ...] = (
    "date", "ticker", "price",
    "1d_log_ret", "20d_log_ret", "252d_log_ret",
    "1d_arith_ret", "20d_arith_ret", "252d_arith_ret",
    "20d_vol", "60d_vol", "252d_vol",
    "exchange", "country", "region",
)


def add_log_returns(
    df: pd.DataFrame,
    horizons: Iterable[tuple[int, str]] = RETURN_HORIZONS,
) -> pd.DataFrame:
    """Add ``{label}_log_ret`` columns computed per ticker."""
    df = df.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker")["price"]
    for d, label in horizons:
        df[f"{label}_log_ret"] = g.transform(lambda s, d=d: np.log(s).diff(d))
    return df


def add_arith_returns(
    df: pd.DataFrame,
    horizons: Iterable[tuple[int, str]] = RETURN_HORIZONS,
) -> pd.DataFrame:
    """Add ``{label}_arith_ret`` columns computed per ticker."""
    df = df.sort_values(["ticker", "date"]).copy()
    g = df.groupby("ticker")["price"]
    for d, label in horizons:
        df[f"{label}_arith_ret"] = g.transform(lambda s, d=d: s.pct_change(d))
    return df


def add_realized_vol(
    df: pd.DataFrame,
    windows: Iterable[tuple[int, str]] = VOL_WINDOWS,
    annualization: int = 252,
) -> pd.DataFrame:
    """Add annualized realized vol ``{label}_vol`` from 1d log returns."""
    df = df.sort_values(["ticker", "date"]).copy()
    log_ret_1d = df.groupby("ticker")["price"].transform(lambda s: np.log(s).diff(1))
    scale = np.sqrt(annualization)
    for w, label in windows:
        df[f"{label}_vol"] = (
            log_ret_1d
            .groupby(df["ticker"])
            .transform(lambda s, w=w: s.rolling(w, min_periods=w).std() * scale)
        )
    return df


def add_features(
    df: pd.DataFrame,
    return_horizons: Iterable[tuple[int, str]] = RETURN_HORIZONS,
    vol_windows: Iterable[tuple[int, str]] = VOL_WINDOWS,
) -> pd.DataFrame:
    """Add log returns, arithmetic returns, and realized volatility."""
    df = add_log_returns(df, return_horizons)
    df = add_arith_returns(df, return_horizons)
    df = add_realized_vol(df, vol_windows)
    return df


def finalize(df: pd.DataFrame, cols: Iterable[str] = FINAL_COLS) -> pd.DataFrame:
    """Reorder columns and sort by (date, ticker)."""
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(["date", "ticker"]).reset_index(drop=True)
