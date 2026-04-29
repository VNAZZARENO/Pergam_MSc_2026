"""Feature engineering: normalized multi-horizon returns and MACD."""

from __future__ import annotations

import sys
import numpy as np
from typing import Iterable, Sequence, Tuple

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd

# Organisation du fichier :
# creation des variables explicatives : momentum, MACD et volatilite.


# [FEATURE] normalized_returns
# Mesure le momentum multi-horizon en le normalisant par le risque local.
def normalized_returns(returns, horizons=(1, 5, 21, 63, 252)):
    """Compute volatility-normalized returns over the given horizons.

    Parameters
    ----------
    returns : pandas.Series
        Daily log or arithmetic returns for one stock.
    horizons : iterable of int
        Lookback horizons in trading days.

    Returns
    -------
    pandas.DataFrame
        One column per horizon, named 'norm_ret_{h}d'.
    """
    out = {}
    for h in horizons:
        cumret = returns.rolling(h).sum()
        vol_window = max(h, 20)
        min_periods = max(vol_window // 2, 2)
        vol = returns.rolling(vol_window, min_periods=min_periods).std() * np.sqrt(h)
        vol = vol.replace(0, np.nan)
        out[f"norm_ret_{h}d"] = cumret / vol
    return pd.DataFrame(out, index=returns.index)


# [FEATURE] macd
# Compare moyennes mobiles rapides/lentes pour capter la tendance.
def macd(prices, pairs=((8, 24), (16, 48), (32, 96))):
    """Compute MACD indicators for the given (short, long) EMA pairs.

    Parameters
    ----------
    prices : pandas.Series
        Price series for one stock.
    pairs : sequence of (int, int)
        Each tuple = (short_span, long_span) for EMA.

    Returns
    -------
    pandas.DataFrame
        One column per pair, named 'macd_{short}_{long}'.
    """
    out = {}
    for short, long in pairs:
        ema_s = prices.ewm(span=short, min_periods=short).mean()
        ema_l = prices.ewm(span=long, min_periods=long).mean()
        raw = ema_s - ema_l
        vol = prices.rolling(long, min_periods=long).std()
        vol = vol.replace(0, np.nan)
        out[f"macd_{short}_{long}"] = raw / vol
    return pd.DataFrame(out, index=prices.index)


# [FEATURE] realised_volatility
# Calcule la volatilite annualisee sur plusieurs horizons.
def realised_volatility(returns, windows=(20, 60, 252)):
    """Annualised rolling volatility at multiple horizons.

    Returns
    -------
    pandas.DataFrame
        One column per window, named 'vol_{w}d'.
    """
    out = {}
    for w in windows:
        out[f"vol_{w}d"] = returns.rolling(w, min_periods=w).std() * np.sqrt(252)
    return pd.DataFrame(out, index=returns.index)


# [FEATURE] equal_weight_return
# Calcule un return equal-weight robuste en ignorant les valeurs manquantes.
def equal_weight_return(returns):
    """Daily equal-weight return from a wide return matrix."""
    return returns.mean(axis=1, skipna=True).rename("ew_1d_ret")


# [FEATURE] add_relative_returns
# Retire les returns de marche/groupe pour isoler la composante idiosyncratique.
def add_relative_returns(
    panel,
    market_returns=None,
    stock_return_col="1d_arith_ret",
    group_cols=("exchange", "country", "region", "sector"),
    lag=True,
    max_lag_gap_days=10,
):
    """Add market- and group-relative returns to a stock panel.

    Parameters
    ----------
    panel : pandas.DataFrame
        Tidy stock panel with date, ticker and daily stock returns.
    market_returns : pandas.DataFrame or None
        Optional date-indexed market returns. Common columns are
        ``sxxr_1d_ret`` and ``ew_1d_ret``.
    stock_return_col : str
        Column containing daily arithmetic stock returns.
    group_cols : iterable of str
        Metadata columns used to build equal-weight group returns.
    lag : bool
        If True, add lagged versions usable at t+1 to avoid lookahead.
    max_lag_gap_days : int
        Maximum calendar gap allowed when shifting returns. Larger gaps reset
        the lagged value to NaN so stale returns are not carried across missing
        data blocks.
    """
    out = panel.copy()
    out = out.sort_values(["ticker", "date"])

    if market_returns is not None:
        market = market_returns.copy()
        if not isinstance(market.index, pd.DatetimeIndex):
            if "date" not in market.columns:
                raise ValueError("market_returns must be date-indexed or contain a date column.")
            market = market.set_index("date")
        market.index = pd.to_datetime(market.index)
        out = out.merge(market, left_on="date", right_index=True, how="left")

    if "sxxr_1d_ret" in out.columns:
        out["1d_ret_vs_sxxr"] = out[stock_return_col] - out["sxxr_1d_ret"]
    if "ew_1d_ret" in out.columns:
        out["1d_ret_vs_ew"] = out[stock_return_col] - out["ew_1d_ret"]

    relative_cols = [
        col for col in ["1d_ret_vs_sxxr", "1d_ret_vs_ew"] if col in out.columns
    ]

    for group_col in group_cols:
        if group_col not in out.columns:
            continue
        group_return_col = f"{group_col}_1d_ret"
        relative_col = f"1d_ret_vs_{group_col}"

        group_returns = (
            out.dropna(subset=[group_col])
            .groupby(["date", group_col], dropna=True)[stock_return_col]
            .mean()
            .rename(group_return_col)
            .reset_index()
        )
        out = out.merge(group_returns, on=["date", group_col], how="left")
        out[relative_col] = out[stock_return_col] - out[group_return_col]
        relative_cols.append(relative_col)

    if lag:
        lag_cols = [stock_return_col] + relative_cols
        date_gap = out.groupby("ticker", sort=False)["date"].diff().dt.days
        for col in lag_cols:
            out[f"{col}_lag1"] = out.groupby("ticker", sort=False)[col].shift(1)
            out.loc[date_gap > max_lag_gap_days, f"{col}_lag1"] = np.nan

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)
