"""Load STOXX 600 price data from the PRICE ATLAS Excel file.

Reads the ``price`` and ``benchmark`` sheets, returns wide/long DataFrames,
attaches country/region from the exchange suffix, and builds an equal-weight
benchmark alongside the cap-weighted SXXR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXCHANGE_TO_COUNTRY = {
    "LN": "United Kingdom",
    "FP": "France",
    "GY": "Germany",
    "NA": "Netherlands",
    "IM": "Italy",
    "SQ": "Spain",
    "SE": "Switzerland",
    "SS": "Sweden",
    "DC": "Denmark",
    "FH": "Finland",
    "NO": "Norway",
    "BB": "Belgium",
    "AV": "Austria",
    "PL": "Portugal",
    "ID": "Ireland",
    "PW": "Poland",
}

COUNTRY_TO_REGION = {
    "United Kingdom": "UK & Ireland",
    "Ireland": "UK & Ireland",
    "France": "Western Europe",
    "Germany": "Western Europe",
    "Netherlands": "Western Europe",
    "Belgium": "Western Europe",
    "Austria": "Western Europe",
    "Switzerland": "Western Europe",
    "Italy": "Southern Europe",
    "Spain": "Southern Europe",
    "Portugal": "Southern Europe",
    "Sweden": "Nordic",
    "Denmark": "Nordic",
    "Finland": "Nordic",
    "Norway": "Nordic",
    "Poland": "Eastern Europe",
}


def load_prices_wide(path: str | Path) -> pd.DataFrame:
    """Load the ``price`` sheet as a wide DataFrame indexed by date."""
    df = pd.read_excel(path, sheet_name="price")
    df = df.rename(columns={"Ticker": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_sxxr(path: str | Path) -> pd.Series:
    """Load the ``benchmark`` sheet and return the SXXR price series."""
    df = pd.read_excel(path, sheet_name="benchmark")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    s = df["price"].copy()
    s.name = "SXXR"
    return s


def wide_to_long(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """Melt the wide price matrix to long format: (date, ticker, price)."""
    df = (
        prices_wide
        .stack(future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "ticker", 0: "price"})
    )
    return (
        df.dropna(subset=["price"])
          .sort_values(["ticker", "date"])
          .reset_index(drop=True)
    )


def add_country_region(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``exchange``, ``country``, ``region`` columns from the ticker."""
    df = df.copy()
    df["exchange"] = df["ticker"].str.split().str[-1]
    df["country"] = df["exchange"].map(EXCHANGE_TO_COUNTRY)
    df["region"] = df["country"].map(COUNTRY_TO_REGION)
    return df


def build_equal_weight_benchmark(prices_wide: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional mean of arithmetic returns, cumulated from 100."""
    ew_return = prices_wide.pct_change().mean(axis=1)
    ew_index = 100.0 * (1.0 + ew_return).cumprod()
    ew_index.iloc[0] = 100.0
    ew_index.name = "EW"
    return ew_index


def build_benchmark_df(sxxr: pd.Series, ew: pd.Series) -> pd.DataFrame:
    """Stack SXXR and EW into a long (date, benchmark, price) DataFrame."""
    sxxr_df = pd.DataFrame({"date": sxxr.index, "benchmark": "SXXR", "price": sxxr.values})
    ew_df = pd.DataFrame({"date": ew.index, "benchmark": "EW", "price": ew.values})
    return (
        pd.concat([sxxr_df, ew_df], ignore_index=True)
          .sort_values(["date", "benchmark"])
          .reset_index(drop=True)
    )


def load_stoxx600(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end loader.

    Returns
    -------
    prices_wide : (dates x tickers) price matrix.
    long_df     : long (date, ticker, price, exchange, country, region).
    benchmarks  : long (date, benchmark, price) with SXXR and EW.
    """
    prices_wide = load_prices_wide(path)
    sxxr = load_sxxr(path)

    long_df = add_country_region(wide_to_long(prices_wide))
    ew = build_equal_weight_benchmark(prices_wide)
    benchmarks = build_benchmark_df(sxxr, ew)

    return prices_wide, long_df, benchmarks
