"""Build the STOXX 600 dataset used in parts 01 and 02.

The script mirrors the data definitions used in notebook 01, then adds the
pieces needed for the CPD discussion:

* raw stock returns,
* market-relative returns versus SXXR and equal-weight STOXX 600,
* group-relative returns for exchange/country/region,
* optional sector-relative returns when a ticker-sector mapping is available,
* lagged versions of usable returns to avoid lookahead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import numpy as np
import pandas as pd

from src.data_loader import (
    add_geography,
    append_price_atlas_tail,
    clean_prices,
    filter_prices_by_yearly_universe,
    load_price_atlas_prices,
    load_sector_mapping,
    load_stoxx600_prices,
    load_universe_by_year,
    prices_to_panel,
    universe_coverage,
)
from src.features import (
    add_relative_returns,
    equal_weight_return,
    macd,
    normalized_returns,
)
from src.preprocessing import log_returns, rolling_vol


DEFAULT_HORIZONS = (1, 5, 20, 21, 63, 126, 252)
DEFAULT_MACD_PAIRS = ((8, 24), (16, 48), (32, 96))
VOL_WINDOWS = (20, 60, 252)


def arithmetic_returns(prices, periods=1):
    """Compute returns without filling missing prices across data gaps."""
    return prices.pct_change(periods=periods, fill_method=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build clean STOXX 600 prices, returns and relative-return features.",
    )
    parser.add_argument("--raw-dir", default="data/raw/stoxx600")
    parser.add_argument("--out-dir", default="data/processed/stoxx600")
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--ffill-limit", type=int, default=5)
    parser.add_argument("--min-price", type=float, default=0.01)
    parser.add_argument(
        "--excel-path",
        default="data/raw/stoxx600/2025_2026_PRICE_ATLAS_data_sxxr_static.xlsx",
        help="PRICE ATLAS workbook used for the stock prices and SXXR benchmark.",
    )
    parser.add_argument(
        "--price-source",
        choices=["excel", "yearly"],
        default="yearly",
        help="Use annual CSV files from 2006-2024 plus the PRICE ATLAS Excel tail "
             "for 2025-2026 by default, with the tail rebased on the CSV overlap. "
             "Use 'excel' to match notebook 01 only.",
    )
    parser.add_argument(
        "--universe-json",
        default="data/universe/stoxx600/universe_by_year.json",
        help="Optional yearly universe JSON used for validation and optional filtering.",
    )
    parser.add_argument(
        "--sector-mapping",
        default="src/sector_mapping.py",
        help="Optional CSV/XLSX/PY with ticker and sector/supersector columns.",
    )
    parser.add_argument(
        "--filter-universe",
        action="store_true",
        help="Mask prices outside the yearly universe. Off by default to avoid dropping unmatched 2025-2026 tickers.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=252,
        help="Drop tickers with fewer valid prices than this.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional limit for quick classroom/demo runs.",
    )
    return parser


def _resolve(path):
    """Resolve a project-relative or absolute path."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _melt_wide(wide, value_name):
    """Convert a wide date-indexed matrix to tidy date/ticker/value format."""
    return (
        wide.reset_index()
        .melt(id_vars="date", var_name="ticker", value_name=value_name)
        .dropna(subset=[value_name])
    )


def load_price_history(raw_dir, excel_path, start_year, end_year, price_source="excel"):
    """Load the price history from the notebook source or the yearly CSV archive."""
    if price_source == "excel":
        if not excel_path.exists():
            raise FileNotFoundError(f"PRICE ATLAS workbook not found: {excel_path}")
        return load_price_atlas_prices(
            excel_path,
            start_year=start_year,
            end_year=end_year,
        )

    raw_prices = load_stoxx600_prices(
        raw_dir,
        start_year=start_year,
        end_year=min(end_year, 2024),
    )

    if excel_path.exists() and end_year >= 2025:
        atlas_start_year = min(2024, max(start_year, 2013))
        atlas_prices = load_price_atlas_prices(
            excel_path,
            start_year=atlas_start_year,
            end_year=end_year,
        )
        return append_price_atlas_tail(raw_prices, atlas_prices)

    return raw_prices


def build_benchmarks(prices, excel_path):
    """Build SXXR and equal-weight benchmark returns."""
    arith_returns = arithmetic_returns(prices)
    ew_return = equal_weight_return(arith_returns)
    ew_index = 100.0 * (1.0 + ew_return.fillna(0.0)).cumprod()
    if not ew_index.empty:
        ew_index.iloc[0] = 100.0

    market_returns = pd.DataFrame({"ew_1d_ret": ew_return})
    benchmark_frames = [
        pd.DataFrame({
            "date": ew_index.index,
            "benchmark": "EW",
            "price": ew_index.values,
            "1d_arith_ret": ew_return.values,
        })
    ]

    if excel_path.exists():
        sxxr = load_price_atlas_prices(excel_path, sheet_name="benchmark")
        if "price" not in sxxr.columns:
            raise ValueError("Benchmark sheet must contain a 'price' column.")
        sxxr_price = sxxr["price"].rename("SXXR")
        sxxr_ret = arithmetic_returns(sxxr_price).rename("sxxr_1d_ret")
        market_returns = market_returns.join(sxxr_ret, how="left")
        benchmark_frames.append(
            pd.DataFrame({
                "date": sxxr_price.index,
                "benchmark": "SXXR",
                "price": sxxr_price.values,
                "1d_arith_ret": sxxr_ret.values,
            })
        )

    benchmarks = (
        pd.concat(benchmark_frames, ignore_index=True)
        .sort_values(["date", "benchmark"])
        .reset_index(drop=True)
    )
    market_returns.index.name = "date"
    return market_returns, benchmarks


def build_feature_panel(prices, market_returns, sector_mapping=None):
    """Create the tidy panel consumed by the notebooks and CPD scripts."""
    log_ret_1d = log_returns(prices, periods=1)
    arith_ret_1d = arithmetic_returns(prices, periods=1)

    panel = prices_to_panel(prices)

    for horizon in DEFAULT_HORIZONS:
        panel = panel.merge(
            _melt_wide(log_returns(prices, periods=horizon), f"{horizon}d_log_ret"),
            on=["date", "ticker"],
            how="left",
        )
        panel = panel.merge(
            _melt_wide(arithmetic_returns(prices, periods=horizon), f"{horizon}d_arith_ret"),
            on=["date", "ticker"],
            how="left",
        )

    for window in VOL_WINDOWS:
        vol = rolling_vol(log_ret_1d, window=window, annualise=True)
        panel = panel.merge(
            _melt_wide(vol, f"{window}d_vol"),
            on=["date", "ticker"],
            how="left",
        )

    panel = add_geography(panel)
    if sector_mapping is not None:
        panel = panel.merge(sector_mapping, on="ticker", how="left")

    panel = add_relative_returns(
        panel,
        market_returns=market_returns,
        stock_return_col="1d_arith_ret",
        group_cols=("exchange", "country", "region", "sector"),
        lag=True,
    )

    return panel


def build_model_features(prices, max_tickers=None):
    """Keep the existing model-style normalized-return/MACD output."""
    feature_prices = prices.iloc[:, :max_tickers] if max_tickers is not None else prices

    panels = []
    for ticker in feature_prices.columns:
        price = feature_prices[ticker].dropna()
        if price.empty:
            continue
        returns = log_returns(price)

        frame = pd.DataFrame({
            "date": price.index,
            "ticker": ticker,
            "price": price.values,
        })
        log_price = np.log(price)
        for horizon in DEFAULT_HORIZONS:
            frame[f"logret_{horizon}d"] = log_price.diff(horizon).values

        vol_daily = rolling_vol(returns, window=63, annualise=False)
        frame["vol_daily"] = vol_daily.values
        frame["vol_annual"] = (vol_daily * np.sqrt(252)).values

        norm = normalized_returns(returns, horizons=DEFAULT_HORIZONS)
        macd_features = macd(price, pairs=DEFAULT_MACD_PAIRS)
        features = pd.concat([norm, macd_features], axis=1).reset_index(drop=True)
        panels.append(pd.concat([frame.reset_index(drop=True), features], axis=1))

    if not panels:
        return pd.DataFrame()
    return pd.concat(panels, ignore_index=True)


def build_sector_returns(feature_panel):
    """Build equal-weight sector return indices for sector-level CPD."""
    required = {"date", "sector", "sector_1d_ret"}
    if not required.issubset(feature_panel.columns):
        return pd.DataFrame()

    sector_returns = (
        feature_panel[["date", "sector", "sector_1d_ret"]]
        .dropna(subset=["sector", "sector_1d_ret"])
        .drop_duplicates(subset=["date", "sector"])
        .sort_values(["sector", "date"])
        .reset_index(drop=True)
    )
    sector_returns["sector_ew_index"] = (
        sector_returns
        .groupby("sector")["sector_1d_ret"]
        .transform(lambda s: 100.0 * (1.0 + s.fillna(0.0)).cumprod())
    )
    return sector_returns.sort_values(["date", "sector"]).reset_index(drop=True)


def build_dataset(
    raw_dir,
    out_dir,
    start_year,
    end_year,
    ffill_limit,
    min_price,
    excel_path,
    price_source,
    universe_json,
    sector_mapping,
    filter_universe,
    min_observations,
    max_tickers,
):
    raw_dir = _resolve(raw_dir)
    out_dir = _resolve(out_dir)
    excel_path = _resolve(excel_path)
    universe_path = _resolve(universe_json)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_price_history(raw_dir, excel_path, start_year, end_year, price_source)
    prices = clean_prices(prices, min_price=min_price, ffill_limit=ffill_limit)

    coverage = pd.DataFrame()
    if universe_path.exists():
        universe = load_universe_by_year(universe_path)
        coverage = universe_coverage(prices, universe)
        coverage.to_csv(out_dir / "universe_coverage.csv", index=False)
        if filter_universe:
            prices = filter_prices_by_yearly_universe(prices, universe)

    prices = prices.dropna(axis=1, thresh=min_observations)
    if max_tickers is not None:
        prices = prices.iloc[:, :max_tickers]

    sector_path = _resolve(sector_mapping) if sector_mapping else None
    sector_df = load_sector_mapping(sector_path) if sector_path and sector_path.exists() else None
    market_returns, benchmarks = build_benchmarks(prices, excel_path)
    feature_panel = build_feature_panel(prices, market_returns, sector_df)
    model_features = build_model_features(prices, max_tickers=None)
    sector_returns = build_sector_returns(feature_panel)

    if sector_df is not None:
        sector_coverage = (
            pd.DataFrame({"ticker": prices.columns})
            .merge(sector_df, on="ticker", how="left")
            .assign(has_sector=lambda x: x["sector"].notna())
        )
        sector_coverage.to_csv(out_dir / "sector_coverage.csv", index=False)

    log_ret_1d = log_returns(prices)
    arith_ret_1d = arithmetic_returns(prices)

    prices.to_csv(out_dir / "prices_clean.csv")
    log_ret_1d.to_csv(out_dir / "returns_1d.csv")
    arith_ret_1d.to_csv(out_dir / "returns_arith_1d.csv")
    prices_to_panel(prices).to_csv(out_dir / "prices_panel.csv", index=False)
    feature_panel.to_csv(out_dir / "stoxx600_processed.csv", index=False)
    feature_panel.to_csv(out_dir / "features_panel.csv", index=False)
    model_features.to_csv(out_dir / "model_features_panel.csv", index=False)
    benchmarks.to_csv(out_dir / "benchmark_stoxx600_ew.csv", index=False)
    market_returns.reset_index().to_csv(out_dir / "benchmark_returns.csv", index=False)
    if not sector_returns.empty:
        sector_returns.to_csv(out_dir / "sector_returns.csv", index=False)

    relative_cols = [col for col in feature_panel.columns if "ret_vs" in col]
    sector_available = "sector" in feature_panel.columns and feature_panel["sector"].notna().any()
    return {
        "n_dates": len(prices),
        "n_tickers": len(prices.columns),
        "n_feature_rows": len(feature_panel),
        "relative_cols": relative_cols,
        "sector_available": bool(sector_available),
        "coverage_rows": len(coverage),
        "out_dir": out_dir,
    }


def main() -> None:
    args = build_parser().parse_args()
    summary = build_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        ffill_limit=args.ffill_limit,
        min_price=args.min_price,
        excel_path=args.excel_path,
        price_source=args.price_source,
        universe_json=args.universe_json,
        sector_mapping=args.sector_mapping,
        filter_universe=args.filter_universe,
        min_observations=args.min_observations,
        max_tickers=args.max_tickers,
    )

    print("Dataset built successfully")
    print(f"Dates: {summary['n_dates']}")
    print(f"Tickers: {summary['n_tickers']}")
    print(f"Feature rows: {summary['n_feature_rows']}")
    print(f"Relative columns: {', '.join(summary['relative_cols'])}")
    print(f"Sector mapping active: {summary['sector_available']}")
    print(f"Universe coverage rows: {summary['coverage_rows']}")
    print(f"Output folder: {summary['out_dir']}")


if __name__ == "__main__":
    main()
