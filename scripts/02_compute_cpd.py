"""Compute changepoint detection scores from the 01 processed dataset.

The base design is still one CPD run per time series. The improvement is that
the time series now come from the cleaner 01 pipeline:

* stock raw returns,
* stock returns relative to the equal-weight market,
* stock returns relative to SXXR when available,
* stock returns relative to their equal-weight sector,
* equal-weight sector return series for sector-level CPD.

No parameter search is done here. The goal is to avoid overfitting and produce
stable, comparable CPD features for the next modelling/backtest steps.
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

from src.cpd import (
    bocpd,
    cusum_continuous,
    detect_gp_cpd,
    detect_ma_cross,
    ensemble_cpd,
    jump_continuous,
    ttest_continuous,
)


ROBUST_SECTOR_SAMPLE_TICKERS = (
    "ASML NA",   # Information Technology
    "ALV GY",    # Financials
    "RNO FP",    # Consumer Discretionary
    "CA FP",     # Consumer Staples
    "NHY NO",    # Energy
    "SDR LN",    # Health Care
    "EN FP",     # Industrials
    "AI FP",     # Materials
    "CAST SS",   # Real Estate
    "DTE GY",    # Utilities
    "LI FP",     # Communication Services
)

DEFAULT_STOCK_SERIES = (
    "stock_raw",
    "stock_vs_ew",
    "stock_vs_sector",
)

STOCK_SERIES_COLUMNS = {
    "stock_raw": "1d_arith_ret_lag1",
    "stock_vs_ew": "1d_ret_vs_ew_lag1",
    "stock_vs_sxxr": "1d_ret_vs_sxxr_lag1",
    "stock_vs_exchange": "1d_ret_vs_exchange_lag1",
    "stock_vs_country": "1d_ret_vs_country_lag1",
    "stock_vs_region": "1d_ret_vs_region_lag1",
    "stock_vs_sector": "1d_ret_vs_sector_lag1",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute CPD scores on stock-level and sector-level return series.",
    )
    parser.add_argument("--in-dir", default="data/processed/stoxx600")
    parser.add_argument("--out-dir", default="data/processed/stoxx600")
    parser.add_argument("--stocks-file", default="stoxx600_processed.csv")
    parser.add_argument("--sectors-file", default="sector_returns.csv")
    parser.add_argument("--method", choices=["fast", "gp"], default="fast")
    parser.add_argument(
        "--stock-series",
        nargs="+",
        default=list(DEFAULT_STOCK_SERIES),
        choices=sorted(STOCK_SERIES_COLUMNS),
        help="Stock-level series to run CPD on.",
    )
    parser.add_argument(
        "--include-sector-level",
        action="store_true",
        default=True,
        help="Also run CPD on equal-weight sector return series.",
    )
    parser.add_argument(
        "--no-sector-level",
        dest="include_sector_level",
        action="store_false",
        help="Disable sector-level CPD.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=list(ROBUST_SECTOR_SAMPLE_TICKERS),
        help="Tickers to analyse. Defaults to a continuous sector-representative sample.",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional cap after applying --tickers. Use with --tickers omitted for quick broad scans.",
    )
    parser.add_argument(
        "--all-tickers",
        action="store_true",
        help="Use all available tickers instead of the default representative sample.",
    )
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--cooldown", type=int, default=30)
    parser.add_argument("--lbw", type=int, default=21)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--include-bocpd",
        action="store_true",
        help="Add BOCPD posterior score to the fast ensemble. Useful but slower.",
    )
    return parser


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_stock_panel(in_dir, stocks_file):
    """Load the processed panel produced by 01."""
    path = _resolve(in_dir) / stocks_file
    return pd.read_csv(path, parse_dates=["date"])


def load_sector_panel(in_dir, sectors_file):
    """Load equal-weight sector returns produced by 01."""
    path = _resolve(in_dir) / sectors_file
    if not path.exists():
        return pd.DataFrame()
    sectors = pd.read_csv(path, parse_dates=["date"])
    sectors = sectors.sort_values(["sector", "date"])
    sectors["sector_1d_ret_lag1"] = sectors.groupby("sector")["sector_1d_ret"].shift(1)
    return sectors


def select_tickers(stocks, requested_tickers, all_tickers, max_tickers):
    """Choose a stable ticker subset for CPD experiments."""
    available = list(dict.fromkeys(stocks["ticker"].dropna().astype(str)))
    if all_tickers:
        tickers = available
    else:
        requested = list(dict.fromkeys(requested_tickers or ROBUST_SECTOR_SAMPLE_TICKERS))
        tickers = [ticker for ticker in requested if ticker in set(available)]
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
    return tickers


def compute_fast_scores(values, threshold, cooldown, include_bocpd):
    """Run fast continuous CPD detectors on one return array."""
    cusum_score = cusum_continuous(values)
    jump_score = jump_continuous(values)
    ttest_score = ttest_continuous(values)
    ma_detections, ma_score = detect_ma_cross(values)

    score_list = [cusum_score, jump_score, ttest_score, ma_score]
    output = {
        "cusum_score": cusum_score,
        "jump_score": jump_score,
        "ttest_score": ttest_score,
        "ma_score": ma_score,
    }

    if include_bocpd:
        bocpd_score = bocpd(values)
        score_list.append(bocpd_score)
        output["bocpd_score"] = bocpd_score

    detections, ensemble_score = ensemble_cpd(
        score_list,
        threshold=threshold,
        cooldown=cooldown,
    )
    output["ensemble_score"] = ensemble_score
    output["is_changepoint"] = np.zeros(len(values), dtype=bool)
    output["is_changepoint"][detections] = True
    output["ma_cross"] = np.zeros(len(values), dtype=bool)
    output["ma_cross"][ma_detections] = True
    return output


def compute_gp_scores(values, lbw, stride, threshold, cooldown):
    """Run the slower paper-style GP score on one return array."""
    detections, severity, location = detect_gp_cpd(
        values,
        lbw=lbw,
        nu_threshold=threshold,
        gamma_min=0.5,
        cooldown=cooldown,
        stride=stride,
    )
    output = {
        "gp_severity": severity,
        "gp_location": location,
        "is_changepoint": np.zeros(len(values), dtype=bool),
    }
    output["is_changepoint"][detections] = True
    return output


def compute_one_series(
    dates,
    entity,
    scope,
    series_type,
    returns,
    method,
    threshold,
    cooldown,
    include_bocpd,
    lbw,
    stride,
    sector=None,
):
    """Compute CPD rows for one entity/series pair."""
    values = returns.to_numpy(dtype=float)
    if method == "fast":
        scores = compute_fast_scores(values, threshold, cooldown, include_bocpd)
    else:
        scores = compute_gp_scores(values, lbw, stride, threshold, cooldown)

    frame = pd.DataFrame({
        "date": dates,
        "scope": scope,
        "entity": entity,
        "ticker": entity if scope == "stock" else pd.NA,
        "sector": sector if sector is not None else (entity if scope == "sector" else pd.NA),
        "series_type": series_type,
        "method": method,
        "return": values,
    })
    for col, val in scores.items():
        frame[col] = val
    return frame


def build_stock_series(stocks, tickers, series_types):
    """Yield stock-level return series from the 01 processed panel."""
    missing_columns = [
        STOCK_SERIES_COLUMNS[series_type]
        for series_type in series_types
        if STOCK_SERIES_COLUMNS[series_type] not in stocks.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing columns from 01 output: {missing_columns}")

    for ticker in tickers:
        stock = stocks.loc[stocks["ticker"] == ticker].sort_values("date")
        if stock.empty:
            continue
        sector = stock["sector"].dropna().iloc[-1] if "sector" in stock and stock["sector"].notna().any() else None
        for series_type in series_types:
            col = STOCK_SERIES_COLUMNS[series_type]
            yield {
                "dates": stock["date"].to_numpy(),
                "entity": ticker,
                "scope": "stock",
                "series_type": series_type,
                "returns": stock[col],
                "sector": sector,
            }


def build_sector_series(sectors):
    """Yield sector-level equal-weight return series."""
    if sectors.empty:
        return
    if "sector_1d_ret_lag1" not in sectors.columns:
        raise KeyError("sector_returns.csv must contain sector_1d_ret or sector_1d_ret_lag1.")
    for sector, group in sectors.groupby("sector", sort=True):
        group = group.sort_values("date")
        yield {
            "dates": group["date"].to_numpy(),
            "entity": sector,
            "scope": "sector",
            "series_type": "sector_ew",
            "returns": group["sector_1d_ret_lag1"],
            "sector": sector,
        }


def summarise_cpd(cpd_panel):
    """Compact summary for comparing series/methods."""
    score_cols = [
        col for col in ["ensemble_score", "gp_severity", "bocpd_score", "cusum_score"]
        if col in cpd_panel.columns
    ]
    aggregations = {
        "is_changepoint": "sum",
        "return": "count",
    }
    for col in score_cols:
        aggregations[col] = "mean"

    summary = (
        cpd_panel
        .groupby(["scope", "entity", "sector", "series_type", "method"], dropna=False)
        .agg(aggregations)
        .rename(columns={"is_changepoint": "n_changepoints", "return": "n_observations"})
        .reset_index()
        .sort_values(["scope", "series_type", "n_changepoints"], ascending=[True, True, False])
    )
    return summary


def compute_cpd(args):
    in_dir = _resolve(args.in_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stocks = load_stock_panel(in_dir, args.stocks_file)
    sectors = load_sector_panel(in_dir, args.sectors_file)
    tickers = select_tickers(stocks, args.tickers, args.all_tickers, args.max_tickers)

    jobs = list(build_stock_series(stocks, tickers, args.stock_series))
    if args.include_sector_level:
        jobs.extend(list(build_sector_series(sectors)))

    rows = []
    for job in jobs:
        rows.append(
            compute_one_series(
                dates=job["dates"],
                entity=job["entity"],
                scope=job["scope"],
                series_type=job["series_type"],
                returns=job["returns"],
                method=args.method,
                threshold=args.threshold,
                cooldown=args.cooldown,
                include_bocpd=args.include_bocpd,
                lbw=args.lbw,
                stride=args.stride,
                sector=job["sector"],
            )
        )

    if not rows:
        raise ValueError("No CPD series were selected. Check tickers and input files.")

    cpd_panel = pd.concat(rows, ignore_index=True)
    suffix = args.method
    if args.method == "fast" and args.include_bocpd:
        suffix = "fast_bocpd"

    out_file = out_dir / f"cpd_scores_{suffix}.csv"
    summary_file = out_dir / f"cpd_summary_{suffix}.csv"
    cpd_panel.to_csv(out_file, index=False)
    summarise_cpd(cpd_panel).to_csv(summary_file, index=False)

    return cpd_panel, out_file, summary_file


def main() -> None:
    args = build_parser().parse_args()
    cpd_panel, out_file, summary_file = compute_cpd(args)

    print("CPD computation finished")
    print(f"Rows: {len(cpd_panel)}")
    print(f"Scopes: {sorted(cpd_panel['scope'].unique())}")
    print(f"Entities: {cpd_panel['entity'].nunique()}")
    print(f"Series types: {sorted(cpd_panel['series_type'].unique())}")
    print(f"Changepoints: {int(cpd_panel['is_changepoint'].sum())}")
    print(f"Scores: {out_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
