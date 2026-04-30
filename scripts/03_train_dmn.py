"""Train the first working DMN-lite model and export positions.

This script intentionally uses a small dependency-free ridge model instead of
the final LSTM. It keeps the economic structure of the paper: momentum,
volatility and changepoint features are mapped to positions, then evaluated in
the backtest script.
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

from src.model import RidgePositionModel, sharpe_loss


BASE_FEATURE_COLS = [
    "21d_arith_ret",
    "63d_arith_ret",
    "126d_arith_ret",
    "252d_arith_ret",
    "20d_vol",
    "60d_vol",
    "252d_vol",
    "1d_ret_vs_ew_lag1",
    "1d_ret_vs_sector_lag1",
]
CPD_FEATURE_COL = "ensemble_score"
RETURN_COL = "1d_arith_ret"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a lightweight walk-forward DMN approximation.",
    )
    parser.add_argument("--data-dir", default="data/processed/stoxx600")
    parser.add_argument("--panel-file", default="stoxx600_processed.csv")
    parser.add_argument("--cpd-file", default="cpd_scores_fast.csv")
    parser.add_argument("--out-positions", default="dmn_lite_positions.csv")
    parser.add_argument("--out-folds", default="dmn_lite_fold_summary.csv")
    parser.add_argument("--first-test-year", type=int, default=2010)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=25.0)
    parser.add_argument("--position-scale", type=float, default=0.10)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument(
        "--no-cpd",
        action="store_true",
        help="Train without the CPD feature, useful for ablation checks.",
    )
    return parser


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_panel(data_dir, panel_file, max_tickers=None):
    """Load the feature panel used by the DMN-lite model."""
    usecols = ["date", "ticker", RETURN_COL] + BASE_FEATURE_COLS
    panel = pd.read_csv(data_dir / panel_file, usecols=usecols, parse_dates=["date"])
    panel = panel.dropna(subset=["ticker"]).sort_values(["ticker", "date"])
    if max_tickers is not None:
        tickers = list(dict.fromkeys(panel["ticker"].astype(str)))[:max_tickers]
        panel = panel.loc[panel["ticker"].isin(tickers)]
    return panel


def load_cpd_scores(data_dir, cpd_file):
    """Load stock-level sector-relative CPD scores if available."""
    path = data_dir / cpd_file
    if not path.exists():
        return pd.DataFrame(columns=["date", "ticker", CPD_FEATURE_COL])

    cpd = pd.read_csv(
        path,
        usecols=["date", "scope", "ticker", "series_type", CPD_FEATURE_COL],
        parse_dates=["date"],
        dtype={"scope": "string", "ticker": "string", "series_type": "string"},
        low_memory=False,
    )
    cpd = cpd.loc[
        (cpd["scope"] == "stock")
        & (cpd["series_type"] == "stock_vs_sector")
        & cpd["ticker"].notna()
    ]
    return cpd[["date", "ticker", CPD_FEATURE_COL]].drop_duplicates(
        subset=["date", "ticker"],
        keep="last",
    )


def make_supervised_frame(panel):
    """Create one-step-ahead targets and the dates positions apply to."""
    out = panel.sort_values(["ticker", "date"]).copy()
    grouped = out.groupby("ticker", sort=False)
    out["position_date"] = grouped["date"].shift(-1)
    out["next_return"] = grouped[RETURN_COL].shift(-1)

    daily_vol = pd.to_numeric(out["60d_vol"], errors="coerce") / np.sqrt(252.0)
    out["target_next_vol_scaled_return"] = out["next_return"] / daily_vol.replace(0.0, np.nan)
    out["target_next_vol_scaled_return"] = out["target_next_vol_scaled_return"].clip(-5.0, 5.0)
    return out.dropna(subset=["position_date"])


def train_walk_forward(frame, feature_cols, first_test_year, last_test_year, alpha, position_scale):
    """Train on expanding history and predict one calendar year at a time."""
    positions = []
    fold_rows = []

    for year in range(first_test_year, last_test_year + 1):
        test_start = pd.Timestamp(year=year, month=1, day=1)
        test_end = pd.Timestamp(year=year, month=12, day=31)

        train = frame.loc[frame["date"] < test_start]
        test = frame.loc[
            (frame["position_date"] >= test_start)
            & (frame["position_date"] <= test_end)
        ]
        train = train.dropna(subset=["target_next_vol_scaled_return"])
        if train.empty or test.empty:
            continue

        model = RidgePositionModel(
            feature_cols=feature_cols,
            alpha=alpha,
            position_scale=position_scale,
        ).fit(train, target_col="target_next_vol_scaled_return")

        fold_positions = test[["position_date", "ticker", "next_return"]].copy()
        fold_positions["dmn_lite_position"] = model.predict_position(test)
        fold_positions["model_score"] = model.predict_score(test)
        fold_positions["fold_year"] = year
        fold_positions = fold_positions.rename(
            columns={"position_date": "date", "next_return": "realized_return"}
        )
        positions.append(fold_positions)

        fold_rows.append({
            "fold_year": year,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_start": train["date"].min(),
            "train_end": train["date"].max(),
            "test_start": fold_positions["date"].min(),
            "test_end": fold_positions["date"].max(),
            "train_objective_sharpe_loss": sharpe_loss(
                model.predict_position(train),
                train["next_return"].to_numpy(dtype=float),
            ),
            "avg_abs_position": float(np.abs(fold_positions["dmn_lite_position"]).mean()),
        })

    if not positions:
        raise ValueError("No walk-forward folds were produced.")
    return pd.concat(positions, ignore_index=True), pd.DataFrame(fold_rows)


def main() -> None:
    args = build_parser().parse_args()
    data_dir = _resolve(args.data_dir)

    panel = load_panel(data_dir, args.panel_file, max_tickers=args.max_tickers)
    feature_cols = list(BASE_FEATURE_COLS)
    if not args.no_cpd:
        cpd = load_cpd_scores(data_dir, args.cpd_file)
        panel = panel.merge(cpd, on=["date", "ticker"], how="left")
        panel[CPD_FEATURE_COL] = panel[CPD_FEATURE_COL].fillna(0.0)
        feature_cols.append(CPD_FEATURE_COL)

    supervised = make_supervised_frame(panel)
    positions, folds = train_walk_forward(
        supervised,
        feature_cols=feature_cols,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year,
        alpha=args.alpha,
        position_scale=args.position_scale,
    )

    positions_path = data_dir / args.out_positions
    folds_path = data_dir / args.out_folds
    positions.to_csv(positions_path, index=False)
    folds.to_csv(folds_path, index=False)

    print("DMN-lite training finished")
    print(f"Rows loaded: {len(panel):,}")
    print(f"Tickers: {panel['ticker'].nunique():,}")
    print(f"Features: {', '.join(feature_cols)}")
    print(f"Position date range: {positions['date'].min().date()} -> {positions['date'].max().date()}")
    print(f"Positions: {positions_path}")
    print(f"Fold summary: {folds_path}")
    print(folds.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
