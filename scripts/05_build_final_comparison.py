"""Build the final Presentation 1 backtest comparison files.

`04_run_backtest.py` runs one model-position file at a time. This script combines
the three model runs with the shared rule-based strategies and benchmarks, then
recomputes the summary from the combined daily returns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd

from src.backtest import performance_summary


DEFAULT_INPUTS = {
    "backtest_returns_with_dmn.csv": {
        "slow_momentum",
        "slow_fast",
        "cpd_adjusted",
        "dmn_lite",
        "benchmark_ew",
        "benchmark_sxxr",
    },
    "backtest_returns_with_lstm.csv": {"dmn_lstm"},
    "backtest_returns_with_lstm_no_cpd.csv": {"dmn_lstm_no_cpd"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine per-model backtest outputs into the final comparison.",
    )
    parser.add_argument("--data-dir", default="data/processed/stoxx600")
    parser.add_argument("--out-returns", default="backtest_returns_final_comparison.csv")
    parser.add_argument("--out-summary", default="backtest_summary_final_comparison.csv")
    return parser


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = build_parser().parse_args()
    data_dir = _resolve(args.data_dir)

    frames = []
    for filename, strategies in DEFAULT_INPUTS.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing backtest input: {path}")
        frame = pd.read_csv(path, parse_dates=["date"])
        frame = frame.loc[frame["strategy"].isin(strategies)].copy()
        frames.append(frame)

    returns = pd.concat(frames, ignore_index=True)
    duplicates = returns.duplicated(subset=["date", "strategy"]).sum()
    if duplicates:
        raise ValueError(f"Combined returns contain {duplicates} duplicate date/strategy rows.")

    summary = performance_summary(returns)
    returns_path = data_dir / args.out_returns
    summary_path = data_dir / args.out_summary
    returns.to_csv(returns_path, index=False)
    summary.to_csv(summary_path, index=False)

    printable = summary.copy()
    percent_cols = ["ann_return", "ann_vol", "max_drawdown", "hit_ratio", "avg_turnover"]
    for col in percent_cols:
        printable[col] = printable[col].map(lambda x: f"{x:.2%}")
    for col in ["sharpe", "sortino", "calmar", "avg_assets"]:
        printable[col] = printable[col].map(lambda x: f"{x:.2f}")

    print("Final comparison built")
    print(f"Strategies: {', '.join(summary['strategy'])}")
    print(f"Returns: {returns_path}")
    print(f"Summary: {summary_path}")
    print(printable.to_string(index=False))


if __name__ == "__main__":
    main()
