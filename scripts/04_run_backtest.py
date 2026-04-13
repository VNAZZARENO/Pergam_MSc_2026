"""CLI entry point: run the full expanding-window backtest and print metrics.

Placeholder script, no implementation yet.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full expanding-window backtest.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _ = args
    raise NotImplementedError


if __name__ == "__main__":
    main()
