"""CLI entry point: train the Deep Momentum Network on one fold.

Placeholder script, no implementation yet.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the LSTM Deep Momentum Network on a single fold.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data-dir", default="data/processed")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _ = args
    raise NotImplementedError


if __name__ == "__main__":
    main()
