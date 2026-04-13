"""CLI entry point: precompute (nu, gamma) changepoint scores.

Placeholder script, no implementation yet.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute changepoint severity and location per asset.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lbw", type=int, default=21)
    parser.add_argument("--in-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/processed")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _ = args
    raise NotImplementedError


if __name__ == "__main__":
    main()
