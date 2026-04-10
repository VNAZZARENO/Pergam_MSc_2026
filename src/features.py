"""Feature engineering: normalized multi-horizon returns and MACD.

Placeholder module — no implementation yet.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def normalized_returns(returns, horizons: Iterable[int]):
    """Compute volatility-normalized returns over the given horizons."""
    raise NotImplementedError


def macd(prices, pairs: Sequence[Tuple[int, int]]):
    """Compute MACD indicators for the given (short, long) pairs."""
    raise NotImplementedError
