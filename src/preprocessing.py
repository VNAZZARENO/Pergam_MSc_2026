"""Return computation and volatility scaling.

Placeholder module — no implementation yet.
"""

from __future__ import annotations


def arithmetic_returns(prices):
    """Compute simple arithmetic returns from a price series / frame."""
    raise NotImplementedError


def ewm_vol(returns, span: int = 60):
    """Exponentially-weighted standard deviation of returns."""
    raise NotImplementedError


def vol_scale(returns, target: float = 0.15):
    """Rescale returns to a target annualized volatility."""
    raise NotImplementedError
