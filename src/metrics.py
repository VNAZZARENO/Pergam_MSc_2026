"""Risk-adjusted performance metrics.

Placeholder module, no implementation yet.
"""

from __future__ import annotations


def sharpe(returns):
    """Annualized Sharpe ratio."""
    raise NotImplementedError


def sortino(returns):
    """Annualized Sortino ratio."""
    raise NotImplementedError


def calmar(returns):
    """Calmar ratio (CAGR / max drawdown)."""
    raise NotImplementedError


def max_drawdown(returns):
    """Maximum drawdown of a return series."""
    raise NotImplementedError


def hit_ratio(returns):
    """Fraction of positive returns."""
    raise NotImplementedError
