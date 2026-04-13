"""Expanding-window backtest harness.

Placeholder module, no implementation yet.
"""

from __future__ import annotations

from typing import Callable


def expanding_window_backtest(data, model_fn: Callable, folds):
    """Run an expanding-window backtest.

    Parameters
    ----------
    data : pandas.DataFrame
        Feature / return panel indexed by date.
    model_fn : Callable
        Factory returning a fresh model for each fold.
    folds : Iterable
        Train / test date splits.

    Returns
    -------
    dict
        Per-fold positions, PnL and metrics.
    """
    raise NotImplementedError
