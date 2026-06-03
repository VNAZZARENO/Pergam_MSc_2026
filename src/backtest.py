"""Expanding-window backtest harness.

Placeholder module. The full backtest logic lives in scripts/04_run_backtest.py.
Common utility functions will be extracted here as the project matures.
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
