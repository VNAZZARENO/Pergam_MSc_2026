"""Changepoint detection via Gaussian Processes.

Implements (will implement) the Matérn 3/2 kernel fit and the changepoint
kernel from Wood, Roberts & Zohren (2022). Placeholder module — no
implementation yet.
"""

from __future__ import annotations


def fit_matern(returns):
    """Fit a GP with a Matérn 3/2 kernel on a return window."""
    raise NotImplementedError


def fit_changepoint_kernel(returns):
    """Fit a GP with the sigmoid-blended changepoint kernel."""
    raise NotImplementedError


def cpd_scores(returns, lbw: int):
    """Return the (severity, location) pair (nu, gamma) for a lookback window.

    Parameters
    ----------
    returns : pandas.Series or numpy.ndarray
        Standardized returns over the lookback window.
    lbw : int
        Lookback window size in days.

    Returns
    -------
    tuple
        (nu, gamma), both in (0, 1).
    """
    raise NotImplementedError
