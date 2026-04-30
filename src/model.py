"""Model helpers for the Deep Momentum Network step.

The paper uses an LSTM trained with a Sharpe-ratio objective. For the first
working implementation we provide a lightweight, dependency-free approximation:
a ridge model that maps momentum, volatility and CPD features to bounded
positions. This gives the project a complete train -> positions -> backtest
pipeline before the heavier LSTM is added.
"""

from __future__ import annotations

import sys

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd


class RidgePositionModel:
    """Linear ridge model returning positions in [-1, 1]."""

    def __init__(self, feature_cols, alpha=10.0, position_scale=1.0):
        self.feature_cols = list(feature_cols)
        self.alpha = float(alpha)
        self.position_scale = float(position_scale)
        self.feature_mean_ = None
        self.feature_std_ = None
        self.coef_ = None

    def _prepare_x(self, frame, fit=False):
        x = frame[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        if fit:
            self.feature_mean_ = x.mean()
            self.feature_std_ = x.std().replace(0.0, 1.0)
        x = x.fillna(self.feature_mean_)
        z = (x - self.feature_mean_) / self.feature_std_
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return np.column_stack([np.ones(len(z)), z.to_numpy(dtype=float)])

    def fit(self, frame, target_col):
        """Fit the ridge model on a supervised training frame."""
        train = frame.dropna(subset=[target_col]).copy()
        if train.empty:
            raise ValueError("Cannot fit RidgePositionModel on an empty training set.")

        x = self._prepare_x(train, fit=True)
        y = pd.to_numeric(train[target_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if len(y) == 0:
            raise ValueError("Training target contains no finite values.")

        penalty = self.alpha * np.eye(x.shape[1])
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        return self

    def predict_score(self, frame):
        """Predict an unbounded score."""
        if self.coef_ is None:
            raise ValueError("Model must be fitted before prediction.")
        x = self._prepare_x(frame, fit=False)
        return x @ self.coef_

    def predict_position(self, frame):
        """Predict a bounded trading position."""
        score = self.predict_score(frame)
        scale = self.position_scale if self.position_scale > 0 else 1.0
        return np.tanh(score / scale)


class DeepMomentumNetwork:
    """Placeholder API for the future LSTM-based DMN implementation."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The full LSTM DMN is a next step. Use RidgePositionModel for the "
            "current working implementation."
        )


def sharpe_loss(positions, returns, eps=1e-8):
    """Negative annualized Sharpe ratio, useful as a model-selection objective."""
    positions = np.asarray(positions, dtype=float)
    returns = np.asarray(returns, dtype=float)
    pnl = positions * returns
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        return 0.0
    return -float(np.sqrt(252.0) * pnl.mean() / (pnl.std() + eps))
