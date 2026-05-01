"""Model helpers for the Deep Momentum Network step.

The project keeps two model layers:

* ``RidgePositionModel`` is the lightweight DMN-lite baseline.
* ``DeepMomentumNetwork`` is the PyTorch LSTM used for the paper-style
  Sharpe-loss experiment.

Both models return bounded stock positions in [-1, 1], so they can be evaluated
by the same backtest code.
"""

from __future__ import annotations

import sys

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd


def _require_torch():
    """Import PyTorch only for the LSTM path.

    Keeping this lazy avoids loading PyTorch when the ridge baseline is used,
    which prevents OpenMP runtime conflicts on some Windows/Anaconda setups.
    """
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise ImportError(
            "PyTorch is required for the LSTM DMN. Install torch in the active "
            "environment first."
        ) from exc
    return torch, nn


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
    """LSTM allocation model inspired by the paper's DMN block.

    The network consumes a rolling sequence of stock-level features and outputs
    one bounded position for the next trading day.
    """

    def __new__(cls, *args, **kwargs):
        torch, nn = _require_torch()

        class _DeepMomentumNetworkImpl(nn.Module):
            def __init__(
                self,
                n_features,
                hidden_size=32,
                num_layers=1,
                dropout=0.0,
                max_position=1.0,
            ):
                super().__init__()
                self.max_position = float(max_position)
                lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
                self.lstm = nn.LSTM(
                    input_size=int(n_features),
                    hidden_size=int(hidden_size),
                    num_layers=int(num_layers),
                    dropout=lstm_dropout,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(int(hidden_size)),
                    nn.Linear(int(hidden_size), 1),
                    nn.Tanh(),
                )

            def forward(self, x):
                output, _ = self.lstm(x)
                last_hidden = output[:, -1, :]
                return self.max_position * self.head(last_hidden).squeeze(-1)

        return _DeepMomentumNetworkImpl(*args, **kwargs)


def sharpe_loss(positions, returns, eps=1e-8):
    """Negative annualized Sharpe ratio, useful as a model-selection objective."""
    positions = np.asarray(positions, dtype=float)
    returns = np.asarray(returns, dtype=float)
    pnl = positions * returns
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) == 0:
        return 0.0
    return -float(np.sqrt(252.0) * pnl.mean() / (pnl.std() + eps))


def torch_sharpe_loss(positions, returns, eps=1e-6):
    """Differentiable negative annualized Sharpe ratio for PyTorch training."""
    torch, _ = _require_torch()
    pnl = positions * returns
    mean = pnl.mean()
    std = pnl.std(unbiased=False).clamp_min(eps)
    return -torch.sqrt(torch.tensor(252.0, device=pnl.device)) * mean / std


def set_torch_seed(seed=42):
    """Set deterministic seeds for reproducible CPU experiments."""
    torch, _ = _require_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
