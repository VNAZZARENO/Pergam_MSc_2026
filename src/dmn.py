"""Deep Momentum Network: LSTM-based architecture for time-series momentum,
following Lim, Zohren & Roberts (2019) and Wood, Roberts & Zohren (2022).

The model takes per-asset feature sequences and outputs positions in (-1, 1)
via a tanh head. Training optimizes the (negative) Sharpe ratio of the
resulting strategy returns, end-to-end.

Reference equations:
    Eq. 11  -- Volatility-scaled strategy return
    Eq. 13  -- LSTM forward pass producing position sequences
    Eq. 14  -- Sharpe-ratio loss
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepMomentumNetwork(nn.Module):
    """Single-layer LSTM head producing trading positions.

    The output activation depends on ``long_only``:
        - long_only=False (default, paper):      tanh    → positions in (-1, 1)
        - long_only=True  (long-only framework): sigmoid → positions in  (0, 1)

    Inputs
    ------
    x : (batch, seq_len, n_features) tensor of stock-level features.

    Outputs
    -------
    positions : (batch, seq_len) tensor, in (-1, 1) or (0, 1).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 20,
        dropout: float = 0.3,
        long_only: bool = False,
    ) -> None:
        super().__init__()
        self.long_only = long_only
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # nn.LSTM dropout only applies between stacked layers
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)
        h = self.dropout(h)
        out = self.head(h).squeeze(-1)  # (batch, seq_len)
        return torch.sigmoid(out) if self.long_only else torch.tanh(out)


def sharpe_loss(
    positions: torch.Tensor,
    returns: torch.Tensor,
    target_vol: float = 0.15,
    ex_ante_vol: torch.Tensor | None = None,
    transaction_cost: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative annualised Sharpe ratio of the volatility-scaled strategy.

    When ``ex_ante_vol`` is provided, positions are rescaled so the strategy
    targets ``target_vol`` annualized volatility (Paper Eq. 11).

    If ``transaction_cost > 0``, the cost of turnover is subtracted at each
    step (Paper Eq. C1).  The cost is on |Δ(position / ex_ante_vol)|, i.e.
    the change in the vol-scaled unleveraged position.

    Parameters
    ----------
    positions : (batch, seq_len) tensor of trading positions.
    returns   : (batch, seq_len) tensor of next-period asset returns.
    target_vol : annualized target volatility (default 15%).
    ex_ante_vol : (batch, seq_len) tensor of ex-ante daily volatility.
    transaction_cost : cost per unit change in scaled position.
    eps : small constant for numerical stability.

    Returns
    -------
    loss : scalar tensor, the negative annualised Sharpe ratio.
    """
    if ex_ante_vol is None:
        scaled_pos = positions
        scaled_ret = positions * returns
    else:
        scaled_pos = positions / (ex_ante_vol + eps)
        scaled_ret = positions * (target_vol / (ex_ante_vol + eps)) * returns

    if transaction_cost > 0:
        # Turnover: change in scaled position. Pad t-1 with zeros.
        prev_scaled = torch.cat(
            [torch.zeros_like(scaled_pos[:, :1]), scaled_pos[:, :-1]], dim=1
        )
        turnover   = (scaled_pos - prev_scaled).abs()
        scaled_ret = scaled_ret - transaction_cost * target_vol * turnover

    flat = scaled_ret.reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() < 2:
        return torch.tensor(0.0, device=positions.device, requires_grad=True)

    mean   = flat.mean()
    std    = flat.std() + eps
    sharpe = mean / std * (252.0 ** 0.5)
    return -sharpe
