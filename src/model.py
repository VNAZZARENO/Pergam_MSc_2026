"""Deep Momentum Network (LSTM) with a Sharpe-ratio loss.

Thin re-export of :mod:`dmn` for backward compatibility. Use :mod:`dmn`
directly in new code.
"""

from __future__ import annotations

from src.dmn import DeepMomentumNetwork, sharpe_loss

__all__ = ["DeepMomentumNetwork", "sharpe_loss"]
