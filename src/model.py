"""Deep Momentum Network (LSTM) with a Sharpe-ratio loss.

Placeholder module — no implementation yet. The concrete framework
(PyTorch / TensorFlow) will be decided at implementation time; for now we
only expose the target API.
"""

from __future__ import annotations


class DeepMomentumNetwork:
    """LSTM-based Deep Momentum Network returning positions in (-1, 1).

    Placeholder — to be subclassed from ``torch.nn.Module`` or
    ``tf.keras.Model`` once the framework is chosen.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


def sharpe_loss(positions, returns):
    """Negative annualized Sharpe ratio, used as training loss."""
    raise NotImplementedError
