"""Deep Momentum Network (LSTM) with a Sharpe-ratio loss.

Placeholder module, no implementation yet. The concrete framework
(PyTorch / TensorFlow) will be decided at implementation time; for now we
only expose the target API.
"""

from __future__ import annotations

# Organisation du fichier :
# squelette du futur modele DMN, sans choix definitif PyTorch/TensorFlow.


# [MODEL] DeepMomentumNetwork
# API cible du futur modele LSTM qui produira des positions de trading.
class DeepMomentumNetwork:
    """LSTM-based Deep Momentum Network returning positions in (-1, 1).

    Placeholder, to be subclassed from ``torch.nn.Module`` or
    ``tf.keras.Model`` once the framework is chosen.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


# [LOSS] sharpe_loss
# Fonction de perte prevue pour entrainer le modele a maximiser le Sharpe.
def sharpe_loss(positions, returns):
    """Negative annualized Sharpe ratio, used as training loss."""
    raise NotImplementedError
