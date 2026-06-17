import torch
import torch.nn as nn


class DeepMomentumNetwork(nn.Module):
    """LSTM à une couche produisant des positions de trading.

    STRUCTURE :
        features (63 jours) → LSTM(hidden=20) → Dropout → Dense(1) → activation → position

    L'activation dépend de `long_only` :
        - long_only=False (papier original) : tanh  → position dans (-1, +1)
                                              négatif = vendre, positif = acheter
        - long_only=True  (adapté Pergam)   : sigmoid → position dans (0, +1)
                                              0 = pas en portefeuille, 1 = plein poids

    Paramètres
    ----------
    n_features  : nombre de features en entrée (10 pour V1 avec CPD)
    hidden_size : taille de la mémoire interne du LSTM (défaut 20, comme le papier)
    dropout     : taux de régularisation pour éviter l'overfitting
    long_only   : True → Pergam (positions longues seulement)
    """

    def __init__(self, n_features, hidden_size=20, dropout=0.3, long_only=False):
        super().__init__()
        self.long_only = long_only

        # Le LSTM lit la séquence temporelle et produit un vecteur caché à chaque pas.
        # batch_first=True : les dimensions sont (batch, séquence, features).
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # le dropout LSTM ne s'applique qu'entre plusieurs couches
        )

        # Dropout appliqué sur la sortie du LSTM → réduit l'overfitting
        self.dropout = nn.Dropout(dropout)

        # Couche linéaire : compresse le vecteur caché (taille hidden) en un scalaire
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x : (batch, seq_len=63, n_features)
        h, _ = self.lstm(x)          # h : (batch, seq_len, hidden_size)
        h = self.dropout(h)
        out = self.head(h).squeeze(-1)  # (batch, seq_len) — un scalaire par jour
        # On applique tanh ou sigmoid selon le mode Pergam ou papier
        return torch.sigmoid(out) if self.long_only else torch.tanh(out)


def sharpe_loss(positions, returns, target_vol=0.15,
                ex_ante_vol=None, transaction_cost=0.0, eps=1e-8):
    """Loss = Sharpe ratio annualisé négatif de la stratégie.

    POURQUOI LE SHARPE ET PAS UNE ERREUR CLASSIQUE (MSE) ?
      On ne prédit pas un prix — on veut maximiser un ratio rendement/risque.
      Un modèle peut avoir 60% de bonnes directions mais perdre de l'argent
      si ses erreurs arrivent lors des gros mouvements. En optimisant le Sharpe,
      on force le modèle à produire des positions stables et rentables.

    COMMENT CA MARCHE (éq. 11 + C1 du papier) :
      1. Pondération par la volatilité : position × (vol_cible / vol_actuelle) × rendement
         → la stratégie cible toujours 15% de volatilité annualisée.
      2. Si transaction_cost > 0 : on soustrait le coût de chaque changement de position.
         → le modèle apprend à ne pas sur-trader (critique à 25 bps chez Pergam).
      3. Loss = -Sharpe annualisé = -(moyenne / écart-type) × √252

    Paramètres
    ----------
    positions        : (batch, seq_len) — positions produites par le LSTM
    returns          : (batch, seq_len) — rendements réels du lendemain
    target_vol       : volatilité cible annualisée (défaut 15%)
    ex_ante_vol      : (batch, seq_len) — volatilité ex-ante journalière
    transaction_cost : coût par unité de changement de position pondérée (25 bps = 0.0025)
    """
    if ex_ante_vol is None:
        # Sans pondération de volatilité (mode simplifié)
        scaled_pos = positions
        scaled_ret = positions * returns
    else:
        # Pondération : on ramène la position à la volatilité cible (éq. 11)
        scaled_pos = positions / (ex_ante_vol + eps)
        scaled_ret = positions * (target_vol / (ex_ante_vol + eps)) * returns

    if transaction_cost > 0:
        # Coût = |changement de position pondérée| × coût × vol_cible (éq. C1)
        # prev_scaled = position pondérée du jour précédent (zéro au premier jour)
        prev_scaled = torch.cat(
            [torch.zeros_like(scaled_pos[:, :1]), scaled_pos[:, :-1]], dim=1
        )
        turnover   = (scaled_pos - prev_scaled).abs()
        scaled_ret = scaled_ret - transaction_cost * target_vol * turnover

    # Aplatir en un vecteur 1D et supprimer les NaN/infinis
    flat = scaled_ret.reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() < 2:
        return torch.tensor(0.0, device=positions.device, requires_grad=True)

    # Sharpe annualisé = (moyenne / écart-type) × √252
    mean   = flat.mean()
    std    = flat.std() + eps
    sharpe = mean / std * (252.0 ** 0.5)
    return -sharpe  # négatif car on minimise la loss
