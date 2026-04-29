"""Return computation and volatility scaling."""

from __future__ import annotations

import sys
import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd

# Organisation du fichier :
# transformations simples prix -> rendements -> volatilite -> scaling.


# [RETURNS] arithmetic_returns
# Calcule les rendements arithmetiques simples a partir des prix.
def arithmetic_returns(prices, periods=1):
    """Compute simple arithmetic returns from a price series / frame."""
    return prices.pct_change(periods=periods, fill_method=None)


# [RETURNS] log_returns
# Calcule les log-rendements, pratiques pour les sommes temporelles.
def log_returns(prices, periods=1):
    """Compute log returns from a price series / frame."""
    return np.log(prices).diff(periods)


# [VOL] ewm_vol
# Estime une volatilite lisse avec une moyenne exponentielle.
def ewm_vol(returns, span=60):
    """Exponentially-weighted standard deviation of returns."""
    return returns.ewm(span=span, min_periods=span // 2).std()


# [VOL] rolling_vol
# Calcule la volatilite realisee sur une fenetre glissante.
def rolling_vol(returns, window=60, annualise=True):
    """Rolling realised volatility."""
    vol = returns.rolling(window, min_periods=window).std()
    return vol * np.sqrt(252) if annualise else vol


# [SCALING] vol_scale
# Ajuste les rendements pour viser une volatilite annuelle cible.
def vol_scale(returns, target=0.15, span=60):
    """Rescale returns to a target annualized volatility (e.g. 0.15)."""
    vol = ewm_vol(returns, span=span) * np.sqrt(252)
    vol = vol.replace(0, np.nan)
    return returns * (target / vol)
