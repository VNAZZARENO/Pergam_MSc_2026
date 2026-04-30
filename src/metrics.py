"""Risk-adjusted performance metrics."""

from __future__ import annotations

import sys
import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd

# Organisation du fichier :
# indicateurs simples pour evaluer rendement, risque et pertes extremes.


# [METRIC] sharpe
# Mesure le rendement moyen ajuste du risque total.
def sharpe(returns, risk_free=0.0):
    """Annualized Sharpe ratio."""
    excess = returns - risk_free / 252
    if excess.std() < 1e-12:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / excess.std())


# [METRIC] sortino
# Variante du Sharpe qui penalise seulement la volatilite negative.
def sortino(returns, risk_free=0.0):
    """Annualized Sortino ratio."""
    excess = returns - risk_free / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() < 1e-12:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / downside.std())


# [METRIC] max_drawdown
# Calcule la pire perte depuis un plus haut historique.
def max_drawdown(returns):
    """Maximum drawdown of a return series."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


# [METRIC] calmar
# Compare le rendement annualise a la pire chute subie.
def calmar(returns):
    """Calmar ratio (annualized return / |max drawdown|)."""
    mdd = max_drawdown(returns)
    if abs(mdd) < 1e-12:
        return 0.0
    ann_ret = (1 + returns).prod() ** (252 / len(returns)) - 1
    return float(ann_ret / abs(mdd))


# [METRIC] hit_ratio
# Donne la proportion de jours ou la performance est positive.
def hit_ratio(returns):
    """Fraction of positive returns."""
    valid = returns.dropna()
    if len(valid) == 0:
        return 0.0
    return float((valid > 0).sum() / len(valid))


# [METRIC] annual_return
# Annualise le rendement compose de la serie.
def annual_return(returns):
    """Annualized return."""
    n = len(returns)
    if n == 0:
        return 0.0
    return float((1 + returns).prod() ** (252 / n) - 1)


# [METRIC] annual_volatility
# Annualise l'ecart-type des rendements quotidiens.
def annual_volatility(returns):
    """Annualized volatility."""
    return float(returns.std() * np.sqrt(252))


# [TABLE] summary_table
# Regroupe les principales metriques dans un tableau lisible.
def summary_table(returns, name="Strategy"):
    """One-line summary of all metrics.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.DataFrame({
        name: {
            "Ann. Return": f"{annual_return(returns):.1%}",
            "Ann. Vol": f"{annual_volatility(returns):.1%}",
            "Sharpe": f"{sharpe(returns):.2f}",
            "Sortino": f"{sortino(returns):.2f}",
            "Calmar": f"{calmar(returns):.2f}",
            "Max DD": f"{max_drawdown(returns):.1%}",
            "Hit Ratio": f"{hit_ratio(returns):.1%}",
        }
    })
