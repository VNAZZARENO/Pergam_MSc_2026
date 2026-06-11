"""Backtest utilities: vol-scaling, transaction costs, portfolio aggregation,
TC sensitivity analysis, and equity-curve visualisation.

Functions extracted from notebooks/04_backtest.ipynb and generalised for
reuse across the pipeline.  NB04 imports from here; scripts/04_run_backtest.py
can also call these directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------


def load_variant(models_dir: Path, fold_type: str, suffix: str) -> pd.DataFrame:
    """Concatenate per-fold prediction CSVs for one DMN variant.

    Parameters
    ----------
    models_dir : directory that contains the prediction CSV files.
    fold_type  : walk-forward type label used in file names (e.g. "expanding").
    suffix     : variant suffix, e.g. "nocpd" or "cpd21_s5".

    Returns
    -------
    DataFrame with columns [date, ticker, position, ex_ante_vol, ...] or empty.
    """
    pattern = f"predictions_fold*_{fold_type}_{suffix}.csv"
    files   = sorted(models_dir.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    df["variant"] = suffix
    return df


# ---------------------------------------------------------------------------
# Vol-scaled strategy return  (Paper Eq. 11)
# ---------------------------------------------------------------------------


def vol_scaled_strategy_return(
    positions: np.ndarray | pd.Series,
    ret_real:  np.ndarray | pd.Series,
    ex_ante_vol: np.ndarray | pd.Series,
    target_vol: float = 0.15,
) -> np.ndarray:
    """Volatility-scaled per-(stock, date) strategy return.

    Implements Paper Eq. 11:
        r_strat = position * (target_vol / sigma_t) * r_t

    Parameters
    ----------
    positions   : trading position for each (stock, date).
    ret_real    : realised next-day return.
    ex_ante_vol : ex-ante daily volatility estimate.
    target_vol  : annualised target volatility (default 15 %).

    Returns
    -------
    Array of per-(stock, date) strategy returns.
    """
    return np.asarray(positions) * (target_vol / np.maximum(ex_ante_vol, 1e-6)) * np.asarray(ret_real)


# ---------------------------------------------------------------------------
# Transaction costs  (Paper Eq. C1)
# ---------------------------------------------------------------------------


def add_transaction_costs(
    df: pd.DataFrame,
    position_col: str,
    vol_col: str,
    gross_col: str,
    net_col: str,
    cost: float = 0.0025,
    target_vol: float = 0.15,
) -> pd.DataFrame:
    """Subtract turnover-proportional transaction costs from gross returns.

    Implements Paper Eq. C1:
        net_ret = gross_ret - cost * target_vol * |Δ(position / sigma_t)|

    Turnover is computed per ticker in chronological order.

    Parameters
    ----------
    df           : DataFrame with at least [date, ticker, position_col, vol_col, gross_col].
    position_col : column name of raw positions.
    vol_col      : column name of ex-ante volatility.
    gross_col    : column name of gross strategy returns.
    net_col      : output column name for net returns.
    cost         : round-trip cost (e.g. 0.0025 = 25 bps).
    target_vol   : annualised target volatility.

    Returns
    -------
    DataFrame with ``net_col`` added.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["_scaled_pos"]   = df[position_col] / np.maximum(df[vol_col], 1e-6)
    df["_d_scaled_pos"] = df.groupby("ticker")["_scaled_pos"].diff().fillna(0.0)
    df[net_col] = df[gross_col] - cost * target_vol * df["_d_scaled_pos"].abs()
    return df.drop(columns=["_scaled_pos", "_d_scaled_pos"])


# ---------------------------------------------------------------------------
# Portfolio aggregation
# ---------------------------------------------------------------------------


def to_portfolio_series(df: pd.DataFrame, strat_col: str) -> pd.Series:
    """Equal-weight average across stocks at each date.

    Matches the paper's convention: the strategy is defined per asset, then
    averaged across the universe.

    Parameters
    ----------
    df        : DataFrame with [date, strat_col].
    strat_col : column to aggregate.

    Returns
    -------
    pd.Series indexed by date.
    """
    return df.groupby("date")[strat_col].mean().sort_index()


# ---------------------------------------------------------------------------
# Vol rescaling  (Paper Exhibit 4)
# ---------------------------------------------------------------------------


def rescale_to_target_vol(
    returns: pd.Series,
    target_vol: float = 0.15,
) -> pd.Series:
    """Rescale a return series to a target annualised volatility.

    Used to put all strategies on the same risk footing for comparison
    (Paper Exhibit 4).

    Parameters
    ----------
    returns    : daily return series.
    target_vol : target annualised volatility.

    Returns
    -------
    Rescaled return series.
    """
    r = returns.dropna()
    if len(r) < 2:
        return r
    realised_vol = r.std() * np.sqrt(252)
    if realised_vol == 0:
        return r
    return r * (target_vol / realised_vol)


# ---------------------------------------------------------------------------
# TC sensitivity  (Paper Exhibit 8)
# ---------------------------------------------------------------------------


def sharpe_at_cost(
    df: pd.DataFrame,
    cost: float,
    position_col: str = "position",
    vol_col: str = "ex_ante_vol",
    gross_col: str = "strat_ret_gross",
) -> float:
    """Compute out-of-sample Sharpe after applying a given per-transaction cost.

    Used to build the TC sensitivity curve (Paper Exhibit 8).

    Parameters
    ----------
    df           : DataFrame with position, vol, and gross returns per (date, ticker).
    cost         : round-trip cost in decimal (e.g. 0.0025 = 25 bps).
    position_col : column name of raw positions.
    vol_col      : column name of ex-ante volatility.
    gross_col    : column name of gross strategy returns.

    Returns
    -------
    Annualised Sharpe ratio (float).
    """
    from src.metrics import compute_display_metrics
    adj  = add_transaction_costs(
        df, position_col=position_col, vol_col=vol_col,
        gross_col=gross_col, net_col="_tmp_net", cost=cost,
    )
    port = to_portfolio_series(adj, "_tmp_net")
    return compute_display_metrics(port).get("Sharpe", np.nan)


# ---------------------------------------------------------------------------
# Equity-curve visualisation
# ---------------------------------------------------------------------------


def _set_xlim_to_plotted_data(ax: "plt.Axes") -> None:
    """Resserre l'axe X aux données effectivement tracées (exclut les NaT)."""
    if not _HAS_MPL:
        return
    all_xdata = [x for line in ax.get_lines() for x in line.get_xdata()]
    valid = [x for x in all_xdata if pd.notna(x)]
    if valid:
        ax.set_xlim(min(valid), max(valid))


def plot_equity_curves(
    portfolios_dict: dict,
    title_suffix: str,
    labels: dict,
    log: bool = False,
) -> "plt.Figure":
    """Trace les equity curves pour les stratégies listées dans ``labels``.

    Parameters
    ----------
    portfolios_dict : dict {label → pd.Series de rendements journaliers}.
    title_suffix    : texte ajouté au titre (ex. "raw signal (25 bps cost)").
    labels          : dict {label_interne → label_affichage}.
    log             : si True, axe Y en échelle logarithmique.

    Returns
    -------
    Figure matplotlib.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plot_equity_curves")
    fig, ax = plt.subplots(figsize=(14, 6))
    for label, ret in portfolios_dict.items():
        if label not in labels:
            continue
        cum = (1 + ret.dropna()).cumprod()
        ax.plot(cum.index, cum.values, lw=1.2, label=labels[label])
    _set_xlim_to_plotted_data(ax)
    ax.axhline(1.0, color="black", lw=1.5, linestyle="--")
    if log:
        ax.set_yscale("log")
    ax.set_title(f"Out-of-sample equity curves ({title_suffix})")
    ax.set_ylabel("Cumulative return" + (" (log)" if log else ""))
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    return fig


def plot_drawdowns(
    portfolios_dict: dict,
    target_vol: float,
    labels: dict,
) -> "plt.Figure":
    """Trace les drawdowns pour les stratégies listées dans ``labels``.

    Parameters
    ----------
    portfolios_dict : dict {label → pd.Series de rendements journaliers}.
    target_vol      : vol cible utilisée pour le titre (ex. 0.15 = 15 %).
    labels          : dict {label_interne → label_affichage}.

    Returns
    -------
    Figure matplotlib.
    """
    if not _HAS_MPL:
        raise ImportError("matplotlib is required for plot_drawdowns")
    fig, ax = plt.subplots(figsize=(14, 5))
    for label, ret in portfolios_dict.items():
        if label not in labels:
            continue
        cum = (1 + ret.dropna()).cumprod()
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax.plot(dd.index, dd.values, lw=1.0, label=labels[label], alpha=0.85)
    _set_xlim_to_plotted_data(ax)
    ax.set_title(f"Drawdowns (rescaled to {target_vol:.0%} vol)")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    return fig
