# Utilitaires backtest — vol-scaling, coûts de transaction, portefeuille, visualisation

from pathlib import Path
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# --- chargement des prédictions ---


def load_variant(models_dir, fold_type, suffix):
    pattern = f"predictions_fold*_{fold_type}_{suffix}.csv"
    files   = sorted(models_dir.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"])
    df["variant"] = suffix
    return df


# --- rendement vol-scalé (éq. 11) ---


def vol_scaled_strategy_return(positions, ret_real, ex_ante_vol, target_vol=0.15):
    return np.asarray(positions) * (target_vol / np.maximum(ex_ante_vol, 1e-6)) * np.asarray(ret_real)


# --- coûts de transaction (éq. C1) ---


def add_transaction_costs(df, position_col, vol_col, gross_col, net_col,
                          cost=0.0025, target_vol=0.15):
    df = df.sort_values(["ticker", "date"]).copy()
    df["_scaled_pos"]   = df[position_col] / np.maximum(df[vol_col], 1e-6)
    df["_d_scaled_pos"] = df.groupby("ticker")["_scaled_pos"].diff().fillna(0.0)
    df[net_col] = df[gross_col] - cost * target_vol * df["_d_scaled_pos"].abs()
    return df.drop(columns=["_scaled_pos", "_d_scaled_pos"])


# --- agrégation portefeuille ---


def to_portfolio_series(df, strat_col):
    return df.groupby("date")[strat_col].mean().sort_index()


# --- rescaling vol (Exhibit 4) ---


def rescale_to_target_vol(returns, target_vol=0.15):
    r = returns.dropna()
    if len(r) < 2:
        return r
    realised_vol = r.std() * np.sqrt(252)
    if realised_vol == 0:
        return r
    return r * (target_vol / realised_vol)


# --- sensibilité aux coûts (Exhibit 8) ---


def sharpe_at_cost(df, cost, position_col="position",
                   vol_col="ex_ante_vol", gross_col="strat_ret_gross"):
    from src.metrics import compute_display_metrics
    adj  = add_transaction_costs(df, position_col=position_col, vol_col=vol_col,
                                 gross_col=gross_col, net_col="_tmp_net", cost=cost)
    port = to_portfolio_series(adj, "_tmp_net")
    return compute_display_metrics(port).get("Sharpe", np.nan)


# --- visualisation ---


def _set_xlim_to_plotted_data(ax):
    # resserre l'axe X aux données tracées (exclut les NaT)
    if not _HAS_MPL:
        return
    all_xdata = [x for line in ax.get_lines() for x in line.get_xdata()]
    valid = [x for x in all_xdata if pd.notna(x)]
    if valid:
        ax.set_xlim(min(valid), max(valid))


def plot_equity_curves(portfolios_dict, title_suffix, labels, log=False):
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


def plot_drawdowns(portfolios_dict, target_vol, labels):
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
