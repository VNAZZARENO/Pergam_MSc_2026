# Métriques de performance — Sharpe, Sortino, MDD, Calmar

import numpy as np
import pandas as pd


def sharpe(returns, annualization=252):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return r.mean() / r.std(ddof=1) * np.sqrt(annualization)


def sortino(returns, annualization=252):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    downside = r[r < 0]
    if len(downside) < 2:
        return np.nan
    dd = downside.std(ddof=1)
    if dd == 0:
        return np.nan
    return r.mean() / dd * np.sqrt(annualization)


def max_drawdown(returns):
    r = np.asarray(returns, dtype=float)
    cumulative = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / peak
    return float(dd.min())


def calmar(returns, annualization=252):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return np.nan
    cagr = (np.prod(1.0 + r) ** (annualization / n)) - 1.0
    mdd = abs(max_drawdown(r))
    return cagr / mdd if mdd > 0 else np.nan


def hit_ratio(returns):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    return float((r > 0).mean()) if len(r) > 0 else np.nan


def compute_metrics(returns, annualization=252):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    ann_ret = (np.prod(1.0 + r) ** (annualization / n)) - 1.0 if n > 0 else np.nan
    ann_vol = r.std(ddof=1) * np.sqrt(annualization) if n > 1 else np.nan
    return {
        "sharpe":       sharpe(r, annualization),
        "sortino":      sortino(r, annualization),
        "calmar":       calmar(r, annualization),
        "ann_return":   ann_ret,
        "ann_vol":      ann_vol,
        "max_drawdown": max_drawdown(r),
        "hit_ratio":    hit_ratio(r),
    }


# --- affichage NB04 ---

def compute_display_metrics(returns):
    nan_keys = ["Returns", "Vol", "Sharpe", "Downside Dev", "Sortino",
                "MDD", "Calmar", "% +ve", "Avg P / Avg L"]
    r = returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in nan_keys}

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe_ = ann_ret / ann_vol if ann_vol > 0 else np.nan

    downside = r[r < 0]
    dd_dev   = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    sortino_ = ann_ret / dd_dev if dd_dev and dd_dev > 0 else np.nan

    cum     = (1 + r).cumprod()
    dd      = (cum - cum.cummax()) / cum.cummax()
    mdd     = dd.min()
    calmar_ = ann_ret / abs(mdd) if mdd != 0 else np.nan

    pct_pos = float((r > 0).mean())
    avg_p   = r[r > 0].mean() if (r > 0).any() else np.nan
    avg_l   = abs(r[r < 0].mean()) if (r < 0).any() else np.nan
    p_to_l  = avg_p / avg_l if avg_l and avg_l > 0 else np.nan

    return {
        "Returns":       ann_ret,
        "Vol":           ann_vol,
        "Sharpe":        sharpe_,
        "Downside Dev":  dd_dev,
        "Sortino":       sortino_,
        "MDD":           mdd,
        "Calmar":        calmar_,
        "% +ve":         pct_pos,
        "Avg P / Avg L": p_to_l,
    }


def format_metrics(df):
    fmt = df.copy()
    for col in ["Returns", "Vol", "Downside Dev", "MDD"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    for col in ["Sharpe", "Sortino", "Calmar", "Avg P / Avg L"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
    if "% +ve" in fmt.columns:
        fmt["% +ve"] = fmt["% +ve"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    return fmt


# --- diagnostics temporels NB04 ---

def rolling_sharpe(returns, window=252):
    r = returns.dropna()
    return (r.rolling(window).mean() / r.rolling(window).std()) * np.sqrt(252)


def yearly_sharpe(returns):
    r = returns.dropna()
    return r.groupby(r.index.year).apply(
        lambda y: float((y.mean() / y.std()) * np.sqrt(252)) if y.std() > 0 else 0.0
    )
