"""NB04 helpers: portfolio-level backtest of DMN variants.

Inputs : positions.parquet (NB03) + panel.parquet + benchmark_ew.parquet (NB01)
Outputs: backtest_portfolio.parquet, backtest_metrics.parquet

All metrics are out-of-sample (walk-forward test periods only).
Returns are net of 25 bps transaction costs (already encoded in strategy_return).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SQRT_252 = math.sqrt(252)
TRANSACTION_COST_BPS = 25.0
TRANSACTION_COST_RATE = TRANSACTION_COST_BPS / 10_000.0

VARIANT_COLORS = {
    "baseline":   "#64748B",
    "cpd_cusum":  "#0891B2",
    "cpd_gp":     "#1B2A6B",
    "cpd_bocpd":  "#D97706",
}

BENCHMARK_COLORS = {
    "SXXR":  "#DC2626",
    "EW":    "#059669",
}


# ---------------------------------------------------------------------------
# Paths / load
# ---------------------------------------------------------------------------

def nb04_paths(root: Path) -> dict[str, Path]:
    proc = root / "data" / "processed" / "stoxx600"
    return {
        "positions":          proc / "positions.parquet",
        "panel":              proc / "panel.parquet",
        "benchmark_ew":       proc / "benchmark_ew.parquet",
        "known_events":       proc / "known_events.csv",
        "backtest_portfolio": proc / "backtest_portfolio.parquet",
        "backtest_metrics":   proc / "backtest_metrics.parquet",
    }


def load_nb04_inputs(root: Path) -> dict:
    paths = nb04_paths(root)
    missing = [k for k in ("positions", "panel", "benchmark_ew") if not paths[k].exists()]
    if missing:
        raise FileNotFoundError(f"Missing NB01/NB03 outputs: {[str(paths[k]) for k in missing]}")

    positions    = pd.read_parquet(paths["positions"])
    benchmark_ew = pd.read_parquet(paths["benchmark_ew"])
    panel_ret    = pd.read_parquet(paths["panel"], columns=["date", "ticker", "next_return"])
    known_events = (pd.read_csv(paths["known_events"], parse_dates=["event_date"])
                    if paths["known_events"].exists()
                    else pd.DataFrame(columns=["event", "event_date"]))

    for df in [positions, benchmark_ew, panel_ret]:
        df["date"] = pd.to_datetime(df["date"])
    if not known_events.empty:
        known_events["event_date"] = pd.to_datetime(known_events["event_date"])

    # Attach realized returns and ensure variant label
    positions = positions.merge(
        panel_ret.rename(columns={"next_return": "target_return"}),
        on=["date", "ticker"], how="left",
    )
    if "variant" not in positions.columns:
        positions["variant"] = "baseline"

    return {"paths": paths, "positions": positions,
            "benchmark_ew": benchmark_ew, "known_events": known_events}


# ---------------------------------------------------------------------------
# Portfolio aggregation
# ---------------------------------------------------------------------------

def build_portfolio(positions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stock-level positions into a fully-invested long-only portfolio.

    DMN positions ∈ [0, 1] are treated as allocation scores and normalised to
    sum to 1 per (date, variant), so the portfolio is always 100% invested and
    directly comparable to SXXR / EW benchmarks.
    """
    positions = positions.copy().sort_values(["variant", "ticker", "date"]).reset_index(drop=True)

    # Normalise raw positions → portfolio weights summing to 1 per day
    pos_sum = (positions.groupby(["date", "variant"])["position"]
               .transform("sum").clip(lower=1e-8))
    positions["weight"] = positions["position"] / pos_sum

    # Turnover on normalised weights (comparable to standard portfolio turnover)
    if "turnover" not in positions.columns:
        previous = positions.groupby(["variant", "ticker"], sort=False)["weight"].shift(1).fillna(0.0)
        positions["turnover"] = (positions["weight"] - previous).abs()

    if "transaction_cost" not in positions.columns:
        positions["transaction_cost"] = TRANSACTION_COST_RATE * positions["turnover"]

    if "gross_strategy_return" not in positions.columns:
        target_col = next(
            (col for col in ["target_return", "realized_return", "target_next_return"] if col in positions.columns),
            None,
        )
        if target_col is not None:
            positions["gross_strategy_return"] = positions["weight"] * positions[target_col]
        elif "strategy_return" in positions.columns:
            positions["gross_strategy_return"] = positions["strategy_return"]
        else:
            raise KeyError(
                "positions must contain gross_strategy_return, strategy_return, "
                "or a target return column."
            )

    positions["strategy_return"] = positions["gross_strategy_return"] - positions["transaction_cost"]

    portfolio = (
        positions.groupby(["date", "variant"])
        .agg(
            net_return=("strategy_return",         "sum"),   # Σ w_i*r_i - Σ TC_i
            gross_return=("gross_strategy_return", "sum"),   # Σ w_i*r_i
            mean_turnover=("turnover",             "mean"),
            mean_cost=("transaction_cost",         "mean"),
            n_stocks=("ticker",                    "nunique"),
            mean_position=("position",             "mean"),
        )
        .reset_index()
        .sort_values(["variant", "date"])
        .reset_index(drop=True)
    )
    # Cumulative returns
    for variant, grp in portfolio.groupby("variant"):
        idx = grp.index
        portfolio.loc[idx, "cum_gross"] = (1 + grp["gross_return"].fillna(0)).cumprod()
        portfolio.loc[idx, "cum_net"]   = (1 + grp["net_return"].fillna(0)).cumprod()
    return portfolio


def add_benchmark(portfolio: pd.DataFrame, benchmark_ew: pd.DataFrame) -> pd.DataFrame:
    """Append SXXR and EW benchmark series aligned to the portfolio date range."""
    date_min = portfolio["date"].min()
    date_max = portfolio["date"].max()

    bm = benchmark_ew.loc[
        benchmark_ew["date"].between(date_min, date_max)
    ][["date", "ew_1d_ret", "sxxr_1d_ret"]].sort_values("date").copy()

    rows = []
    for label, col in [("EW", "ew_1d_ret"), ("SXXR", "sxxr_1d_ret")]:
        if col not in bm.columns:
            continue
        bm_sub = bm[["date", col]].dropna().rename(columns={col: "net_return"})
        bm_sub["variant"] = label
        bm_sub["gross_return"] = bm_sub["net_return"]
        bm_sub["mean_turnover"] = np.nan
        bm_sub["mean_cost"] = 0.0
        bm_sub["n_stocks"] = np.nan
        bm_sub["mean_position"] = np.nan
        bm_sub["cum_gross"] = (1 + bm_sub["net_return"].fillna(0)).cumprod()
        bm_sub["cum_net"]   = bm_sub["cum_gross"]
        rows.append(bm_sub)

    if rows:
        return pd.concat([portfolio, *rows], ignore_index=True)
    return portfolio


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _metrics(returns: pd.Series, turnover: pd.Series | None = None, eps: float = 1e-8) -> dict:
    r = returns.dropna()
    if len(r) < 20:
        return {}
    ann = SQRT_252
    mean_r = r.mean()
    std_r  = r.std(ddof=0)
    neg_r  = r[r < 0]
    downside = neg_r.std(ddof=0) if len(neg_r) > 5 else std_r
    sharpe  = ann * mean_r / (std_r + eps)
    sortino = ann * mean_r / (downside + eps)
    ann_ret = float((1 + r).prod() ** (252 / len(r)) - 1)
    cum = (1 + r).cumprod()
    dd  = 1 - cum / cum.cummax()
    max_dd  = float(dd.max())
    calmar  = ann_ret / (max_dd + eps)
    win_rate = float((r > 0).mean())
    out = {
        "sharpe":       round(sharpe, 3),
        "sortino":      round(sortino, 3),
        "ann_return":   round(ann_ret, 4),
        "ann_vol":      round(std_r * ann, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar":       round(calmar, 3),
        "win_rate":     round(win_rate, 3),
        "n_days":       len(r),
    }
    if turnover is not None:
        out["mean_daily_turnover"] = round(float(turnover.dropna().mean()), 4)
        out["ann_cost_bps"] = round(float(turnover.dropna().mean() * TRANSACTION_COST_BPS * 252), 1)
    return out


def compute_metrics_table(portfolio: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, grp in portfolio.groupby("variant"):
        ret_col = "net_return" if variant not in ("EW", "SXXR") else "net_return"
        to_col  = "mean_turnover" if "mean_turnover" in grp.columns else None
        m = _metrics(grp[ret_col], grp[to_col] if to_col else None)
        m["variant"] = variant
        rows.append(m)
    cols = ["variant", "sharpe", "sortino", "ann_return", "ann_vol",
            "max_drawdown", "calmar", "win_rate", "mean_daily_turnover",
            "ann_cost_bps", "n_days"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]].sort_values("sharpe", ascending=False).reset_index(drop=True)


def yearly_metrics(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Net Sharpe per year per variant."""
    port = portfolio.copy()
    port["year"] = port["date"].dt.year
    rows = []
    for (variant, year), grp in port.groupby(["variant", "year"]):
        m = _metrics(grp["net_return"])
        if m:
            rows.append({"variant": variant, "year": year, "sharpe": m["sharpe"],
                         "ann_return": m.get("ann_return", np.nan),
                         "max_drawdown": m.get("max_drawdown", np.nan)})
    return pd.DataFrame(rows).sort_values(["variant", "year"]).reset_index(drop=True)


def input_summary(data: dict, portfolio: pd.DataFrame) -> pd.DataFrame:
    pos = data["positions"]
    variants = sorted(pos["variant"].unique())
    return pd.DataFrame([
        {"item": "variants",     "value": ", ".join(variants)},
        {"item": "date range",   "value": f"{portfolio['date'].min().date()} → {portfolio['date'].max().date()}"},
        {"item": "trading days", "value": portfolio["date"].nunique()},
        {"item": "n_stocks (avg/day)", "value": f"{pos.groupby('date')['ticker'].nunique().mean():.0f}"},
        {"item": "costs",        "value": f"{TRANSACTION_COST_BPS:.0f} bps per leg"},
    ])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cumulative_pnl(portfolio: pd.DataFrame, known_events: pd.DataFrame | None = None) -> go.Figure:
    """Cumulative net return (rebased to 100) for all variants and benchmarks."""
    fig = go.Figure()
    all_variants = sorted(portfolio["variant"].unique())
    dmn_variants = [v for v in all_variants if v not in ("EW", "SXXR")]
    benchmarks   = [v for v in all_variants if v in ("EW", "SXXR")]

    for variant in dmn_variants:
        grp = portfolio.loc[portfolio["variant"].eq(variant)].sort_values("date")
        fig.add_trace(go.Scatter(
            x=grp["date"], y=grp["cum_net"] * 100,
            mode="lines", name=variant,
            line=dict(color=VARIANT_COLORS.get(variant, "#333"), width=1.8),
        ))
    for bm in benchmarks:
        grp = portfolio.loc[portfolio["variant"].eq(bm)].sort_values("date")
        fig.add_trace(go.Scatter(
            x=grp["date"], y=grp["cum_net"] * 100,
            mode="lines", name=bm,
            line=dict(color=BENCHMARK_COLORS.get(bm, "#999"), width=1.2, dash="dash"),
        ))

    if known_events is not None and not known_events.empty:
        for ev in known_events["event_date"]:
            if portfolio["date"].min() <= ev <= portfolio["date"].max():
                fig.add_vline(x=ev, line=dict(color="rgba(220,0,0,0.12)", width=1, dash="dot"))

    fig.update_layout(
        title="Cumulative net return — walk-forward out-of-sample (rebased 100)",
        yaxis_title="Index", xaxis_title="Date",
        template="plotly_white", height=500, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def plot_rolling_sharpe(portfolio: pd.DataFrame, window: int = 252) -> go.Figure:
    """Rolling annualised Sharpe (252-day window) per variant."""
    fig = go.Figure()
    dmn_variants = [v for v in sorted(portfolio["variant"].unique()) if v not in ("EW", "SXXR")]
    for variant in dmn_variants:
        grp = portfolio.loc[portfolio["variant"].eq(variant)].sort_values("date").copy()
        roll_sharpe = (grp["net_return"]
                       .rolling(window, min_periods=window // 2)
                       .apply(lambda r: SQRT_252 * r.mean() / (r.std(ddof=0) + 1e-8)))
        fig.add_trace(go.Scatter(
            x=grp["date"], y=roll_sharpe,
            mode="lines", name=variant,
            line=dict(color=VARIANT_COLORS.get(variant, "#333"), width=1.5),
        ))
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(
        title=f"Rolling {window}-day Sharpe (net of 25 bps)",
        yaxis_title="Sharpe", template="plotly_white", height=420, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def plot_drawdown(portfolio: pd.DataFrame) -> go.Figure:
    """Drawdown (%) per variant."""
    fig = go.Figure()
    dmn_variants = [v for v in sorted(portfolio["variant"].unique()) if v not in ("EW", "SXXR")]
    for variant in dmn_variants:
        grp = portfolio.loc[portfolio["variant"].eq(variant)].sort_values("date").copy()
        cum = (1 + grp["net_return"].fillna(0)).cumprod()
        dd  = (cum / cum.cummax() - 1) * 100
        fig.add_trace(go.Scatter(
            x=grp["date"], y=dd,
            mode="lines", name=variant, fill="tozeroy",
            line=dict(color=VARIANT_COLORS.get(variant, "#333"), width=1.0),
            opacity=0.5,
        ))
    fig.update_layout(
        title="Drawdown (%) — net of 25 bps",
        yaxis_title="Drawdown (%)", template="plotly_white",
        height=400, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def plot_yearly_sharpe(yearly: pd.DataFrame) -> go.Figure:
    """Yearly net Sharpe bar chart grouped by variant."""
    dmn = yearly.loc[~yearly["variant"].isin(["EW", "SXXR"])]
    fig = px.bar(
        dmn, x="year", y="sharpe", color="variant",
        barmode="group", color_discrete_map=VARIANT_COLORS,
        title="Net Sharpe by year — walk-forward out-of-sample",
    )
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(template="plotly_white", height=450, legend_title_text="",
                      yaxis_title="Net Sharpe (annualised)")
    return fig


def plot_sharpe_distribution(yearly: pd.DataFrame) -> go.Figure:
    """Box plot of yearly net Sharpe distribution per variant."""
    dmn = yearly.loc[~yearly["variant"].isin(["EW", "SXXR"])]
    fig = px.box(
        dmn, x="variant", y="sharpe", color="variant",
        color_discrete_map=VARIANT_COLORS, points="all",
        title="Distribution of yearly net Sharpe — all folds",
    )
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(template="plotly_white", height=420, showlegend=False,
                      yaxis_title="Net Sharpe (annualised)")
    return fig


def plot_turnover(portfolio: pd.DataFrame) -> go.Figure:
    """Rolling 63-day average daily turnover."""
    fig = go.Figure()
    dmn_variants = [v for v in sorted(portfolio["variant"].unique()) if v not in ("EW", "SXXR")]
    for variant in dmn_variants:
        grp = portfolio.loc[portfolio["variant"].eq(variant)].sort_values("date")
        roll_to = grp["mean_turnover"].rolling(63, min_periods=20).mean()
        fig.add_trace(go.Scatter(
            x=grp["date"], y=roll_to * 100,
            mode="lines", name=variant,
            line=dict(color=VARIANT_COLORS.get(variant, "#333"), width=1.5),
        ))
    fig.update_layout(
        title="Rolling 63-day average daily turnover (%)",
        yaxis_title="Turnover (%/day)", template="plotly_white",
        height=380, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_nb04_outputs(root: Path, portfolio: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    paths = nb04_paths(root)
    paths["backtest_portfolio"].parent.mkdir(parents=True, exist_ok=True)
    portfolio.to_parquet(paths["backtest_portfolio"], index=False)
    metrics.to_parquet(paths["backtest_metrics"], index=False)
    return pd.DataFrame([
        {"output": "backtest_portfolio", "rows": len(portfolio),
         "path": paths["backtest_portfolio"].as_posix()},
        {"output": "backtest_metrics",   "rows": len(metrics),
         "path": paths["backtest_metrics"].as_posix()},
    ])
