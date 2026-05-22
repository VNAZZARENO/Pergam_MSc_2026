"""NB02: changepoint detection on STOXX 600 — 4 methods compared.

Fast (full universe):
  adaptive_cusum  — CUSUM with vol-adjusted threshold (team contribution)
  rolling_ttest   — Welch t-test short vs long window (simple reference)

Heavy (30-stock stratified sample):
  bocpd           — Bayesian online CPD, posterior run-length probability
  gp_matern32     — GP marginal-likelihood ratio, Matern 3/2 kernel (paper)

All methods: causal, online, scores in [0, 1].
NB03 outputs: nu (severity) and gamma (recency) per method, lagged 1 day.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.known_events import known_events_frame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL          = "mkt_rel_1d"       # market-relative daily return
EVENT_PRE_DAYS      = 5                  # event window: days before
EVENT_POST_DAYS     = 10                 # event window: days after
DETECTION_QUANTILE  = 0.95              # score threshold for flagging
NU_GAMMA_WINDOW     = 63               # gamma decay window (days)
RANDOM_SEED         = 42
HEAVY_SAMPLE_SIZE   = 30
MIN_OBS_SCORE       = 252

METHOD_COLORS = {
    "adaptive_cusum": "#1f77b4",
    "rolling_ttest":  "#ff7f0e",
    "bocpd":          "#2ca02c",
    "gp_matern32":    "#d62728",
}
EVENT_TYPE_COLORS = {
    "Credit":           "#e41a1c",
    "Global risk":      "#ff7f00",
    "Sovereign":        "#984ea3",
    "Policy":           "#4daf4a",
    "Political":        "#377eb8",
    "Market structure": "#a65628",
    "Pandemic":         "#f781bf",
    "Geopolitical":     "#999999",
    "Rates":            "#e6ab02",
}


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

def resolve_project_root(start=None) -> Path:
    root = (Path.cwd() if start is None else Path(start)).resolve()
    for p in [root, root.parent]:
        if (p / "configs" / "default.yaml").exists():
            return p
    raise FileNotFoundError("configs/default.yaml not found")


def nb02_paths(root=None) -> dict[str, Path]:
    r = resolve_project_root(root)
    d = r / "data" / "processed" / "stoxx600"
    return {
        "project_root": r,
        "processed_dir": d,
        "panel":         d / "panel.parquet",
        "benchmark_ew":  d / "benchmark_ew.parquet",
        "cpd_scores":    d / "cpd_scores.parquet",
        "cpd_metrics":   d / "cpd_metrics.parquet",
        "cpd_features":  d / "cpd_features_nb03.parquet",
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_nb02_inputs(root=None) -> dict:
    paths = nb02_paths(root)
    for key in ("panel", "benchmark_ew"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing NB01 output: {paths[key]}")

    panel        = pd.read_parquet(paths["panel"])
    benchmark_ew = pd.read_parquet(paths["benchmark_ew"])
    for df in (panel, benchmark_ew):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    if TARGET_COL not in panel.columns:
        raise ValueError(f"Column '{TARGET_COL}' not found in panel. Run NB01 first.")

    events = known_events_frame(panel["date"].min(), panel["date"].max())
    return {
        "paths":        paths,
        "panel":        panel,
        "benchmark_ew": benchmark_ew,
        "known_events": events,
    }


def input_summary(data: dict) -> pd.DataFrame:
    p = data["panel"]
    return pd.DataFrame([
        {"item": "panel rows",    "value": f"{len(p):,}"},
        {"item": "tickers",       "value": f"{p['ticker'].nunique():,}"},
        {"item": "date range",    "value": f"{p['date'].min().date()} → {p['date'].max().date()}"},
        {"item": "CPD target",    "value": TARGET_COL},
        {"item": "known events",  "value": str(len(data["known_events"]))},
    ])


# ---------------------------------------------------------------------------
# Score utilities
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def robust_zscore(x, window: int = 60, min_periods: int = 30) -> np.ndarray:
    s  = pd.Series(np.asarray(x, dtype=float))
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((s - mu) / sd.replace(0, np.nan)).to_numpy()


def clean_series(x) -> np.ndarray:
    v = np.asarray(x, dtype=float)
    return np.where(np.isfinite(v), v, np.nan)


def fill_nans(score, fill: float = 0.0) -> np.ndarray:
    return pd.Series(score).replace([np.inf, -np.inf], np.nan).fillna(fill).to_numpy()


# ---------------------------------------------------------------------------
# CPD algorithms — Fast
# ---------------------------------------------------------------------------

def adaptive_cusum_score(x,
                         z_window: int  = 60,
                         vol_window: int = 60,
                         k: float        = 0.25,
                         h_base: float   = 4.0) -> np.ndarray:
    """CUSUM on rolling z-scores with vol-adaptive threshold."""
    z         = fill_nans(robust_zscore(x, window=z_window))
    local_vol = pd.Series(clean_series(x)).rolling(vol_window, min_periods=30).std(ddof=0)
    vol_ref   = local_vol.rolling(252, min_periods=60).median()
    h_t       = (h_base * (local_vol / vol_ref)
                 .replace([np.inf, -np.inf], np.nan)
                 .clip(0.7, 1.8)).fillna(h_base).to_numpy()
    g_pos = np.zeros(len(z))
    g_neg = np.zeros(len(z))
    for i in range(1, len(z)):
        g_pos[i] = max(0.0, g_pos[i - 1] + z[i] - k)
        g_neg[i] = max(0.0, g_neg[i - 1] - z[i] - k)
    return np.clip(sigmoid(np.maximum(g_pos, g_neg) - h_t), 0, 1)


def rolling_ttest_score(x,
                        short_window: int = 5,
                        long_window: int  = 30,
                        threshold: float  = 3.0,
                        scale: float      = 1.0) -> np.ndarray:
    """Welch t-statistic between short and long window means."""
    s     = pd.Series(clean_series(x)).shift(1)
    short = s.rolling(short_window, min_periods=max(3, short_window - 1))
    long  = s.rolling(long_window,  min_periods=max(10, long_window - 5))
    denom = np.sqrt(
        (short.var(ddof=1) / short.count()) +
        (long.var(ddof=1)  / long.count())
    )
    stat = ((short.mean() - long.mean()).abs() / denom.replace(0, np.nan)).to_numpy()
    return np.clip(sigmoid((fill_nans(stat) - threshold) / scale), 0, 1)


# ---------------------------------------------------------------------------
# CPD algorithms — Heavy
# ---------------------------------------------------------------------------

def bocpd_score(x,
                hazard: float         = 1 / 250,
                max_run_length: int   = 500,
                min_var: float        = 1e-6) -> np.ndarray:
    """Bayesian online CPD — posterior probability of a new changepoint."""
    y      = clean_series(x)
    score  = np.zeros(len(y))
    finite = y[np.isfinite(y)]
    init_v = np.nanvar(finite) if len(finite) > 5 else min_var

    run_probs = np.array([1.0])
    means     = np.array([0.0])
    counts    = np.array([0.0])
    vars_     = np.array([max(init_v, min_var)])

    for t, obs in enumerate(y):
        if not np.isfinite(obs):
            continue
        pred_var = np.maximum(vars_ * (1 + 1 / np.maximum(counts, 1)), min_var)
        pred     = (np.exp(-0.5 * ((obs - means) ** 2) / pred_var)
                    / np.sqrt(2 * np.pi * pred_var))
        growth   = run_probs * (1 - hazard) * pred
        cp_prob  = np.sum(run_probs * hazard * pred)
        new_p    = np.r_[cp_prob, growth]
        norm     = new_p.sum()
        new_p    = new_p / norm if (norm > 0 and np.isfinite(norm)) else np.array([1.0])
        score[t] = new_p[0]

        nc  = np.r_[0.0, counts + 1]
        nm  = np.r_[obs, (means * counts + obs) / np.maximum(counts + 1, 1)]
        uv  = ((counts * vars_ + (obs - means) * (obs - nm[1:]))
               / np.maximum(counts + 1, 1))
        nv  = np.maximum(np.nan_to_num(
            np.r_[np.nanvar(y[max(0, t - 60):t + 1]), uv], nan=min_var), min_var)

        if len(new_p) > max_run_length:
            new_p = new_p[:max_run_length] / new_p[:max_run_length].sum()
            nc, nm, nv = nc[:max_run_length], nm[:max_run_length], nv[:max_run_length]
        run_probs, counts, means, vars_ = new_p, nc, nm, nv

    return np.clip(score, 0, 1)


def _matern32(t: np.ndarray, ls: float = 10.0, sf: float = 1.0) -> np.ndarray:
    r = np.abs(t[:, None] - t[None, :])
    a = np.sqrt(3) * r / ls
    return sf ** 2 * (1 + a) * np.exp(-a)


def _gp_lml(y: np.ndarray, ls: float = 10.0, noise: float = 0.25) -> float:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 8 or np.nanstd(y) == 0:
        return np.nan
    y = (y - np.nanmean(y)) / np.nanstd(y)
    t = np.arange(len(y), dtype=float)
    K = _matern32(t, ls=ls) + (noise ** 2 + 1e-6) * np.eye(len(y))
    try:
        L     = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return (-0.5 * y @ alpha
                - np.sum(np.log(np.diag(L)))
                - 0.5 * len(y) * np.log(2 * np.pi))
    except np.linalg.LinAlgError:
        return np.nan


def gp_matern_score(x,
                    window: int    = 40,
                    min_side: int  = 12,
                    ls: float      = 10.0,
                    noise: float   = 0.25,
                    step: int      = 5,       # stride=5 for speed
                    llr_scale: float = 5.0) -> np.ndarray:
    """GP Matern 3/2 log-marginal-likelihood ratio (Bayes factor approximation)."""
    y     = clean_series(x)
    score = np.full(len(y), np.nan)
    half  = window // 2
    for i in np.arange(half, len(y) - half, step):
        seg = y[i - half: i + half]
        if np.isfinite(seg).sum() < window * 0.8:
            continue
        left, right = seg[:half], seg[half:]
        if np.isfinite(left).sum() < min_side or np.isfinite(right).sum() < min_side:
            continue
        ll_f = _gp_lml(seg,   ls=ls, noise=noise)
        ll_l = _gp_lml(left,  ls=ls, noise=noise)
        ll_r = _gp_lml(right, ls=ls, noise=noise)
        if np.isfinite(ll_f) and np.isfinite(ll_l) and np.isfinite(ll_r):
            score[i] = sigmoid(((ll_l + ll_r) - ll_f) / llr_scale)
    score = (pd.Series(score).interpolate(limit_direction="both")
             .fillna(0.0).to_numpy())
    return np.clip(score, 0, 1)


# ---------------------------------------------------------------------------
# Method registries
# ---------------------------------------------------------------------------

FAST_METHODS: dict[str, callable] = {
    "adaptive_cusum": adaptive_cusum_score,
    "rolling_ttest":  rolling_ttest_score,
}

HEAVY_METHODS: dict[str, callable] = {
    "bocpd":       bocpd_score,
    "gp_matern32": gp_matern_score,
}

ALL_METHODS = {**FAST_METHODS, **HEAVY_METHODS}


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

def eligible_tickers(panel: pd.DataFrame, min_obs: int = MIN_OBS_SCORE) -> list[str]:
    counts = panel.groupby("ticker")[TARGET_COL].apply(lambda s: s.notna().sum())
    return counts.loc[counts.ge(min_obs)].index.tolist()


def sample_heavy_tickers(panel: pd.DataFrame,
                         eligible: list[str],
                         sample_size: int = HEAVY_SAMPLE_SIZE,
                         required_ticker: str | None = None) -> list[str]:
    if not eligible:
        return []
    stats = (panel.loc[panel["ticker"].isin(eligible)]
             .groupby("ticker").agg(n=("date", "size")))
    try:
        stats["bucket"] = pd.qcut(stats["n"], q=5, duplicates="drop")
        sample = (stats.groupby("bucket", observed=True, group_keys=False)
                  .sample(frac=1, random_state=RANDOM_SEED)
                  .head(sample_size).index.tolist())
    except (ValueError, IndexError):
        sample = stats.sample(min(sample_size, len(stats)),
                              random_state=RANDOM_SEED).index.tolist()
    if required_ticker in eligible and required_ticker not in sample:
        sample[-1] = required_ticker
    return sorted(set(sample))


def select_example_ticker(panel: pd.DataFrame, preferred: str = "ASML") -> str:
    cands = sorted(panel.loc[panel["ticker"].str.contains(preferred, case=False, na=False),
                              "ticker"].unique())
    return cands[0] if cands else str(panel["ticker"].iloc[0])


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _compute_scores_for_methods(panel_df: pd.DataFrame,
                                 methods: dict,
                                 tickers: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    src  = panel_df if tickers is None else panel_df.loc[panel_df["ticker"].isin(tickers)]
    data = src[["date", "ticker", TARGET_COL]].sort_values(["ticker", "date"])
    rows, logs = [], []
    for ticker, grp in data.groupby("ticker", sort=False):
        y = grp[TARGET_COL].to_numpy()
        if np.isfinite(y).sum() < MIN_OBS_SCORE:
            continue
        for method, func in methods.items():
            t0    = time.perf_counter()
            score = fill_nans(func(y))
            elapsed = time.perf_counter() - t0
            rows.append(pd.DataFrame({
                "date":   pd.to_datetime(grp["date"].to_numpy()),
                "ticker": ticker,
                "method": method,
                "score":  np.clip(score, 0, 1),
            }))
            logs.append({"ticker": ticker, "method": method, "seconds": elapsed})

    scores = (pd.concat(rows, ignore_index=True) if rows
              else pd.DataFrame(columns=["date", "ticker", "method", "score"]))
    timing = pd.DataFrame(logs, columns=["ticker", "method", "seconds"])
    return scores, timing


def build_cpd_scores(panel: pd.DataFrame,
                     example_ticker: str | None = None) -> dict:
    """Run all 4 methods. Fast on full universe, heavy on 30-stock sample."""
    eligible     = eligible_tickers(panel)
    heavy_sample = sample_heavy_tickers(panel, eligible,
                                        required_ticker=example_ticker)

    fast_scores,  fast_timing  = _compute_scores_for_methods(panel, FAST_METHODS)
    heavy_scores, heavy_timing = _compute_scores_for_methods(panel, HEAVY_METHODS,
                                                              tickers=heavy_sample)

    cpd_scores = (pd.concat([fast_scores, heavy_scores], ignore_index=True)
                  .assign(method=lambda x: x["method"].astype("category"))
                  .sort_values(["method", "ticker", "date"])
                  .reset_index(drop=True))
    cpd_timing = pd.concat([fast_timing, heavy_timing], ignore_index=True)

    return {
        "cpd_scores":   cpd_scores,
        "cpd_timing":   cpd_timing,
        "eligible":     eligible,
        "heavy_sample": heavy_sample,
    }


def score_summary(cpd_scores: pd.DataFrame) -> pd.DataFrame:
    return (cpd_scores.groupby("method", observed=True)
            .agg(tickers=("ticker", "nunique"),
                 rows=("score", "size"),
                 mean=("score", "mean"),
                 std=("score", "std"))
            .round(4).reset_index())


def timing_table(cpd_timing: pd.DataFrame) -> pd.DataFrame:
    return (cpd_timing.groupby("method", observed=True)
            .agg(stocks=("ticker", "nunique"),
                 mean_sec_per_stock=("seconds", "mean"),
                 total_sec=("seconds", "sum"))
            .round(3).sort_values("mean_sec_per_stock").reset_index())


# ---------------------------------------------------------------------------
# ν and γ extraction
# ---------------------------------------------------------------------------

def extract_nu_gamma(scores: np.ndarray,
                     threshold: float = 0.5,
                     window: int = NU_GAMMA_WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """
    nu    = score (severity, already in [0,1])
    gamma = recency of last detected changepoint:
            1.0 when score >= threshold, decays by 1/window per day afterward.
            Represents how recently a regime change was detected.
    """
    nu    = np.clip(scores, 0, 1)
    gamma = np.zeros(len(nu))
    for t in range(1, len(nu)):
        if nu[t] >= threshold:
            gamma[t] = 1.0
        else:
            gamma[t] = max(0.0, gamma[t - 1] - 1.0 / window)
    return nu, gamma


def build_nb03_features(cpd_scores: pd.DataFrame,
                        threshold: float = 0.5) -> pd.DataFrame:
    """Build wide (date, ticker) feature table with nu/gamma for each method, lagged 1 day."""
    parts = []
    for method, grp in cpd_scores.groupby("method", observed=True):
        method = str(method)
        for ticker, sub in grp.groupby("ticker", observed=True):
            sub   = sub.sort_values("date").reset_index(drop=True)
            nu, gamma = extract_nu_gamma(sub["score"].to_numpy(), threshold=threshold)
            parts.append(pd.DataFrame({
                "date":                    sub["date"],
                "ticker":                  ticker,
                f"nu_{method}":            nu,
                f"gamma_{method}":         gamma,
                f"nu_{method}_lag1":       pd.Series(nu).shift(1).to_numpy(),
                f"gamma_{method}_lag1":    pd.Series(gamma).shift(1).to_numpy(),
            }))

    if not parts:
        return pd.DataFrame()

    wide = parts[0]
    for df in parts[1:]:
        wide = wide.merge(df, on=["date", "ticker"], how="outer")

    return wide.sort_values(["date", "ticker"]).reset_index(drop=True)


def nu_gamma_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in features.columns if c.startswith("nu_") or c.startswith("gamma_")]:
        s = features[col].dropna()
        rows.append({
            "feature":    col,
            "non_null_%": round(features[col].notna().mean() * 100, 1),
            "mean":       round(s.mean(), 4),
            "std":        round(s.std(),  4),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Evaluation vs known events
# ---------------------------------------------------------------------------

def _event_windows(events: pd.DataFrame) -> pd.DataFrame:
    return events.assign(
        start=lambda x: x["event_date"] - pd.Timedelta(days=EVENT_PRE_DAYS),
        end=lambda x:   x["event_date"] + pd.Timedelta(days=EVENT_POST_DAYS),
    )


def build_daily_scores(cpd_scores: pd.DataFrame,
                       known_events: pd.DataFrame,
                       q: float = 0.90) -> pd.DataFrame:
    """Cross-sectional q-th percentile score per (method, date), with event flags."""
    daily = (cpd_scores.groupby(["method", "date"], observed=True)["score"]
             .quantile(q).reset_index(name="daily_score"))
    wins  = _event_windows(known_events)
    dates = daily["date"].drop_duplicates()
    is_ev = pd.Series(False, index=dates)
    for row in wins.itertuples(index=False):
        is_ev |= dates.between(row.start, row.end)
    ev_map = is_ev.reset_index().rename(columns={"index": "date", 0: "is_event"})
    return daily.merge(ev_map, on="date", how="left").fillna({"is_event": False})


def compute_cpd_metrics(daily_scores: pd.DataFrame,
                        cpd_timing:   pd.DataFrame,
                        q: float = DETECTION_QUANTILE) -> pd.DataFrame:
    rows = []
    for method, grp in daily_scores.groupby("method", observed=True):
        thr  = grp["daily_score"].quantile(q)
        pred = grp["daily_score"].ge(thr)
        ev   = grp["is_event"].astype(bool)
        tp = int((pred & ev).sum())
        fp = int((pred & ~ev).sum())
        tn = int((~pred & ~ev).sum())
        fn = int((~pred & ev).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fpr  = fp / (fp + tn) if fp + tn else 0.0
        rows.append({"method": str(method), "precision": prec, "recall": rec,
                     "fpr": fpr, "f1": f1, "tp": tp, "fp": fp, "fn": fn})

    metrics = pd.DataFrame(rows)
    timing  = (cpd_timing.groupby("method")
               .agg(mean_sec_per_stock=("seconds", "mean")).reset_index())
    return metrics.merge(timing, on="method", how="left").round(3)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_events_timeline(benchmark_ew: pd.DataFrame,
                         known_events: pd.DataFrame) -> go.Figure:
    """SXXR cumulative return with known macro events as colored vertical lines."""
    bm = benchmark_ew.dropna(subset=["sxxr_1d_ret"]).sort_values("date").copy()
    bm["cum"] = (1 + bm["sxxr_1d_ret"].fillna(0)).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bm["date"], y=bm["cum"],
        mode="lines", name="SXXR",
        line=dict(color="black", width=1.5),
    ))

    shown_types: set[str] = set()
    for ev in known_events.sort_values("event_date").itertuples(index=False):
        ev_type = ev.event_type
        color   = EVENT_TYPE_COLORS.get(ev_type, "#888888")
        show    = ev_type not in shown_types
        shown_types.add(ev_type)
        fig.add_vline(
            x=ev.event_date,
            line=dict(color=color, width=1, dash="dot"),
        )
        fig.add_trace(go.Scatter(
            x=[ev.event_date], y=[bm["cum"].min()],
            mode="markers", name=ev_type,
            marker=dict(color=color, size=6),
            legendgroup=ev_type,
            showlegend=show,
            hovertext=ev.event,
            hoverinfo="text+x",
        ))

    fig.update_layout(
        title="Known macro events — SXXR total return 2006–2026",
        template="plotly_white",
        height=420,
        yaxis_title="Cumulative return (base 1)",
        legend=dict(orientation="h", y=-0.25, x=0),
        margin=dict(l=50, r=20, t=50, b=20),
    )
    return fig


def plot_method_on_ticker(ticker_panel:  pd.DataFrame,
                          score:         np.ndarray,
                          method:        str,
                          known_events:  pd.DataFrame,
                          threshold:     float = 0.5) -> go.Figure:
    """Price + CPD score for one ticker and one method.

    Green markers = events detected (score >= threshold in window).
    Red markers   = events missed.
    """
    tp = ticker_panel.sort_values("date").reset_index(drop=True)
    dates = tp["date"].to_numpy()

    wins = _event_windows(known_events)
    detected, missed = [], []
    for row in wins.itertuples(index=False):
        mask = (tp["date"] >= row.start) & (tp["date"] <= row.end)
        if mask.any() and score[mask].max() >= threshold:
            detected.append(row.event_date)
        elif mask.any():
            missed.append(row.event_date)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.4, 0.6], vertical_spacing=0.04,
                        subplot_titles=[f"{tp['ticker'].iloc[0]} — price",
                                        f"{method} score"])

    # Price
    fig.add_trace(go.Scatter(x=tp["date"], y=tp["price"],
                             mode="lines", line=dict(color="black", width=1.2),
                             showlegend=False), row=1, col=1)

    # Score
    color = METHOD_COLORS.get(method, "#444")
    fig.add_trace(go.Scatter(x=dates, y=score, mode="lines",
                             line=dict(color=color, width=1.2),
                             name=method, showlegend=False), row=2, col=1)

    # Threshold line
    fig.add_hline(y=threshold, line=dict(color="gray", width=1, dash="dash"), row=2, col=1)

    # Event verticals on both panels
    for ev_date in known_events["event_date"]:
        if dates.min() <= ev_date <= dates.max():
            fig.add_vline(x=ev_date, line=dict(color="rgba(200,0,0,0.18)", width=1, dash="dot"))

    # Detected (green) / Missed (red) markers on score panel
    for ev_date in detected:
        idx = np.argmin(np.abs(dates - np.datetime64(ev_date)))
        fig.add_trace(go.Scatter(x=[dates[idx]], y=[score[idx]], mode="markers",
                                 marker=dict(color="green", size=9, symbol="circle"),
                                 name="detected", showlegend=False,
                                 hovertext="detected"), row=2, col=1)
    for ev_date in missed:
        idx = np.argmin(np.abs(dates - np.datetime64(ev_date)))
        fig.add_trace(go.Scatter(x=[dates[idx]], y=[score[idx]], mode="markers",
                                 marker=dict(color="red", size=9, symbol="x"),
                                 name="missed", showlegend=False,
                                 hovertext="missed"), row=2, col=1)

    fig.update_yaxes(range=[0, 1.05], row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=50, r=20, t=50, b=20),
        hovermode="x unified",
    )
    return fig


def plot_fpr_recall(metrics: pd.DataFrame) -> go.Figure:
    """Scatter FPR (x) vs Recall (y) — one point per method.

    Bubble size ∝ computation time.  Star = ideal point (0, 1).
    """
    fig = go.Figure()

    # Ideal point
    fig.add_trace(go.Scatter(
        x=[0], y=[1], mode="markers+text",
        marker=dict(symbol="star", size=18, color="gold",
                    line=dict(color="black", width=1)),
        text=["ideal"], textposition="top right",
        name="ideal", showlegend=False,
    ))

    # Methods
    max_sec = max(metrics["mean_sec_per_stock"].max(), 1e-9)
    for _, row in metrics.iterrows():
        sz    = 14 + 30 * (row["mean_sec_per_stock"] / max_sec)
        color = METHOD_COLORS.get(row["method"], "#888")
        fig.add_trace(go.Scatter(
            x=[row["fpr"]], y=[row["recall"]],
            mode="markers+text",
            marker=dict(size=sz, color=color,
                        line=dict(color="white", width=1.5)),
            text=[row["method"]],
            textposition="top center",
            name=row["method"],
            hovertemplate=(
                f"<b>{row['method']}</b><br>"
                f"Recall : {row['recall']:.3f}<br>"
                f"FPR    : {row['fpr']:.3f}<br>"
                f"F1     : {row['f1']:.3f}<br>"
                f"Speed  : {row['mean_sec_per_stock']:.3f}s/stock"
                "<extra></extra>"
            ),
        ))

    # Arrow toward ideal
    fig.add_annotation(
        x=0.05, y=0.9, ax=0.35, ay=0.5,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.3,
        arrowcolor="rgba(0,150,0,0.5)", arrowwidth=2,
        text="better →", font=dict(color="rgba(0,120,0,0.8)", size=11),
    )

    fig.update_layout(
        title=("FPR vs Recall — method comparison<br>"
               "<sup>Bubble size ∝ computation time · Star = ideal</sup>"),
        xaxis=dict(title="False Positive Rate", range=[-0.05, 1.05]),
        yaxis=dict(title="Recall",              range=[-0.05, 1.15]),
        template="plotly_white",
        height=520, width=750,
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_nb02_outputs(root,
                      cpd_scores:   pd.DataFrame,
                      cpd_metrics:  pd.DataFrame,
                      nb03_features: pd.DataFrame) -> pd.DataFrame:
    paths = nb02_paths(root)
    paths["processed_dir"].mkdir(parents=True, exist_ok=True)

    (cpd_scores.assign(score=lambda x: x["score"].astype("float32"))
     .to_parquet(paths["cpd_scores"],   index=False))
    cpd_metrics.to_parquet(paths["cpd_metrics"],  index=False)
    nb03_features.to_parquet(paths["cpd_features"], index=False)

    return pd.DataFrame([
        {"output": "cpd_scores",        "rows": f"{len(cpd_scores):,}",
         "path": paths["cpd_scores"].relative_to(paths["project_root"]).as_posix()},
        {"output": "cpd_metrics",       "rows": f"{len(cpd_metrics):,}",
         "path": paths["cpd_metrics"].relative_to(paths["project_root"]).as_posix()},
        {"output": "cpd_features_nb03", "rows": f"{len(nb03_features):,}",
         "path": paths["cpd_features"].relative_to(paths["project_root"]).as_posix()},
    ])
