"""NB02 helpers: changepoint detection on STOXX 600 stocks.

Methods
-------
OFFLINE (upper-bound baseline):
  binary_segmentation  — sees full series, informational ceiling for recall

FAST (full universe):
  cusum_sigmoid        — two-sided CUSUM on rolling z-scores, sigmoid output
  adaptive_cusum       — CUSUM with vol-adjusted threshold
  jump_process         — large-return detector
  rolling_ttest        — Welch t-stat between short/long windows
  ma_crossover         — MACD-style crossover normalised by vol

HEAVY (stratified sample, ~30 stocks):
  bocpd                — Bayesian online CPD, posterior run-length mass
  gp_matern32          — GP marginal likelihood ratio, Matern 3/2 kernel

All scores are causal (no lookahead) and bounded in [0, 1].
"""

from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Callable

import numpy as np
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.known_events import known_events_frame


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVENT_WINDOW_DAYS = 5
LATENCY_WINDOW_DAYS = 10
DETECTION_QUANTILE = 0.95
ROBUSTNESS_QUANTILES = [0.90, 0.95, 0.99]
DAILY_SCORE_Q = 0.90
RANDOM_SEED = 42
HEAVY_SAMPLE_SIZE = 30
MIN_OBS_SCORE = 252
SELECTED_METHOD = "cusum_sigmoid"

COLOR_TARGET = {"raw": "#1f77b4", "vs_market": "#ff7f0e", "vs_sector": "#2ca02c"}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def resolve_project_root(start=None) -> Path:
    root = Path.cwd() if start is None else Path(start)
    root = root.resolve()
    if (root / "configs" / "default.yaml").exists():
        return root
    if (root.parent / "configs" / "default.yaml").exists():
        return root.parent
    raise FileNotFoundError("Could not find configs/default.yaml")


def nb02_paths(root=None) -> dict[str, Path]:
    project_root = resolve_project_root(root)
    processed = project_root / "data" / "processed" / "stoxx600"
    return {
        "project_root": project_root,
        "processed_dir": processed,
        "panel": processed / "panel.parquet",
        "universe": processed / "universe_pit.parquet",
        "benchmark_ew": processed / "benchmark_ew.parquet",
        "cpd_scores": processed / "cpd_scores.parquet",
        "cpd_metrics": processed / "cpd_metrics.parquet",
        "cpd_features": processed / "cpd_features_nb03.parquet",
    }


def load_nb02_inputs(root=None) -> dict:
    paths = nb02_paths(root)
    missing = [p for p in [paths["panel"], paths["universe"], paths["benchmark_ew"]]
               if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing NB01 outputs: {[str(p) for p in missing]}")

    panel = pd.read_parquet(paths["panel"])
    universe_pit = pd.read_parquet(paths["universe"])
    benchmark_ew = pd.read_parquet(paths["benchmark_ew"])
    for frame in [panel, universe_pit, benchmark_ew]:
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])

    events = known_events_frame(panel["date"].min(), panel["date"].max())
    return {
        "paths": paths,
        "panel": panel,
        "universe_pit": universe_pit,
        "benchmark_ew": benchmark_ew,
        "target_cols": target_columns(panel),
        "known_events": events,
    }


def target_columns(panel: pd.DataFrame) -> dict[str, str]:
    targets = {"raw": "1d_log_ret", "vs_market": "1d_ret_vs_market"}
    if "1d_ret_vs_sector" in panel.columns:
        targets["vs_sector"] = "1d_ret_vs_sector"
    return {k: v for k, v in targets.items() if v in panel.columns}


def input_summary(data: dict) -> pd.DataFrame:
    panel = data["panel"]
    return pd.DataFrame([
        {"item": "panel rows", "value": f"{len(panel):,}"},
        {"item": "tickers", "value": f"{panel['ticker'].nunique():,}"},
        {"item": "date range", "value": f"{panel['date'].min().date()} → {panel['date'].max().date()}"},
        {"item": "targets", "value": ", ".join(data["target_cols"].keys())},
        {"item": "universe type", "value": "point-in-time (PIT)"},
    ])


# ---------------------------------------------------------------------------
# Score utilities
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def robust_zscore(x, window: int = 60, min_periods: int = 30) -> np.ndarray:
    s = pd.Series(x, dtype=float)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    return ((s - mu) / sd.replace(0, np.nan)).to_numpy()


def clean_series(x) -> np.ndarray:
    v = np.asarray(x, dtype=float)
    return np.where(np.isfinite(v), v, np.nan)


def fill_score_nans(score, fill_value: float = 0.0) -> np.ndarray:
    return pd.Series(score).replace([np.inf, -np.inf], np.nan).fillna(fill_value).to_numpy()


def score_frame(dates, ticker, method, target, score) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "ticker": ticker,
        "method": method, "target": target,
        "score": np.clip(score, 0, 1),
    })


# ---------------------------------------------------------------------------
# CPD methods — OFFLINE
# ---------------------------------------------------------------------------

def binary_segmentation_score(x, n_bkps: int = 10, model: str = "rbf"):
    """Offline binary segmentation (ruptures). Sees full series — upper-bound baseline."""
    y = clean_series(x)
    score = np.zeros(len(y))
    finite_y = y[np.isfinite(y)]
    if len(finite_y) < 20:
        return score
    try:
        import ruptures as rpt
        signal = np.nan_to_num(y, nan=0.0).reshape(-1, 1)
        algo = rpt.Binseg(model=model).fit(signal)
        bkps = algo.predict(n_bkps=min(n_bkps, len(y) // 20))
        bkps = [b for b in bkps if b < len(y)]
        for b in bkps:
            lo, hi = max(0, b - 2), min(len(y), b + 3)
            score[lo:hi] = 1.0
        # Smooth into a score rather than binary flag
        score = pd.Series(score).rolling(5, center=True, min_periods=1).mean().to_numpy()
    except ImportError:
        # Fallback: simple variance-change detector
        w = 20
        for i in range(w, len(y) - w):
            if np.isfinite(y[i - w:i]).sum() >= w // 2 and np.isfinite(y[i:i + w]).sum() >= w // 2:
                var_l = np.nanvar(y[i - w:i])
                var_r = np.nanvar(y[i:i + w])
                score[i] = sigmoid(abs(var_r - var_l) / (max(var_l, var_r, 1e-8)))
    return np.clip(score, 0, 1)


# ---------------------------------------------------------------------------
# CPD methods — FAST
# ---------------------------------------------------------------------------

def cumsum_sigmoid_score(x, z_window: int = 60, k: float = 0.25, h: float = 5.0, scale: float = 1.0):
    """Two-sided CUSUM on rolling z-scores, mapped to a continuous [0,1] score."""
    z = fill_score_nans(robust_zscore(x, window=z_window))
    g_pos = np.zeros(len(z))
    g_neg = np.zeros(len(z))
    for i in range(1, len(z)):
        g_pos[i] = max(0.0, g_pos[i - 1] + z[i] - k)
        g_neg[i] = max(0.0, g_neg[i - 1] - z[i] - k)
    return sigmoid((np.maximum(g_pos, g_neg) - h) / scale)


def adaptive_cusum_score(x, z_window: int = 60, vol_window: int = 60, k: float = 0.25, h_base: float = 4.0):
    """CUSUM with vol-adjusted threshold."""
    z = fill_score_nans(robust_zscore(x, window=z_window))
    local_vol = pd.Series(clean_series(x), dtype=float).rolling(vol_window, min_periods=30).std(ddof=0)
    vol_ref = local_vol.rolling(252, min_periods=60).median()
    h_t = (h_base * (local_vol / vol_ref).replace([np.inf, -np.inf], np.nan)
           .clip(0.7, 1.8)).fillna(h_base).to_numpy()
    g_pos = np.zeros(len(z))
    g_neg = np.zeros(len(z))
    for i in range(1, len(z)):
        g_pos[i] = max(0.0, g_pos[i - 1] + z[i] - k)
        g_neg[i] = max(0.0, g_neg[i - 1] - z[i] - k)
    return sigmoid(np.maximum(g_pos, g_neg) - h_t)


def jump_process_score(x, z_window: int = 60, threshold: float = 3.0, scale: float = 1.0):
    z = np.abs(robust_zscore(x, window=z_window))
    return sigmoid((fill_score_nans(z) - threshold) / scale)


def rolling_ttest_score(x, short_window: int = 5, long_window: int = 30,
                        min_short: int = 4, min_long: int = 20,
                        threshold: float = 3.0, scale: float = 1.0):
    s = pd.Series(clean_series(x), dtype=float).shift(1)
    short = s.rolling(short_window, min_periods=min_short)
    long = s.rolling(long_window, min_periods=min_long)
    denom = np.sqrt((short.var(ddof=1) / short.count()) + (long.var(ddof=1) / long.count()))
    stat = ((short.mean() - long.mean()).abs() / denom.replace(0, np.nan)).to_numpy()
    return sigmoid((fill_score_nans(stat) - threshold) / scale)


def ma_crossover_score(x, fast: int = 21, slow: int = 63, scale: float = 1.0):
    s = pd.Series(clean_series(x), dtype=float)
    equity = s.fillna(0.0).cumsum()
    ma_fast = equity.rolling(fast, min_periods=max(5, fast // 4)).mean()
    ma_slow = equity.rolling(slow, min_periods=max(5, slow // 4)).mean()
    sigma = s.rolling(slow, min_periods=max(5, slow // 4)).std(ddof=1).replace(0, np.nan)
    return fill_score_nans(sigmoid((ma_fast - ma_slow).abs() / sigma / scale - 1.0).to_numpy())


# ---------------------------------------------------------------------------
# CPD methods — HEAVY
# ---------------------------------------------------------------------------

def bocpd_score(x, hazard: float = 1 / 250, max_run_length: int = 500, min_var: float = 1e-6):
    """Bayesian online CPD — posterior probability of a new changepoint."""
    y = clean_series(x)
    score = np.zeros(len(y))
    finite = y[np.isfinite(y)]
    init_var = np.nanvar(finite) if len(finite) > 5 else min_var
    run_probs = np.array([1.0])
    means = np.array([0.0])
    counts = np.array([0.0])
    vars_ = np.array([max(init_var, min_var)])

    for t, obs in enumerate(y):
        if not np.isfinite(obs):
            score[t] = 0.0
            continue
        pred_var = np.maximum(vars_ * (1 + 1 / np.maximum(counts, 1)), min_var)
        pred = np.exp(-0.5 * ((obs - means) ** 2) / pred_var) / np.sqrt(2 * np.pi * pred_var)
        growth = run_probs * (1 - hazard) * pred
        cp_prob = np.sum(run_probs * hazard * pred)
        new_probs = np.r_[cp_prob, growth]
        norm = new_probs.sum()
        if norm <= 0 or not np.isfinite(norm):
            new_probs = np.array([1.0])
        else:
            new_probs = new_probs / norm
        score[t] = new_probs[0]
        new_counts = np.r_[0.0, counts + 1]
        new_means = np.r_[obs, (means * counts + obs) / np.maximum(counts + 1, 1)]
        upd_vars = (counts * vars_ + (obs - means) * (obs - new_means[1:])) / np.maximum(counts + 1, 1)
        new_vars = np.r_[np.nanvar(y[max(0, t - 60):t + 1]), upd_vars]
        new_vars = np.maximum(np.nan_to_num(new_vars, nan=min_var), min_var)
        if len(new_probs) > max_run_length:
            new_probs = new_probs[:max_run_length] / new_probs[:max_run_length].sum()
            new_counts = new_counts[:max_run_length]
            new_means = new_means[:max_run_length]
            new_vars = new_vars[:max_run_length]
        run_probs, counts, means, vars_ = new_probs, new_counts, new_means, new_vars
    return np.clip(score, 0, 1)


def _matern32(t, ls: float = 10.0, sf: float = 1.0):
    r = np.abs(t[:, None] - t[None, :])
    a = np.sqrt(3) * r / ls
    return sf ** 2 * (1 + a) * np.exp(-a)


def _gp_lml(y, ls: float = 10.0, noise: float = 0.25):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 8 or np.nanstd(y) == 0:
        return np.nan
    y = (y - np.nanmean(y)) / np.nanstd(y)
    t = np.arange(len(y), dtype=float)
    K = _matern32(t, ls=ls) + (noise ** 2 + 1e-6) * np.eye(len(y))
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(y) * np.log(2 * np.pi)
    except np.linalg.LinAlgError:
        return np.nan


def gp_matern_score(x, window: int = 40, min_side: int = 12, ls: float = 10.0,
                    noise: float = 0.25, step: int = 3, llr_scale: float = 5.0):
    """GP Matern 3/2 log-marginal-likelihood ratio — approximates a Bayes factor."""
    y = clean_series(x)
    score = np.full(len(y), np.nan)
    half = window // 2
    for i in np.arange(half, len(y) - half, step):
        seg = y[i - half:i + half]
        if np.isfinite(seg).sum() < window * 0.8:
            continue
        left, right = seg[:half], seg[half:]
        if np.isfinite(left).sum() < min_side or np.isfinite(right).sum() < min_side:
            continue
        ll_f = _gp_lml(seg, ls=ls, noise=noise)
        ll_l = _gp_lml(left, ls=ls, noise=noise)
        ll_r = _gp_lml(right, ls=ls, noise=noise)
        if np.isfinite(ll_f) and np.isfinite(ll_l) and np.isfinite(ll_r):
            score[i] = sigmoid(((ll_l + ll_r) - ll_f) / llr_scale)
    score = pd.Series(score).interpolate(limit_direction="both").fillna(0.0).to_numpy()
    return np.clip(score, 0, 1)


# ---------------------------------------------------------------------------
# Method registries
# ---------------------------------------------------------------------------

OFFLINE_METHODS: dict[str, Callable] = {
    "binary_segmentation": binary_segmentation_score,
}

FAST_METHODS: dict[str, Callable] = {
    "cusum_sigmoid": cumsum_sigmoid_score,
    "adaptive_cusum": adaptive_cusum_score,
    "jump_process": jump_process_score,
    "rolling_ttest": rolling_ttest_score,
    "ma_crossover": ma_crossover_score,
}

HEAVY_METHODS: dict[str, Callable] = {
    "bocpd": bocpd_score,
    "gp_matern32": gp_matern_score,
}


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

def select_example_ticker(panel: pd.DataFrame, preferred: str = "ASML") -> str:
    cands = sorted(panel.loc[panel["ticker"].str.contains(preferred, case=False, na=False),
                              "ticker"].unique())
    return cands[0] if cands else str(panel["ticker"].iloc[0])


def eligible_tickers(panel: pd.DataFrame, target_cols: dict, min_obs: int = MIN_OBS_SCORE) -> list[str]:
    counts = panel.groupby("ticker")[list(target_cols.values())].agg(lambda x: x.notna().sum())
    return counts.loc[counts.min(axis=1).ge(min_obs)].index.to_list()


def sample_heavy_tickers(panel, eligible, sample_size=HEAVY_SAMPLE_SIZE,
                         random_seed=RANDOM_SEED, required_ticker=None) -> list[str]:
    if not eligible:
        return []
    stats = panel.loc[panel["ticker"].isin(eligible)].groupby("ticker").agg(n_obs=("date", "size"))
    try:
        stats = stats.assign(bucket=pd.qcut(stats["n_obs"], q=5, duplicates="drop"))
        sample = (stats.groupby("bucket", observed=True, group_keys=False)
                  .sample(frac=1, random_state=random_seed)
                  .head(sample_size).index.to_list())
    except (ValueError, IndexError):
        sample = stats.sample(min(sample_size, len(stats)), random_state=random_seed).index.to_list()
    if required_ticker in eligible and required_ticker not in sample:
        if sample:
            sample[-1] = required_ticker
        else:
            sample = [required_ticker]
    return sorted(set(sample))


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_scores(panel_df, methods, targets, tickers=None, min_obs=MIN_OBS_SCORE):
    rows, logs = [], []
    src = panel_df if tickers is None else panel_df.loc[panel_df["ticker"].isin(tickers)]
    for target, col in targets.items():
        target_df = src[["date", "ticker", col]].sort_values(["ticker", "date"])
        for ticker, grp in target_df.groupby("ticker", sort=False):
            y = grp[col].to_numpy()
            if np.isfinite(y).sum() < min_obs:
                continue
            for method, func in methods.items():
                tic = time.perf_counter()
                score = fill_score_nans(func(y))
                elapsed = time.perf_counter() - tic
                rows.append(score_frame(grp["date"].to_numpy(), ticker, method, target, score))
                logs.append({"ticker": ticker, "target": target, "method": method, "seconds": elapsed})
    scores = (pd.concat(rows, ignore_index=True) if rows
              else pd.DataFrame(columns=["date", "ticker", "method", "target", "score"]))
    timing = pd.DataFrame(logs, columns=["ticker", "target", "method", "seconds"])
    return scores, timing


def build_cpd_scores(panel, target_cols, example_ticker=None,
                     random_seed=RANDOM_SEED, heavy_sample_size=HEAVY_SAMPLE_SIZE,
                     min_obs=MIN_OBS_SCORE, include_offline=False) -> dict:
    """Run all CPD methods. Offline (binary segmentation) is optional."""
    eligible = eligible_tickers(panel, target_cols, min_obs=min_obs)
    heavy_sample = sample_heavy_tickers(panel, eligible, sample_size=heavy_sample_size,
                                        random_seed=random_seed, required_ticker=example_ticker)

    fast_scores, fast_timing = compute_scores(panel, FAST_METHODS, target_cols, min_obs=min_obs)
    heavy_scores, heavy_timing = compute_scores(panel, HEAVY_METHODS, target_cols,
                                                tickers=heavy_sample, min_obs=min_obs)

    all_parts = [fast_scores, heavy_scores]
    all_timing = [fast_timing, heavy_timing]
    if include_offline:
        off_scores, off_timing = compute_scores(panel, OFFLINE_METHODS, target_cols,
                                                tickers=heavy_sample, min_obs=min_obs)
        all_parts.append(off_scores)
        all_timing.append(off_timing)

    cpd_scores = (
        pd.concat(all_parts, ignore_index=True)
        .dropna(subset=["score"])
        .assign(
            ticker=lambda x: x["ticker"].astype("category"),
            method=lambda x: x["method"].astype("category"),
            target=lambda x: x["target"].astype("category"),
            score=lambda x: x["score"].clip(0, 1),
        )
        .sort_values(["method", "target", "ticker", "date"])
        .reset_index(drop=True)
    )
    return {
        "cpd_scores": cpd_scores,
        "cpd_timing": pd.concat(all_timing, ignore_index=True),
        "eligible_tickers": eligible,
        "heavy_sample": heavy_sample,
    }


# ---------------------------------------------------------------------------
# Sector-level CPD aggregation
# ---------------------------------------------------------------------------

def sector_cpd_scores(cpd_scores: pd.DataFrame, panel: pd.DataFrame,
                      method: str = SELECTED_METHOD, detection_quantile: float = DETECTION_QUANTILE) -> pd.DataFrame:
    """Aggregate stock-level CPD scores to sector level (mean and flagging rate)."""
    if "sector" not in panel.columns:
        return pd.DataFrame()
    target = "vs_market" if "vs_market" in cpd_scores["target"].astype(str).unique() else "raw"
    sel = cpd_scores.loc[
        cpd_scores["method"].astype(str).eq(method) & cpd_scores["target"].astype(str).eq(target)
    ].copy()
    threshold = sel["score"].quantile(detection_quantile)
    sel["flag"] = sel["score"].ge(threshold)
    sel["year"] = pd.to_datetime(sel["date"]).dt.year
    sector_map = panel[["ticker", "sector"]].drop_duplicates()
    sel = sel.merge(sector_map, on="ticker", how="left").dropna(subset=["sector"])
    return (sel.groupby(["sector", "year"])
            .agg(mean_score=("score", "mean"), flagging_rate=("flag", "mean"))
            .mul({"mean_score": 1, "flagging_rate": 100}).round(3)
            .reset_index())


# ---------------------------------------------------------------------------
# Daily scores & metrics
# ---------------------------------------------------------------------------

def event_windows(known_events: pd.DataFrame) -> pd.DataFrame:
    return known_events.assign(
        start=lambda x: x["event_date"] - pd.Timedelta(days=EVENT_WINDOW_DAYS),
        end=lambda x: x["event_date"] + pd.Timedelta(days=LATENCY_WINDOW_DAYS),
    )


def build_daily_scores(cpd_scores, benchmark_ew, known_events, daily_score_q=DAILY_SCORE_Q):
    date_labels = (benchmark_ew[["date"]].drop_duplicates().sort_values("date")
                   .assign(is_event_window=False))
    for row in event_windows(known_events).itertuples(index=False):
        mask = date_labels["date"].between(row.start, row.end)
        date_labels.loc[mask, "is_event_window"] = True
    return (cpd_scores.groupby(["method", "target", "date"], observed=True)["score"]
            .quantile(daily_score_q).reset_index(name="daily_score")
            .merge(date_labels, on="date", how="left")
            .assign(is_event_window=lambda x: x["is_event_window"].fillna(False).astype(bool)))


def compute_cpd_metrics(daily_scores, cpd_timing, detection_quantile=DETECTION_QUANTILE):
    rows = []
    for (method, target), grp in daily_scores.groupby(["method", "target"], observed=True):
        grp = grp.sort_values("date").copy()
        threshold = grp["daily_score"].quantile(detection_quantile)
        grp["pred"] = grp["daily_score"].ge(threshold)
        tp = int((grp["pred"] & grp["is_event_window"]).sum())
        fp = int((grp["pred"] & ~grp["is_event_window"]).sum())
        tn = int((~grp["pred"] & ~grp["is_event_window"]).sum())
        fn = int((~grp["pred"] & grp["is_event_window"]).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"method": method, "target": target, "detection_quantile": detection_quantile,
                     "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                     "precision": prec, "recall": rec, "fpr": fpr, "f1": f1})
    metrics = pd.DataFrame(rows)
    timing_sum = (cpd_timing.groupby(["method", "target"])
                  .agg(mean_seconds_per_stock=("seconds", "mean")).reset_index())
    return (metrics.merge(timing_sum, on=["method", "target"], how="left")
            .sort_values(["target", "method"]).reset_index(drop=True))


def compute_robustness_metrics(daily_scores, cpd_timing, quantiles=ROBUSTNESS_QUANTILES):
    return pd.concat([compute_cpd_metrics(daily_scores, cpd_timing, q) for q in quantiles],
                     ignore_index=True)


def score_summary(cpd_scores: pd.DataFrame) -> pd.DataFrame:
    return (cpd_scores.groupby(["method", "target"], observed=True)
            .agg(rows=("score", "size"), tickers=("ticker", "nunique"),
                 score_mean=("score", "mean"), score_std=("score", "std"))
            .reset_index())


# ---------------------------------------------------------------------------
# NB03 feature selection
# ---------------------------------------------------------------------------

def selected_target(target_cols: dict) -> str:
    return "vs_market" if "vs_market" in target_cols else "raw"


def select_nb03_features(cpd_scores, target_cols, method=SELECTED_METHOD,
                         detection_quantile=DETECTION_QUANTILE):
    target = selected_target(target_cols)
    mask = (cpd_scores["method"].astype(str).eq(method)
            & cpd_scores["target"].astype(str).eq(target))
    sel = cpd_scores.loc[mask, ["date", "ticker", "score"]].copy()
    threshold = sel["score"].quantile(detection_quantile)
    sel = sel.rename(columns={"score": "selected_cpd_score"})
    sel["selected_cpd_flag"] = sel["selected_cpd_score"].ge(threshold)
    sel = sel.sort_values(["ticker", "date"]).reset_index(drop=True)
    sel["selected_cpd_score_lag1"] = sel.groupby("ticker", sort=False)["selected_cpd_score"].shift(1)
    sel["selected_cpd_flag_lag1"] = sel.groupby("ticker", sort=False)["selected_cpd_flag"].shift(1)
    sel["return_definition"] = target
    sel["method"] = method
    sel = sel.sort_values(["date", "ticker"]).reset_index(drop=True)
    decision = pd.DataFrame([
        {"choice": "method", "value": method, "rationale": "online, causal, continuous score, scalable"},
        {"choice": "target", "value": target, "rationale": "market shocks removed → idiosyncratic focus"},
        {"choice": "detection_quantile", "value": f"q{int(detection_quantile*100)}",
         "rationale": "robust across tested thresholds"},
        {"choice": "lag", "value": "t → t+1", "rationale": "no lookahead for the LSTM"},
    ])
    return sel, decision


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_nb02_outputs(root, cpd_scores, cpd_metrics_robustness, selected_features):
    paths = nb02_paths(root)
    paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    (cpd_scores.assign(score=lambda x: x["score"].astype("float32"))
     .to_parquet(paths["cpd_scores"], index=False))
    cpd_metrics_robustness.to_parquet(paths["cpd_metrics"], index=False)
    selected_features.to_parquet(paths["cpd_features"], index=False)
    return pd.DataFrame([
        {"output": "cpd_scores", "path": paths["cpd_scores"].as_posix(), "rows": len(cpd_scores)},
        {"output": "cpd_metrics", "path": paths["cpd_metrics"].as_posix(), "rows": len(cpd_metrics_robustness)},
        {"output": "cpd_features_nb03", "path": paths["cpd_features"].as_posix(), "rows": len(selected_features)},
    ])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_known_events_timeline(benchmark_ew, known_events):
    sxxr = benchmark_ew[["date", "sxxr_1d_ret"]].dropna().sort_values("date")
    sxxr_cum = sxxr.assign(cum=(1 + sxxr["sxxr_1d_ret"]).cumprod())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sxxr_cum["date"], y=sxxr_cum["cum"],
                             mode="lines", name="SXXR", line=dict(color="black", width=1.2)))
    for ev in known_events.itertuples(index=False):
        fig.add_vline(x=ev.event_date, line=dict(color="rgba(220,0,0,0.28)", width=1, dash="dash"))
        fig.add_annotation(x=ev.event_date, y=1.0, yref="paper", text=ev.event,
                           showarrow=False, textangle=-90,
                           font=dict(size=7, color="rgba(120,0,0,0.85)"),
                           xanchor="left", yanchor="bottom")
    fig.update_layout(title="KNOWN_EVENTS on SXXR cumulative",
                      height=440, width=1200, showlegend=False,
                      margin=dict(l=50, r=20, t=60, b=40))
    fig.update_yaxes(title="SXXR cumulative (base 1)")
    return fig


def plot_method_example(ticker_panel, scores_by_target, method_label, ticker_label, known_events):
    targets_used = list(scores_by_target.keys())
    n_rows = 1 + len(targets_used)
    titles = [f"{ticker_label} — close price"] + [f"{method_label} | {t}" for t in targets_used]
    row_heights = [0.35] + [0.65 / len(targets_used)] * len(targets_used)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=row_heights, subplot_titles=titles)
    fig.add_trace(go.Scatter(x=ticker_panel["date"], y=ticker_panel["price"],
                             mode="lines", name="Price", line=dict(color="black", width=1.2)),
                  row=1, col=1)
    for i, target in enumerate(targets_used, start=2):
        sc = scores_by_target[target]
        fig.add_trace(go.Scatter(x=sc["date"], y=sc["score"], mode="lines", name=target,
                                 line=dict(color=COLOR_TARGET.get(target, "#444"), width=1.0)),
                      row=i, col=1)
        fig.update_yaxes(range=[0, 1.02], row=i, col=1)
    for ev in known_events["event_date"]:
        if ticker_panel["date"].min() <= ev <= ticker_panel["date"].max():
            fig.add_vline(x=ev, line=dict(color="rgba(220,0,0,0.15)", width=1, dash="dash"),
                          row="all", col=1)
    fig.update_layout(height=240 + 180 * len(targets_used), width=1150,
                      showlegend=False, hovermode="x unified",
                      margin=dict(l=50, r=20, t=50, b=20))
    return fig


def plot_fpr_recall(metrics_df, title_suffix=""):
    fig = go.Figure()
    for method, sub in metrics_df.groupby("method", observed=True):
        sec = sub["mean_seconds_per_stock"].fillna(0).to_numpy()
        size = 10 + 25 * (sec / max(sec.max(), 1e-9))
        fig.add_trace(go.Scatter(
            x=sub["recall"], y=sub["fpr"], mode="markers+text",
            text=sub["target"].astype(str), textposition="top center",
            name=str(method), marker=dict(size=size, line=dict(width=1, color="DarkSlateGrey")),
        ))
    fig.add_trace(go.Scatter(x=[1], y=[0], mode="markers+text", marker=dict(size=18, symbol="star", color="black"),
                             text=["ideal"], textposition="bottom center", showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=f"FPR vs Recall — {title_suffix}", xaxis_title="Recall", yaxis_title="FPR",
                      height=520, width=900, margin=dict(l=50, r=20, t=60, b=50))
    return fig


def plot_robustness(cpd_metrics_robustness, quantiles=ROBUSTNESS_QUANTILES):
    fig = make_subplots(rows=1, cols=len(quantiles), shared_yaxes=True,
                        subplot_titles=[f"q={q:.2f}" for q in quantiles])
    for idx, q in enumerate(quantiles, start=1):
        sub = cpd_metrics_robustness.loc[cpd_metrics_robustness["detection_quantile"].eq(q)]
        for method, ss in sub.groupby("method", observed=True):
            fig.add_trace(go.Scatter(x=ss["recall"], y=ss["fpr"], mode="markers",
                                     name=str(method), legendgroup=str(method),
                                     showlegend=(idx == 1),
                                     marker=dict(size=10, line=dict(width=1, color="DarkSlateGrey")),
                                     text=ss["target"].astype(str)),
                          row=1, col=idx)
        fig.add_trace(go.Scatter(x=[1], y=[0], mode="markers",
                                 marker=dict(size=14, symbol="star", color="black"),
                                 showlegend=False, hoverinfo="skip"), row=1, col=idx)
        fig.update_xaxes(title_text="Recall", row=1, col=idx)
    fig.update_yaxes(title_text="FPR", row=1, col=1)
    fig.update_layout(height=460, width=1200, margin=dict(l=50, r=20, t=60, b=50))
    return fig


def plot_sector_heatmap(panel, cpd_scores, target_cols, method=SELECTED_METHOD,
                        detection_quantile=DETECTION_QUANTILE):
    sector_data = sector_cpd_scores(cpd_scores, panel, method=method,
                                    detection_quantile=detection_quantile)
    if sector_data.empty:
        return None
    pivot = sector_data.pivot(index="sector", columns="year", values="flagging_rate").sort_index()
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Reds",
                    labels=dict(x="Year", y="Sector", color="% flagged days"),
                    title=f"Sector CPD heatmap — {method} (% flagged days)")
    fig.update_layout(height=500, width=1100, margin=dict(l=140, r=20, t=60, b=40))
    return fig


def timing_table(cpd_timing: pd.DataFrame) -> pd.DataFrame:
    return (cpd_timing.groupby(["method"], observed=True)
            .agg(n_stocks=("ticker", "nunique"),
                 mean_sec_per_stock=("seconds", "mean"),
                 total_seconds=("seconds", "sum"))
            .round(4).sort_values("mean_sec_per_stock").reset_index())
