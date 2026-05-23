"""NB03: Deep Momentum Network (DMN) — STOXX 600 baseline.

Architecture : LSTM(64) → Linear(1) → Sigmoid → position ∈ [0, 1]
Loss         : −Sharpe annualisé avec proxy 25 bps de coûts de transaction
Walk-forward : fenêtre expansive, pas annuel depuis 2019
Séquence     : τ = 21 jours (paper Wood et al. 2022: quasi-optimal)
Features     : 9 momentum (NB01) + 6 CPD lag1 (NB02) = 15 total
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOMENTUM_FEATURES = [
    "norm_1d", "norm_5d", "norm_21d", "norm_63d", "norm_252d",
    "macd_1_21", "macd_8_24", "macd_12_26",
    "ewma_vol",
]
CPD_FEATURES = [
    "nu_adaptive_cusum_lag1",  "gamma_adaptive_cusum_lag1",
    "nu_jump_process_lag1",    "gamma_jump_process_lag1",
    "nu_rolling_ttest_lag1",   "gamma_rolling_ttest_lag1",
]
FEATURE_COLS = MOMENTUM_FEATURES + CPD_FEATURES   # 15 features total
TARGET_COL   = "next_return"

SEQ_LEN       = 21      # LSTM lookback τ (paper: notable gain at 21d, quasi-optimal)
HIDDEN_SIZE   = 64
NUM_LAYERS    = 1
TC_BPS        = 25      # transaction cost proxy in bps
LEARNING_RATE = 1e-3
MAX_EPOCHS    = 50
BATCH_SIZE    = 512
PATIENCE      = 5       # early stopping patience on validation Sharpe
VAL_FRAC      = 0.10    # chronological hold-out within training set
RANDOM_SEED   = 42
TEST_START    = 2019    # first walk-forward test year

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def resolve_project_root(start=None) -> Path:
    root = (Path.cwd() if start is None else Path(start)).resolve()
    for p in [root, root.parent]:
        if (p / "configs" / "default.yaml").exists():
            return p
    raise FileNotFoundError("configs/default.yaml not found")


def nb03_paths(root=None) -> dict[str, Path]:
    r = resolve_project_root(root)
    d = r / "data" / "processed" / "stoxx600"
    return {
        "project_root":  r,
        "processed_dir": d,
        "panel":         d / "panel.parquet",
        "cpd_features":  d / "cpd_features_nb03.parquet",
        "positions":     d / "positions.parquet",
        "fold_metrics":  d / "fold_metrics.parquet",
    }


# ---------------------------------------------------------------------------
# Data loading & feature matrix
# ---------------------------------------------------------------------------

def load_nb03_inputs(root=None) -> dict:
    paths = nb03_paths(root)
    for key in ("panel", "cpd_features"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing input: {paths[key]}")

    panel = pd.read_parquet(paths["panel"])
    cpd   = pd.read_parquet(paths["cpd_features"])
    for df in (panel, cpd):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    return {"paths": paths, "panel": panel, "cpd_features": cpd}


def build_feature_matrix(panel: pd.DataFrame,
                         cpd_features: pd.DataFrame) -> pd.DataFrame:
    """Merge panel + CPD features into a single tidy frame.

    Missing CPD features → 0 (neutral signal: no detected changepoint).
    """
    base_cols = (["date", "ticker", "price"]
                 + [c for c in MOMENTUM_FEATURES if c in panel.columns]
                 + [TARGET_COL])
    base = panel[[c for c in base_cols if c in panel.columns]].copy()

    cpd_cols = (["date", "ticker"]
                + [c for c in CPD_FEATURES if c in cpd_features.columns])
    feat = base.merge(cpd_features[cpd_cols], on=["date", "ticker"], how="left")

    for col in CPD_FEATURES:
        if col in feat.columns:
            feat[col] = feat[col].fillna(0.0)

    # Require at least 3 momentum features and a valid target
    must_have = [c for c in MOMENTUM_FEATURES[:3] if c in feat.columns]
    feat = feat.dropna(subset=must_have + [TARGET_COL])

    return feat.sort_values(["ticker", "date"]).reset_index(drop=True)


def input_summary(feat: pd.DataFrame) -> pd.DataFrame:
    fcols = [c for c in FEATURE_COLS if c in feat.columns]
    return pd.DataFrame([
        {"item": "rows",       "value": f"{len(feat):,}"},
        {"item": "tickers",    "value": f"{feat['ticker'].nunique():,}"},
        {"item": "date range", "value": f"{feat['date'].min().date()} → {feat['date'].max().date()}"},
        {"item": "features",   "value": str(len(fcols))},
        {"item": "target",     "value": TARGET_COL},
        {"item": "device",     "value": str(DEVICE)},
    ])


def feature_summary(feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURE_COLS:
        if col not in feat.columns:
            continue
        s = feat[col].dropna()
        rows.append({
            "feature":    col,
            "type":       "CPD" if ("nu_" in col or "gamma_" in col) else "Momentum",
            "non_null_%": round(feat[col].notna().mean() * 100, 1),
            "mean":       round(float(s.mean()), 4),
            "std":        round(float(s.std()),  4),
        })
    return pd.DataFrame(rows)


def select_example_ticker(feat: pd.DataFrame, preferred: str = "ASML") -> str:
    cands = sorted(feat.loc[
        feat["ticker"].str.contains(preferred, case=False, na=False), "ticker"
    ].unique())
    return cands[0] if cands else str(feat["ticker"].value_counts().index[0])


def model_architecture_table(n_features: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"layer": "Input",   "shape": f"(batch, {SEQ_LEN}, {n_features})",
         "note": f"{SEQ_LEN}-day sequence · {n_features} features"},
        {"layer": "LSTM",    "shape": f"(batch, {SEQ_LEN}, {HIDDEN_SIZE})",
         "note": f"hidden={HIDDEN_SIZE} · layers={NUM_LAYERS}"},
        {"layer": "Linear",  "shape": "(batch, 1)",
         "note": "last hidden state only"},
        {"layer": "Sigmoid", "shape": "(batch, 1)",
         "note": "position ∈ [0, 1]  (long-only)"},
        {"layer": "Loss",    "shape": "scalar",
         "note": f"−Sharpe annualisé · TC proxy {TC_BPS} bps"},
    ])


# ---------------------------------------------------------------------------
# Sequence dataset
# ---------------------------------------------------------------------------

def make_sequences(feat: pd.DataFrame,
                   feature_cols: list[str],
                   seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Build (N, seq_len, n_features) X and (N,) y from a feature frame.

    At row i (date t): X = features[t-seq_len+1 … t], y = next_return[t].
    Features at t are computed from data ≤ t → causally valid.
    next_return[t] = mkt_rel_1d[t+1] = return earned from t to t+1.
    """
    X_list, y_list = [], []
    for _, grp in feat.groupby("ticker", sort=False):
        grp  = grp.sort_values("date")
        vals = grp[feature_cols].to_numpy(dtype=np.float32)
        tgt  = grp[TARGET_COL].to_numpy(dtype=np.float32)
        vals = np.nan_to_num(vals, nan=0.0)
        n    = len(vals)
        for i in range(seq_len - 1, n):
            y = tgt[i]
            if not np.isfinite(y):
                continue
            X_list.append(vals[i - seq_len + 1: i + 1])
            y_list.append(y)

    if not X_list:
        return (np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
                np.empty(0, dtype=np.float32))
    return np.stack(X_list), np.array(y_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DMN(nn.Module):
    """Deep Momentum Network: LSTM → Linear → Sigmoid → position ∈ [0, 1]."""

    def __init__(self, n_features: int,
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers:  int = NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return torch.sigmoid(self.head(out[:, -1, :]))


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sharpe_loss(positions: torch.Tensor,
                returns:   torch.Tensor,
                tc_bps:    float = TC_BPS) -> torch.Tensor:
    """Negative annualised Sharpe with TC proxy (position magnitude cost).

    Proper Δposition TC requires sequential order → applied fully in NB04.
    Here: TC ≈ tc × |position| as a penalty on large positions.
    """
    tc        = tc_bps / 10_000
    pos       = positions.squeeze()
    strat_ret = pos * returns - tc * pos.abs()
    mean_r    = strat_ret.mean()
    std_r     = strat_ret.std() + 1e-8
    return -(mean_r / std_r) * (252 ** 0.5)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_loader(X: np.ndarray, y: np.ndarray,
                 batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X).to(DEVICE),
        torch.from_numpy(y).to(DEVICE),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_fold(X_train: np.ndarray, y_train: np.ndarray,
               n_features:  int,
               max_epochs:  int   = MAX_EPOCHS,
               patience:    int   = PATIENCE,
               lr:          float = LEARNING_RATE,
               batch_size:  int   = BATCH_SIZE,
               val_frac:    float = VAL_FRAC,
               seed:        int   = RANDOM_SEED) -> tuple[DMN, list[float]]:
    """Train one DMN fold (chronological val split, early stopping on val Sharpe).

    Returns (best_model, val_sharpe_per_epoch).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_val       = max(batch_size, int(len(X_train) * val_frac))
    X_tr, X_vl = X_train[:-n_val], X_train[-n_val:]
    y_tr, y_vl = y_train[:-n_val], y_train[-n_val:]

    tr_loader   = _make_loader(X_tr, y_tr, batch_size=batch_size, shuffle=True)
    X_vl_t      = torch.from_numpy(X_vl).to(DEVICE)
    y_vl_t      = torch.from_numpy(y_vl).to(DEVICE)

    model       = DMN(n_features).to(DEVICE)
    opt         = torch.optim.Adam(model.parameters(), lr=lr)

    best_val    = -np.inf
    best_state  = None
    wait        = 0
    history: list[float] = []

    for _ in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = sharpe_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vl_pos  = model(X_vl_t).squeeze()
            vl_ret  = vl_pos * y_vl_t - (TC_BPS / 10_000) * vl_pos.abs()
            val_sr  = float((vl_ret.mean() / (vl_ret.std() + 1e-8)) * (252 ** 0.5))
        history.append(val_sr)

        if val_sr > best_val:
            best_val   = val_sr
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def walk_forward_splits(feat: pd.DataFrame,
                        test_start: int = TEST_START) -> list[dict]:
    """Expanding-window splits: one per test year from test_start onward."""
    years      = sorted(feat["date"].dt.year.unique())
    test_years = [y for y in years if y >= test_start]
    splits     = []
    for ty in test_years:
        tr_end   = pd.Timestamp(f"{ty - 1}-12-31")
        te_start = pd.Timestamp(f"{ty}-01-01")
        te_end   = pd.Timestamp(f"{ty}-12-31")
        if feat.loc[feat["date"] <= tr_end].shape[0] < 1_000:
            continue
        splits.append({
            "test_year":  ty,
            "train_end":  tr_end,
            "test_start": te_start,
            "test_end":   te_end,
        })
    return splits


def _predict_positions(model: DMN,
                       feat:  pd.DataFrame,
                       sp:    dict,
                       feature_cols: list[str],
                       seq_len: int) -> pd.DataFrame:
    """Predict positions for one test year.

    Uses a calendar-day buffer before test_start so every test date
    has a full seq_len-day lookback available.
    """
    test_start = sp["test_start"]
    test_end   = sp["test_end"]
    ctx_start  = test_start - pd.Timedelta(days=seq_len * 3)

    rows = []
    model.eval()
    with torch.no_grad():
        for ticker, grp in feat.groupby("ticker", sort=False):
            grp    = grp.sort_values("date")
            window = grp.loc[grp["date"] >= ctx_start].copy()
            vals   = window[feature_cols].to_numpy(dtype=np.float32)
            vals   = np.nan_to_num(vals, nan=0.0)
            dates  = window["date"].to_numpy()
            n      = len(vals)
            if n < seq_len:
                continue

            in_test  = ((window["date"] >= test_start) &
                        (window["date"] <= test_end)).to_numpy()
            idx_test = np.where(in_test)[0]
            valid    = idx_test[idx_test >= seq_len - 1]
            if len(valid) == 0:
                continue

            seqs = np.stack([vals[i - seq_len + 1: i + 1] for i in valid])
            pos  = model(torch.from_numpy(seqs).to(DEVICE)).squeeze().cpu().numpy()
            if pos.ndim == 0:
                pos = pos.reshape(1)

            for i, p in zip(valid, pos):
                rows.append({
                    "date":     dates[i],
                    "ticker":   ticker,
                    "position": float(p),
                })
    return pd.DataFrame(rows)


def run_walk_forward(feat: pd.DataFrame,
                     feature_cols: list[str] | None = None,
                     seq_len:  int  = SEQ_LEN,
                     verbose:  bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full walk-forward training and prediction.

    Returns:
        positions    : (date, ticker, position) — out-of-sample only
        fold_metrics : per-fold training statistics
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in feat.columns]

    n_features = len(feature_cols)
    splits     = walk_forward_splits(feat)
    all_pos, fold_rows = [], []

    for sp in splits:
        ty = sp["test_year"]
        t0 = time.perf_counter()

        tr_feat    = feat.loc[feat["date"] <= sp["train_end"]]
        X_tr, y_tr = make_sequences(tr_feat, feature_cols, seq_len)
        if len(X_tr) < 500:
            if verbose:
                print(f"  fold {ty} — too few training samples, skipped")
            continue

        model, history = train_fold(X_tr, y_tr, n_features=n_features)
        elapsed        = time.perf_counter() - t0

        pos_df = _predict_positions(model, feat, sp, feature_cols, seq_len)
        if not pos_df.empty:
            all_pos.append(pos_df)

        best_sharpe = max(history) if history else np.nan
        fold_rows.append({
            "test_year":       ty,
            "val_sharpe_best": round(best_sharpe, 3),
            "epochs":          len(history),
            "n_train_seq":     len(X_tr),
            "seconds":         round(elapsed, 1),
        })
        if verbose:
            print(f"  fold {ty} | val_sharpe={best_sharpe:.3f} | "
                  f"epochs={len(history)} | {elapsed:.0f}s")

    positions    = (pd.concat(all_pos, ignore_index=True) if all_pos
                    else pd.DataFrame(columns=["date", "ticker", "position"]))
    fold_metrics = pd.DataFrame(fold_rows)
    return positions, fold_metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_feature_correlation(feat: pd.DataFrame,
                              feature_cols: list[str] | None = None) -> go.Figure:
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in feat.columns]
    corr = feat[feature_cols].corr().round(2)
    fig  = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        colorbar=dict(title="ρ"),
    ))
    fig.update_layout(
        title="Feature correlation matrix — momentum vs CPD signals",
        template="plotly_white",
        height=540, width=660,
        margin=dict(l=140, r=40, t=60, b=140),
        xaxis=dict(tickangle=-45),
    )
    return fig


def plot_fold_sharpe(fold_metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fold_metrics["test_year"],
        y=fold_metrics["val_sharpe_best"],
        mode="lines+markers",
        line=dict(color="steelblue", width=2),
        marker=dict(size=8),
    ))
    fig.add_hline(y=0, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        title="Walk-forward — best validation Sharpe per fold",
        xaxis_title="Test year",
        yaxis_title="Sharpe (validation)",
        template="plotly_white",
        height=380,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def plot_position_on_ticker(feat:      pd.DataFrame,
                             positions: pd.DataFrame,
                             ticker:    str) -> go.Figure:
    t_feat = (feat.loc[feat["ticker"].eq(ticker), ["date", "price"]]
              .sort_values("date"))
    t_pos  = (positions.loc[positions["ticker"].eq(ticker), ["date", "position"]]
              .sort_values("date"))
    if t_pos.empty:
        return go.Figure().update_layout(
            title=f"{ticker} — no out-of-sample positions yet")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.05,
                        subplot_titles=[f"{ticker} — price",
                                        "DMN position [0, 1]"])
    fig.add_trace(go.Scatter(
        x=t_feat["date"], y=t_feat["price"],
        mode="lines", line=dict(color="black", width=1.2),
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_pos["date"], y=t_pos["position"],
        mode="lines", line=dict(color="steelblue", width=1.2),
        showlegend=False), row=2, col=1)
    fig.add_hline(y=0.5, line=dict(color="gray", dash="dash", width=1), row=2, col=1)
    fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=440,
        margin=dict(l=60, r=20, t=60, b=20),
        hovermode="x unified",
    )
    return fig


def plot_position_distribution(positions: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=positions["position"],
        nbinsx=50,
        marker_color="steelblue",
        opacity=0.8,
        histnorm="probability density",
    ))
    fig.add_vline(x=0.5, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        title="Distribution of DMN positions — full out-of-sample period",
        xaxis_title="Position [0, 1]",
        yaxis_title="Density",
        template="plotly_white",
        height=360,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_nb03_outputs(root,
                      positions:    pd.DataFrame,
                      fold_metrics: pd.DataFrame) -> pd.DataFrame:
    paths = nb03_paths(root)
    paths["processed_dir"].mkdir(parents=True, exist_ok=True)
    positions.to_parquet(paths["positions"],    index=False)
    fold_metrics.to_parquet(paths["fold_metrics"], index=False)
    return pd.DataFrame([
        {"output": "positions",    "rows": f"{len(positions):,}",
         "path": paths["positions"].relative_to(paths["project_root"]).as_posix()},
        {"output": "fold_metrics", "rows": f"{len(fold_metrics):,}",
         "path": paths["fold_metrics"].relative_to(paths["project_root"]).as_posix()},
    ])
