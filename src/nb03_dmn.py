"""NB03 helpers: LSTM Deep Momentum Network — Pergam equity adaptation.

Differences from the original paper (Lim, Zohren & Roberts 2019 / Wood et al. 2022)
--------------------------------------------------------------------------------------
* Universe   : ~600 European equities (vs 50 liquid futures)
* Positions  : long-only sigmoid [0, 1]  (vs long/short tanh [-1, 1])
* Costs      : 25 bps per leg, integrated in the Sharpe loss
* No leverage
* Returns denoised relative to market / sector before use as CPD targets

Variants
--------
baseline    — LSTM on momentum features only
cpd_cusum   — baseline + CUSUM score (lag 1 day)
cpd_gp      — baseline + GP Matern32 score (lag 20 days, subsample of tickers)
cpd_bocpd   — baseline + BOCPD posterior score (lag 1 day, subsample of tickers)
"""

from __future__ import annotations

import copy
import math
import random
import sys
import time
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PANEL_PATH         = "data/processed/stoxx600/panel.parquet"
CPD_SCORES_PATH    = "data/processed/stoxx600/cpd_scores.parquet"
BENCHMARK_EW_PATH  = "data/processed/stoxx600/benchmark_ew.parquet"
KNOWN_EVENTS_PATH  = "data/processed/stoxx600/known_events.csv"
DMN_POSITIONS_PATH = "data/processed/stoxx600/dmn_positions.parquet"
DMN_METRICS_PATH   = "data/processed/stoxx600/dmn_metrics.parquet"
DMN_DIAGNOSTICS_PATH = "data/processed/stoxx600/dmn_diagnostics.parquet"


# ---------------------------------------------------------------------------
# Feature constants
# ---------------------------------------------------------------------------

BASE_FEATURES = [
    "1d_log_ret_lag1",
    "20d_log_ret_lag1",
    "252d_log_ret_lag1",
    "1d_arith_ret_lag1",
    "1d_ret_vs_market_lag1",
    "vol_60d_lag1",
    "norm_ret_60d_lag1",
    "month_of_year",
]

# CPD methods
CUSUM_METHOD  = "cusum_sigmoid"
GP_METHOD     = "gp_matern32"
BOCPD_METHOD  = "bocpd"

# CPD feature names (all lagged → no lookahead)
CUSUM_FEATURE = "cpd_score_cusum_sigmoid_lag1"
GP_FEATURE    = "cpd_score_gp_matern32_lag20"
BOCPD_FEATURE = "cpd_score_bocpd_lag1"

PREFERRED_CPD_TARGETS = ("vs_market", "raw")

# Training
TARGET_COL     = "next_return"
SEQUENCE_LENGTH = 63
HIDDEN_SIZE     = 64
DENSE_HIDDEN    = 32
BATCH_SIZE      = 128
LEARNING_RATE   = 1e-3
MAX_EPOCHS      = 20
EARLY_STOPPING_PATIENCE = 5
DROPOUT         = 0.0
SEED            = 42

# Finance
TRADING_DAYS          = 252.0
SQRT_252              = math.sqrt(TRADING_DAYS)
TRANSACTION_COST_BPS  = 25.0                          # Pergam equity universe
TRANSACTION_COST_RATE = TRANSACTION_COST_BPS / 10_000.0  # 0.0025

GP_CAUSAL_LAG_DAYS = 20  # GP computation window requires ~20d look-back


# ---------------------------------------------------------------------------
# Torch helpers
# ---------------------------------------------------------------------------

def _torch():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    return torch, nn, DataLoader, Dataset


def seed_everything(seed=SEED):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch, _, _, _ = _torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def device_name() -> str:
    torch, _, _, _ = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_parquet(path, columns=None, filters=None) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, columns=columns, filters=filters)
    except Exception:
        return pd.read_parquet(path, columns=columns)


def load_nb03_inputs(root: Path) -> dict:
    panel_path       = root / PANEL_PATH
    cpd_scores_path  = root / CPD_SCORES_PATH
    benchmark_path   = root / BENCHMARK_EW_PATH
    known_events_path = root / KNOWN_EVENTS_PATH

    missing = [p for p in [panel_path, cpd_scores_path, benchmark_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"NB03 inputs missing: {[str(p) for p in missing]}")

    panel_cols = ["date", "ticker", "price", "target_next_return", "1d_arith_ret",
                  *[c for c in BASE_FEATURES if c != "month_of_year"]]
    import pyarrow.parquet as pq
    available = pq.read_schema(panel_path).names
    panel = _read_parquet(panel_path, columns=[c for c in panel_cols if c in available])

    cpd_scores = _read_parquet(cpd_scores_path,
                               columns=["date", "ticker", "method", "target", "score"],
                               filters=[("method", "in", [CUSUM_METHOD, GP_METHOD, BOCPD_METHOD]),
                                        ("target", "in", list(PREFERRED_CPD_TARGETS))])
    cpd_scores = cpd_scores.loc[
        cpd_scores["method"].astype(str).isin([CUSUM_METHOD, GP_METHOD, BOCPD_METHOD])
        & cpd_scores["target"].astype(str).isin(PREFERRED_CPD_TARGETS)
    ].copy()

    benchmark_ew = _read_parquet(benchmark_path)
    known_events = (pd.read_csv(known_events_path, parse_dates=["event_date"])
                    if known_events_path.exists()
                    else pd.DataFrame(columns=["event", "event_date"]))

    for frame in [panel, cpd_scores, benchmark_ew, known_events]:
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
        if "event_date" in frame.columns:
            frame["event_date"] = pd.to_datetime(frame["event_date"])

    return {"paths": {"panel": panel_path, "cpd_scores": cpd_scores_path,
                      "benchmark_ew": benchmark_path, "known_events": known_events_path},
            "panel": panel, "cpd_scores": cpd_scores,
            "benchmark_ew": benchmark_ew, "known_events": known_events}


# ---------------------------------------------------------------------------
# Model data preparation
# ---------------------------------------------------------------------------

def _choose_target(cpd_scores: pd.DataFrame, method: str) -> str:
    avail = set(cpd_scores.loc[cpd_scores["method"].astype(str).eq(method),
                               "target"].astype(str))
    for t in PREFERRED_CPD_TARGETS:
        if t in avail:
            return t
    if avail:
        return sorted(avail)[0]
    raise ValueError(f"No CPD scores for method={method!r}")


def _merge_score(frame, cpd_scores, method, target, raw_col):
    score = (cpd_scores.loc[
        cpd_scores["method"].astype(str).eq(method)
        & cpd_scores["target"].astype(str).eq(target),
        ["date", "ticker", "score"]]
             .drop_duplicates(["date", "ticker"])
             .rename(columns={"score": raw_col}))
    return frame.merge(score, on=["date", "ticker"], how="left")


def prepare_model_data(panel: pd.DataFrame, cpd_scores: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in BASE_FEATURES if c != "month_of_year" and c not in panel.columns]
    if missing:
        raise ValueError(f"Panel missing columns: {missing}")

    source_cols = ["date", "ticker", "price", "target_next_return", "1d_arith_ret",
                   *[c for c in BASE_FEATURES if c != "month_of_year"]]
    out = panel[[c for c in source_cols if c in panel.columns]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out["month_of_year"] = ((out["date"].dt.month - 1) / 11.0).astype("float32")
    out[TARGET_COL] = out["target_next_return"]
    if out[TARGET_COL].isna().all():
        out[TARGET_COL] = out.groupby("ticker", sort=False)["1d_arith_ret"].shift(-1)

    # Merge CPD scores
    cusum_target = _choose_target(cpd_scores, CUSUM_METHOD)
    gp_target    = _choose_target(cpd_scores, GP_METHOD)
    out = _merge_score(out, cpd_scores, CUSUM_METHOD, cusum_target, "_cusum_raw")
    out = _merge_score(out, cpd_scores, GP_METHOD, gp_target, "_gp_raw")

    bocpd_available = BOCPD_METHOD in cpd_scores["method"].astype(str).unique()
    if bocpd_available:
        bocpd_target = _choose_target(cpd_scores, BOCPD_METHOD)
        out = _merge_score(out, cpd_scores, BOCPD_METHOD, bocpd_target, "_bocpd_raw")
        out.attrs["bocpd_target"] = bocpd_target
    else:
        out["_bocpd_raw"] = np.nan
        out.attrs["bocpd_target"] = "n/a"

    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Causal lags (t → t+1 for CUSUM/BOCPD; t → t+20 for GP due to computation window)
    out[CUSUM_FEATURE] = out.groupby("ticker", sort=False)["_cusum_raw"].shift(1).fillna(0.0)
    out[GP_FEATURE]    = out.groupby("ticker", sort=False)["_gp_raw"].shift(GP_CAUSAL_LAG_DAYS)
    out[BOCPD_FEATURE] = out.groupby("ticker", sort=False)["_bocpd_raw"].shift(1).fillna(0.0)

    out.attrs["cusum_target"] = cusum_target
    out.attrs["gp_target"]    = gp_target
    return out


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def input_summary(data: dict, frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"item": "panel",            "value": data["paths"]["panel"].as_posix()},
        {"item": "cpd_scores",       "value": data["paths"]["cpd_scores"].as_posix()},
        {"item": "rows",             "value": f"{len(frame):,}"},
        {"item": "tickers",          "value": frame["ticker"].nunique()},
        {"item": "date range",       "value": f"{frame['date'].min().date()} → {frame['date'].max().date()}"},
        {"item": "CUSUM target",     "value": frame.attrs.get("cusum_target", "")},
        {"item": "GP target",        "value": frame.attrs.get("gp_target", "")},
        {"item": "BOCPD target",     "value": frame.attrs.get("bocpd_target", "")},
        {"item": "transaction cost", "value": f"{TRANSACTION_COST_BPS:.0f} bps (Pergam equity universe)"},
    ])


def architecture_summary() -> pd.DataFrame:
    return pd.DataFrame([
        {"item": "network",                "value": "LSTM → dense ReLU → sigmoid"},
        {"item": "position range",         "value": "[0, 1] long-only"},
        {"item": "sequence length",        "value": SEQUENCE_LENGTH},
        {"item": "hidden size",            "value": HIDDEN_SIZE},
        {"item": "dense hidden",           "value": DENSE_HIDDEN},
        {"item": "max epochs",             "value": MAX_EPOCHS},
        {"item": "early stopping patience","value": EARLY_STOPPING_PATIENCE},
        {"item": "seed",                   "value": SEED},
    ])


def loss_summary() -> pd.DataFrame:
    return pd.DataFrame([
        {"item": "gross return",   "value": "w_t · r_{t+1}"},
        {"item": "turnover",       "value": "|w_t − w_{t−1}|"},
        {"item": "cost per unit",  "value": f"{TRANSACTION_COST_RATE:.4f}  ({TRANSACTION_COST_BPS:.0f} bps, Pergam)"},
        {"item": "net return",     "value": "gross return − cost · turnover"},
        {"item": "objective",      "value": "minimise  −Sharpe(net return) annualised"},
    ])


def feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, cfg in variant_configs(frame).items():
        tickers = frame["ticker"].nunique() if cfg["tickers"] is None else len(cfg["tickers"])
        rows.append({"variant": variant, "n_features": len(cfg["feature_cols"]),
                     "n_tickers": tickers, "comment": cfg["comment"]})
    return pd.DataFrame(rows)


def variant_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, cfg in variant_configs(frame).items():
        tickers = frame["ticker"].nunique() if cfg["tickers"] is None else len(cfg["tickers"])
        rows.append({"variant": key, "label": cfg["label"],
                     "n_features": len(cfg["feature_cols"]), "n_tickers": tickers,
                     "position": "sigmoid [0, 1]",
                     "loss": f"net Sharpe − {TRANSACTION_COST_BPS:.0f} bps costs",
                     "features": ", ".join(cfg["feature_cols"])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Walk-forward folds
# ---------------------------------------------------------------------------

def make_folds(frame: pd.DataFrame, first_test_year: int = 2013,
               last_test_year: int = 2026) -> pd.DataFrame:
    """Annual walk-forward folds — training starts from 2006 data."""
    max_date = frame["date"].max()
    min_date = frame["date"].min()
    rows = []
    for fold_id, test_year in enumerate(range(first_test_year, last_test_year + 1), start=1):
        train_start = min_date  # use all available history
        train_end   = pd.Timestamp(f"{test_year - 2}-12-31")
        val_start   = pd.Timestamp(f"{test_year - 1}-01-01")
        val_end     = pd.Timestamp(f"{test_year - 1}-12-31")
        test_start  = pd.Timestamp(f"{test_year}-01-01")
        test_end    = min(pd.Timestamp(f"{test_year}-12-31"), max_date)
        if test_start > max_date or test_end < min_date:
            continue
        rows.append({"fold": fold_id, "train_start": train_start, "train_end": train_end,
                     "validation_start": val_start, "validation_end": val_end,
                     "test_start": test_start, "test_end": test_end,
                     "train_years": f"{int(min_date.year)}-{test_year - 2}",
                     "validation_year": test_year - 1, "test_year": test_year})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

def variant_configs(frame: pd.DataFrame | None = None) -> dict:
    gp_tickers = bocpd_tickers = None
    if frame is not None:
        if GP_FEATURE in frame.columns:
            gp_tickers = sorted(frame.loc[frame[GP_FEATURE].notna(), "ticker"].dropna().unique())
        if BOCPD_FEATURE in frame.columns:
            bocpd_tickers = sorted(frame.loc[frame[BOCPD_FEATURE].notna(), "ticker"].dropna().unique())
    return {
        "baseline": {
            "label": "Baseline", "feature_cols": list(BASE_FEATURES),
            "tickers": None, "comment": "Momentum features only."},
        "cpd_cusum": {
            "label": "Baseline + CUSUM", "feature_cols": [*BASE_FEATURES, CUSUM_FEATURE],
            "tickers": None, "comment": "CUSUM score, lag 1 day."},
        "cpd_gp": {
            "label": "Baseline + GP", "feature_cols": [*BASE_FEATURES, GP_FEATURE],
            "tickers": gp_tickers,
            "comment": f"GP Matern32 score, lag {GP_CAUSAL_LAG_DAYS} days."},
        "cpd_bocpd": {
            "label": "Baseline + BOCPD", "feature_cols": [*BASE_FEATURES, BOCPD_FEATURE],
            "tickers": bocpd_tickers, "comment": "BOCPD posterior score, lag 1 day."},
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DMNDataset:
    """Sequence dataset returning (current_window, previous_window, target) triples."""

    def __init__(self, frame, feature_cols, target_col=TARGET_COL, sequence_length=SEQUENCE_LENGTH):
        _, _, _, Dataset = _torch()
        feature_cols = list(feature_cols)

        class _Dataset(Dataset):
            def __init__(inner):
                inner.blocks = []
                inner.samples = []
                data = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
                for ticker, group in data.groupby("ticker", sort=False):
                    x = group[feature_cols].to_numpy(dtype=np.float32)
                    y = group[target_col].to_numpy(dtype=np.float32)
                    dates = group["date"].to_numpy()
                    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
                    valid_window = (pd.Series(valid)
                                   .rolling(sequence_length, min_periods=sequence_length)
                                   .sum().eq(sequence_length).to_numpy())
                    prev_window = np.r_[False, valid_window[:-1]]
                    end_idx = np.flatnonzero(valid_window & prev_window)
                    if len(end_idx) == 0:
                        continue
                    bid = len(inner.blocks)
                    inner.blocks.append({"ticker": ticker, "x": x, "y": y, "dates": dates})
                    inner.samples.extend((bid, int(i)) for i in end_idx)

            def __len__(inner):
                return len(inner.samples)

            def __getitem__(inner, idx):
                bid, end_idx = inner.samples[idx]
                b = inner.blocks[bid]
                s = end_idx - sequence_length + 1
                torch, _, _, _ = _torch()
                return (torch.from_numpy(b["x"][s:end_idx + 1]),
                        torch.from_numpy(b["x"][s - 1:end_idx]),
                        torch.tensor(b["y"][end_idx], dtype=torch.float32))

            def metadata(inner, idx):
                bid, end_idx = inner.samples[idx]
                b = inner.blocks[bid]
                return {"date": b["dates"][end_idx], "ticker": b["ticker"],
                        "target_return": b["y"][end_idx]}

            def diagnostics(inner):
                if not inner.blocks:
                    return pd.DataFrame([{"n_tickers": 0, "n_samples": 0}])
                spb = pd.Series([sum(1 for s in inner.samples if s[0] == bid)
                                 for bid in range(len(inner.blocks))])
                return pd.DataFrame([{"n_tickers": len(inner.blocks),
                                      "n_samples": len(inner.samples),
                                      "samples_median": spb.median(),
                                      "samples_min": spb.min(),
                                      "samples_max": spb.max()}])

        self.dataset = _Dataset()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMDMN:
    """Long-only LSTM DMN (sigmoid output → positions in [0, 1])."""

    def __new__(cls, input_size, hidden_size=HIDDEN_SIZE, dense_hidden=DENSE_HIDDEN, dropout=DROPOUT):
        torch, nn, _, _ = _torch()

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(int(input_size), int(hidden_size), num_layers=1,
                                    dropout=0.0, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(int(hidden_size), int(dense_hidden)), nn.ReLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(dense_hidden), 1), nn.Sigmoid())

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return _Model()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def net_strategy_returns(positions, realized_returns, previous_positions,
                          cost_rate=TRANSACTION_COST_RATE):
    turnover = (positions - previous_positions).abs()
    return positions * realized_returns - float(cost_rate) * turnover


def sharpe_loss(positions, realized_returns, previous_positions, eps=1e-6,
                cost_rate=TRANSACTION_COST_RATE):
    net = net_strategy_returns(positions, realized_returns, previous_positions, cost_rate=cost_rate)
    return -math.sqrt(TRADING_DAYS) * net.mean() / (net.std(unbiased=False) + eps)


def batch_sharpe(positions, realized_returns, previous_positions, eps=1e-6):
    with _torch()[0].no_grad():
        gross = positions * realized_returns
        net = net_strategy_returns(positions, realized_returns, previous_positions)
        gs = gross.mean() / (gross.std(unbiased=False) + eps) * SQRT_252
        ns = net.mean() / (net.std(unbiased=False) + eps) * SQRT_252
    return float(gs.detach().cpu()), float(ns.detach().cpu())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def split_fold_data(frame, fold_row, tickers=None):
    fold = fold_row._asdict() if hasattr(fold_row, "_asdict") else dict(fold_row)
    src = frame if tickers is None else frame.loc[frame["ticker"].isin(tickers)]
    return (src.loc[src["date"].between(fold["train_start"], fold["train_end"])].copy(),
            src.loc[src["date"].between(fold["validation_start"], fold["validation_end"])].copy(),
            src.loc[src["date"].between(fold["test_start"], fold["test_end"])].copy())


def normalize_fold_features(train, val, test, feature_cols):
    mu  = train[list(feature_cols)].mean().fillna(0.0)
    std = train[list(feature_cols)].std(ddof=0).replace(0, np.nan).fillna(1.0)
    outputs = []
    for frame in [train.copy(), val.copy(), test.copy()]:
        frame[list(feature_cols)] = (frame[list(feature_cols)] - mu) / std
        frame[list(feature_cols)] = frame[list(feature_cols)].replace([np.inf, -np.inf], np.nan)
        outputs.append(frame)
    scaler = pd.DataFrame({"feature": list(feature_cols), "mu_train": mu.values, "std_train": std.values})
    return (*outputs, scaler)


def make_loader(dataset, batch_size=BATCH_SIZE, shuffle=False):
    torch, _, DataLoader, _ = _torch()
    gen = torch.Generator()
    gen.manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=gen if shuffle else None, drop_last=False, num_workers=0)


def evaluate_model(model, loader, device=None):
    torch, _, _, _ = _torch()
    device = device or torch.device(device_name())
    model.eval()
    pos_list, prev_list, ret_list = [], [], []
    with torch.no_grad():
        for xb, prev_xb, yb in loader:
            pos = model(xb.to(device).float()).cpu()
            prev_pos = model(prev_xb.to(device).float()).cpu()
            pos_list.append(pos); prev_list.append(prev_pos); ret_list.append(yb)
    positions = torch.cat(pos_list)
    previous_positions = torch.cat(prev_list)
    returns = torch.cat(ret_list)
    loss = sharpe_loss(positions, returns, previous_positions)
    gs, ns = batch_sharpe(positions, returns, previous_positions)
    return float(loss.detach().cpu()), gs, ns


def train_one_fold(frame, variant, fold_row, show_diagnostics=False):
    torch, _, _, _ = _torch()
    device = torch.device(device_name())
    cfg = variant_configs(frame)[variant]
    feature_cols = cfg["feature_cols"]
    fold = fold_row._asdict() if hasattr(fold_row, "_asdict") else dict(fold_row)

    train, val, test = split_fold_data(frame, fold_row, tickers=cfg["tickers"])
    train, val, test, scaler = normalize_fold_features(train, val, test, feature_cols)

    train_ds = DMNDataset(train, feature_cols).dataset
    val_ds   = DMNDataset(val, feature_cols).dataset
    test_ds  = DMNDataset(test, feature_cols).dataset

    diagnostics = (pd.concat([train_ds.diagnostics().assign(split="train"),
                               val_ds.diagnostics().assign(split="validation"),
                               test_ds.diagnostics().assign(split="test")], ignore_index=True)
                   if show_diagnostics else pd.DataFrame())

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Empty train/val for {variant}, fold {fold['fold']}")

    seed_everything(SEED + int(fold["fold"]))
    model = LSTMDMN(input_size=len(feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds, shuffle=False)

    best_val, best_state, best_epoch, patience_count = np.inf, None, 0, 0
    history = []
    tic_total = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        tic = time.perf_counter()
        model.train()
        batch_losses = []
        for xb, prev_xb, yb in train_loader:
            xb, prev_xb, yb = xb.to(device).float(), prev_xb.to(device).float(), yb.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            loss = sharpe_loss(model(xb), yb, model(prev_xb))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
        val_loss, val_gs, val_ns = evaluate_model(model, val_loader, device=device)

        if np.isfinite(val_loss) and val_loss < best_val - 1e-6:
            best_val, best_epoch, best_state, patience_count = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            patience_count += 1

        history.append({"variant": variant, "fold": fold["fold"], "test_year": fold["test_year"],
                         "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_gross_sharpe": val_gs, "val_net_sharpe": val_ns,
                         "best_epoch": best_epoch,
                         "n_train_samples": len(train_ds), "n_val_samples": len(val_ds),
                         "n_test_samples": len(test_ds),
                         "seconds_epoch": time.perf_counter() - tic,
                         "seconds_total": time.perf_counter() - tic_total})
        if patience_count >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"model": model, "fold": fold, "test_dataset": test_ds,
            "history": pd.DataFrame(history), "scaler": scaler,
            "dataset_diagnostics": diagnostics}


def predict_test(fold_result, variant) -> pd.DataFrame:
    torch, _, _, _ = _torch()
    device = torch.device(device_name())
    model   = fold_result["model"]
    dataset = fold_result["test_dataset"]
    fold    = fold_result["fold"]
    if len(dataset) == 0:
        return pd.DataFrame()

    loader = make_loader(dataset, shuffle=False)
    rows, cursor = [], 0
    model.eval()
    with torch.no_grad():
        for xb, prev_xb, yb in loader:
            bs = len(yb)
            pos      = model(xb.to(device).float()).cpu().numpy()
            prev_pos = model(prev_xb.to(device).float()).cpu().numpy()
            ret      = yb.numpy()
            meta = [dataset.metadata(i) for i in range(cursor, cursor + bs)]
            cursor += bs
            batch = pd.DataFrame(meta)
            batch["variant"] = variant
            batch["fold"]    = fold["fold"]
            batch["test_year"] = fold["test_year"]
            batch["position"]  = pos
            batch["previous_position_model"] = prev_pos
            batch["turnover"]  = np.abs(pos - prev_pos)
            batch["target_return"] = ret
            batch["gross_strategy_return"] = pos * ret
            batch["transaction_cost"] = TRANSACTION_COST_RATE * batch["turnover"]
            batch["strategy_return"]  = batch["gross_strategy_return"] - batch["transaction_cost"]
            rows.append(batch)
    return pd.concat(rows, ignore_index=True)


def _ann_sharpe(values, eps=1e-6):
    v = pd.Series(values).dropna()
    if v.empty:
        return np.nan
    return float(v.mean() / (v.std(ddof=0) + eps) * SQRT_252)


def fold_position_diagnostics(positions, history, variant) -> pd.DataFrame:
    if positions.empty:
        raise ValueError(f"Empty positions for {variant}")
    best = history.loc[history["val_loss"].idxmin()]
    pos = positions["position"].astype(float)
    return pd.DataFrame([{
        "variant": variant, "fold": int(best["fold"]),
        "test_year": int(positions["test_year"].iloc[0]),
        "best_epoch": int(best["best_epoch"]), "n_epochs": int(history["epoch"].max()),
        "val_gross_sharpe": float(best["val_gross_sharpe"]),
        "val_net_sharpe": float(best["val_net_sharpe"]),
        "val_loss": float(best["val_loss"]),
        "test_gross_sharpe": _ann_sharpe(positions["gross_strategy_return"]),
        "test_net_sharpe":   _ann_sharpe(positions["strategy_return"]),
        "mean_turnover":     float(positions["turnover"].mean()),
        "mean_transaction_cost": float(positions["transaction_cost"].mean()),
        "n_train_samples": int(best["n_train_samples"]),
        "n_val_samples":   int(best["n_val_samples"]),
        "n_test_samples":  int(best["n_test_samples"]),
        "n_positions": len(positions),
        "position_mean":   float(pos.mean()), "position_std": float(pos.std(ddof=0)),
        "share_long_strict": float((pos > 0.5).mean()),
        "share_near_cash":   float((pos < 0.1).mean()),
        "position_min":    float(pos.min()), "position_max": float(pos.max()),
        "seconds_total":   float(history["seconds_total"].iloc[-1]),
    }])


def run_variant(frame, folds, variant, checkpoint=False, root=None):
    pos_parts, met_parts, diag_parts = [], [], []
    n = len(folds)
    for fold_row in folds.itertuples(index=False):
        fold = fold_row._asdict() if hasattr(fold_row, "_asdict") else dict(fold_row)
        print(f"[{variant}  fold {fold['fold']}/{n}]  test={fold['test_year']}")
        result = train_one_fold(frame, variant, fold_row, show_diagnostics=(fold["fold"] == 1))
        pos  = predict_test(result, variant)
        hist = result["history"]
        diag = fold_position_diagnostics(pos, hist, variant)
        pos_parts.append(pos); met_parts.append(hist); diag_parts.append(diag)
        if checkpoint and root is not None:
            save_nb03_outputs(root, pd.concat(pos_parts, ignore_index=True),
                              pd.concat(met_parts, ignore_index=True),
                              pd.concat(diag_parts, ignore_index=True))
    return (pd.concat(pos_parts, ignore_index=True) if pos_parts else pd.DataFrame(),
            pd.concat(met_parts, ignore_index=True) if met_parts else pd.DataFrame(),
            pd.concat(diag_parts, ignore_index=True) if diag_parts else pd.DataFrame())


def run_all_variants(frame, folds, variants=None, checkpoint=False, root=None):
    variants = variants or ["baseline", "cpd_cusum", "cpd_gp", "cpd_bocpd"]
    all_pos, all_met, all_diag = [], [], []
    for v in variants:
        pos, met, diag = run_variant(frame, folds, v, checkpoint=checkpoint, root=root)
        all_pos.append(pos); all_met.append(met); all_diag.append(diag)
    return (pd.concat(all_pos, ignore_index=True),
            pd.concat(all_met, ignore_index=True),
            pd.concat(all_diag, ignore_index=True))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def position_diagnostics(positions) -> pd.DataFrame:
    rows = []
    for (variant, fold), group in positions.groupby(["variant", "fold"], sort=False):
        pos = group["position"].astype(float)
        ret = group["target_return"].astype(float)
        rows.append({"variant": variant, "fold": fold,
                     "test_year": int(group["test_year"].iloc[0]),
                     "pearson":  pos.corr(ret, method="pearson") if pos.nunique() > 1 else np.nan,
                     "spearman": pos.corr(ret, method="spearman") if pos.nunique() > 1 else np.nan,
                     "test_gross_sharpe_recomputed": _ann_sharpe(group["gross_strategy_return"]),
                     "test_net_sharpe_recomputed":   _ann_sharpe(group["strategy_return"]),
                     "mean_turnover_recomputed":     float(group["turnover"].mean())})
    return pd.DataFrame(rows)


def enrich_diagnostics(positions, diagnostics) -> pd.DataFrame:
    pred = position_diagnostics(positions)
    drop = ["pearson", "spearman", "test_gross_sharpe_recomputed",
            "test_net_sharpe_recomputed", "mean_turnover_recomputed"]
    return (diagnostics.drop(columns=drop, errors="ignore")
            .merge(pred, on=["variant", "fold", "test_year"], how="left"))


def variant_summary(diagnostics) -> pd.DataFrame:
    return (diagnostics.groupby("variant", as_index=False)
            .agg(folds=("fold", "nunique"),
                 mean_val_net_sharpe=("val_net_sharpe", "mean"),
                 mean_test_net_sharpe=("test_net_sharpe", "mean"),
                 median_test_net_sharpe=("test_net_sharpe", "median"),
                 mean_test_gross_sharpe=("test_gross_sharpe", "mean"),
                 mean_turnover=("mean_turnover", "mean"),
                 mean_transaction_cost=("mean_transaction_cost", "mean"),
                 total_seconds=("seconds_total", "sum"))
            .sort_values("mean_test_net_sharpe", ascending=False)
            .round(4))


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_nb03_outputs(root, positions, metrics, diagnostics) -> pd.DataFrame:
    paths = {"dmn_positions": root / DMN_POSITIONS_PATH,
             "dmn_metrics":   root / DMN_METRICS_PATH,
             "dmn_diagnostics": root / DMN_DIAGNOSTICS_PATH}
    paths["dmn_positions"].parent.mkdir(parents=True, exist_ok=True)
    positions.to_parquet(paths["dmn_positions"], index=False)
    metrics.to_parquet(paths["dmn_metrics"], index=False)
    diagnostics.to_parquet(paths["dmn_diagnostics"], index=False)
    return pd.DataFrame([
        {"output": name, "path": path.relative_to(root).as_posix(),
         "rows": len(frame), "size_mb": round(path.stat().st_size / 1e6, 2)}
        for (name, path), frame in zip(paths.items(), [positions, metrics, diagnostics])
    ])


def load_existing_outputs(root):
    paths = [root / DMN_POSITIONS_PATH, root / DMN_METRICS_PATH, root / DMN_DIAGNOSTICS_PATH]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"NB03 outputs missing: {[p.as_posix() for p in missing]}")
    return tuple(pd.read_parquet(p) for p in paths)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_loss_curves(metrics, variant):
    import plotly.express as px
    df = metrics.loc[metrics["variant"].eq(variant)].melt(
        id_vars=["variant", "fold", "test_year", "epoch"],
        value_vars=["train_loss", "val_loss"], var_name="series", value_name="loss")
    fig = px.line(df, x="epoch", y="loss", color="series", facet_col="test_year",
                  facet_col_wrap=4, markers=True, title=f"Loss curves — {variant}")
    fig.update_layout(height=650, legend_title_text="")
    return fig


def plot_sharpe_by_fold(diagnostics):
    import plotly.express as px
    fig = px.bar(diagnostics, x="test_year", y="test_net_sharpe", color="variant",
                 barmode="group",
                 title=f"Test Sharpe by fold — net of {TRANSACTION_COST_BPS:.0f} bps costs")
    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_layout(height=480, legend_title_text="")
    return fig


def plot_turnover_by_fold(diagnostics):
    import plotly.express as px
    fig = px.bar(diagnostics, x="test_year", y="mean_turnover", color="variant",
                 barmode="group", title="Mean daily turnover by fold")
    fig.update_layout(height=420, legend_title_text="")
    return fig


def plot_position_distribution(positions):
    import plotly.express as px
    fig = px.histogram(positions, x="position", color="variant", facet_col="variant",
                       nbins=50, opacity=0.85, title="Long-only position distributions [0, 1]")
    fig.update_xaxes(range=[0, 1])
    fig.update_layout(height=420, showlegend=False)
    return fig


def plot_example_ticker(positions, frame, known_events):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    variants = ["baseline", "cpd_cusum", "cpd_gp", "cpd_bocpd"]
    variants = [v for v in variants if v in positions["variant"].unique()]

    candidates = sorted(positions.loc[positions["ticker"].str.contains("ASML", case=False, na=False),
                                       "ticker"].unique())

    def has_all(ticker):
        return set(positions.loc[positions["ticker"].eq(ticker), "variant"].unique()) == set(variants)

    ticker = (candidates[0] if candidates and has_all(candidates[0])
              else positions.loc[positions["variant"].eq(variants[-1])]
              .groupby("ticker").size().idxmax())
    label = f"{ticker}{' (GP-covered)' if 'ASML' not in ticker else ''}"

    price = (frame.loc[frame["ticker"].eq(ticker) & frame["date"].ge(pd.Timestamp("2013-01-01")),
                        ["date", "price"]].dropna().sort_values("date"))
    price["idx"] = 100.0 * price["price"] / price["price"].iloc[0]
    pos = positions.loc[positions["ticker"].eq(ticker)].sort_values(["variant", "date"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.4, 0.6])
    fig.add_trace(go.Scatter(x=price["date"], y=price["idx"], mode="lines",
                              name=f"{label} price"), row=1, col=1)
    for v in variants:
        g = pos.loc[pos["variant"].eq(v)]
        fig.add_trace(go.Scatter(x=g["date"], y=g["position"], mode="lines", name=v), row=2, col=1)

    if known_events is not None and not known_events.empty:
        for ev in known_events["event_date"]:
            if price["date"].min() <= ev <= price["date"].max():
                fig.add_vline(x=ev, line=dict(color="rgba(200,0,0,0.15)", width=1, dash="dash"),
                              row="all", col=1)

    fig.update_yaxes(title_text="Price index", row=1, col=1)
    fig.update_yaxes(title_text="Position", range=[0, 1.02], row=2, col=1)
    fig.update_layout(height=700, title=f"DMN positions — {label}", hovermode="x unified")
    return fig
