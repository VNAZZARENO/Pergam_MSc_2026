"""Train a paper-style LSTM DMN with a differentiable Sharpe loss.

This script sits next to ``03_train_dmn.py``. The existing script remains the
fast DMN-lite ridge baseline; this one is the heavier PyTorch experiment that
is closer to the paper: rolling feature sequences -> LSTM -> bounded positions
-> negative Sharpe-ratio loss -> walk-forward out-of-sample positions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.model import DeepMomentumNetwork, set_torch_seed, torch_sharpe_loss


BASE_FEATURE_COLS = [
    "21d_arith_ret",
    "63d_arith_ret",
    "126d_arith_ret",
    "252d_arith_ret",
    "20d_vol",
    "60d_vol",
    "252d_vol",
    "1d_ret_vs_ew_lag1",
    "1d_ret_vs_sector_lag1",
]
CPD_FEATURE_COL = "ensemble_score"
RETURN_COL = "1d_arith_ret"
POSITION_COL = "dmn_lstm_position"


class StockSequenceDataset(Dataset):
    """Lazy rolling-window dataset over a sorted ticker/date panel."""

    def __init__(self, features, returns, end_indices, sequence_length):
        self.features = features
        self.returns = returns
        self.end_indices = np.asarray(end_indices, dtype=np.int64)
        self.sequence_length = int(sequence_length)

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, item):
        end = int(self.end_indices[item])
        start = end - self.sequence_length + 1
        x = torch.from_numpy(self.features[start : end + 1])
        y = torch.tensor(self.returns[end], dtype=torch.float32)
        return x, y


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an LSTM DMN with Sharpe loss and walk-forward folds.",
    )
    parser.add_argument("--data-dir", default="data/processed/stoxx600")
    parser.add_argument("--panel-file", default="stoxx600_processed.csv")
    parser.add_argument("--cpd-file", default="cpd_scores_fast.csv")
    parser.add_argument("--out-positions", default="dmn_lstm_positions.csv")
    parser.add_argument("--out-folds", default="dmn_lstm_fold_summary.csv")
    parser.add_argument("--first-test-year", type=int, default=2010)
    parser.add_argument("--last-test-year", type=int, default=2026)
    parser.add_argument("--sequence-length", type=int, default=63)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-sequences", type=int, default=80_000)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--no-cpd",
        action="store_true",
        help="Train without the CPD feature, useful for an ablation run.",
    )
    return parser


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_feature_columns(panel):
    """Create missing momentum/volatility features from available prices."""
    out = panel.sort_values(["ticker", "date"]).copy()
    grouped = out.groupby("ticker", sort=False)

    if "price" in out.columns:
        for window in [21, 63, 126, 252]:
            col = f"{window}d_arith_ret"
            if col not in out.columns:
                out[col] = grouped["price"].pct_change(window)
        if RETURN_COL not in out.columns:
            out[RETURN_COL] = grouped["price"].pct_change()

    if "21d_arith_ret" not in out.columns and "20d_arith_ret" in out.columns:
        out["21d_arith_ret"] = out["20d_arith_ret"]

    if RETURN_COL in out.columns:
        returns = pd.to_numeric(out[RETURN_COL], errors="coerce")
        for window in [20, 60, 252]:
            col = f"{window}d_vol"
            if col not in out.columns:
                out[col] = (
                    returns.groupby(out["ticker"], sort=False)
                    .rolling(window, min_periods=max(5, window // 4))
                    .std()
                    .reset_index(level=0, drop=True)
                    * np.sqrt(252.0)
                )

    for col in BASE_FEATURE_COLS:
        if col not in out.columns:
            out[col] = 0.0
    return out


def load_panel(data_dir, panel_file, max_tickers=None):
    """Load the STOXX panel and construct any missing model features."""
    path = data_dir / panel_file
    available_cols = set(pd.read_csv(path, nrows=0).columns)
    candidate_cols = [
        "date",
        "ticker",
        "price",
        RETURN_COL,
        "20d_arith_ret",
        "21d_arith_ret",
        "63d_arith_ret",
        "126d_arith_ret",
        "252d_arith_ret",
        "20d_vol",
        "60d_vol",
        "252d_vol",
        "1d_ret_vs_ew_lag1",
        "1d_ret_vs_sector_lag1",
    ]
    usecols = [col for col in candidate_cols if col in available_cols]
    panel = pd.read_csv(path, usecols=usecols, parse_dates=["date"])
    panel = panel.dropna(subset=["ticker"]).sort_values(["ticker", "date"])
    if max_tickers is not None:
        tickers = list(dict.fromkeys(panel["ticker"].astype(str)))[:max_tickers]
        panel = panel.loc[panel["ticker"].isin(tickers)]
    return ensure_feature_columns(panel)


def load_cpd_scores(data_dir, cpd_file):
    """Load stock-level sector-relative CPD scores when available."""
    path = data_dir / cpd_file
    if not path.exists():
        return pd.DataFrame(columns=["date", "ticker", CPD_FEATURE_COL])

    cpd = pd.read_csv(
        path,
        usecols=["date", "scope", "ticker", "series_type", CPD_FEATURE_COL],
        parse_dates=["date"],
        dtype={"scope": "string", "ticker": "string", "series_type": "string"},
        low_memory=False,
    )
    cpd = cpd.loc[
        (cpd["scope"] == "stock")
        & (cpd["series_type"] == "stock_vs_sector")
        & cpd["ticker"].notna()
    ]
    return cpd[["date", "ticker", CPD_FEATURE_COL]].drop_duplicates(
        subset=["date", "ticker"],
        keep="last",
    )


def make_supervised_frame(panel):
    """Create next-day returns and the date where the predicted position is used."""
    out = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = out.groupby("ticker", sort=False)
    out["position_date"] = grouped["date"].shift(-1)
    out["next_return"] = grouped[RETURN_COL].shift(-1)
    return out.dropna(subset=["position_date"]).reset_index(drop=True)


def build_end_indices(frame, mask, sequence_length):
    """Return valid sequence end locations for a row-level mask."""
    valid = mask.to_numpy(dtype=bool) & np.isfinite(frame["next_return"].to_numpy(dtype=float))
    end_indices = []
    for _, group in frame.groupby("ticker", sort=False):
        idx = group.index.to_numpy()
        if len(idx) < sequence_length:
            continue
        group_valid = valid[idx]
        possible = idx[sequence_length - 1 :]
        possible_valid = group_valid[sequence_length - 1 :]
        end_indices.extend(possible[possible_valid])
    return np.asarray(end_indices, dtype=np.int64)


def standardize_features(frame, feature_cols, train_mask):
    """Standardize features using training history only."""
    x = frame[feature_cols].apply(pd.to_numeric, errors="coerce")
    train_x = x.loc[train_mask]
    mean = train_x.mean()
    std = train_x.std().replace(0.0, 1.0)
    z = ((x - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.ascontiguousarray(z.to_numpy(dtype=np.float32))


def train_fold(frame, feature_cols, year, args, rng):
    """Train one expanding fold and return its positions plus diagnostics."""
    test_start = pd.Timestamp(year=year, month=1, day=1)
    test_end = pd.Timestamp(year=year, month=12, day=31)
    train_mask = frame["date"] < test_start
    test_mask = (frame["position_date"] >= test_start) & (frame["position_date"] <= test_end)

    train_end_indices = build_end_indices(frame, train_mask, args.sequence_length)
    test_end_indices = build_end_indices(frame, test_mask, args.sequence_length)
    if len(train_end_indices) == 0 or len(test_end_indices) == 0:
        return None, None

    if args.max_train_sequences and len(train_end_indices) > args.max_train_sequences:
        train_end_indices = rng.choice(
            train_end_indices,
            size=args.max_train_sequences,
            replace=False,
        )

    features = standardize_features(frame, feature_cols, train_mask)
    returns = frame["next_return"].to_numpy(dtype=np.float32)

    train_ds = StockSequenceDataset(
        features=features,
        returns=returns,
        end_indices=train_end_indices,
        sequence_length=args.sequence_length,
    )
    test_ds = StockSequenceDataset(
        features=features,
        returns=returns,
        end_indices=test_end_indices,
        sequence_length=args.sequence_length,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = DeepMomentumNetwork(
        n_features=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    model.train()
    epoch_losses = []
    for _ in range(args.epochs):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            positions = model(x_batch)
            loss = torch_sharpe_loss(positions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(batch_losses)))

    model.eval()
    predicted = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            predicted.append(model(x_batch).cpu().numpy())
    predicted = np.concatenate(predicted)

    fold_positions = frame.loc[
        test_end_indices,
        ["position_date", "ticker", "next_return"],
    ].copy()
    fold_positions[POSITION_COL] = predicted
    fold_positions["fold_year"] = year
    fold_positions = fold_positions.rename(
        columns={"position_date": "date", "next_return": "realized_return"}
    )

    fold_summary = {
        "fold_year": year,
        "train_sequences": int(len(train_end_indices)),
        "test_sequences": int(len(test_end_indices)),
        "train_start": frame.loc[train_mask, "date"].min(),
        "train_end": frame.loc[train_mask, "date"].max(),
        "test_start": fold_positions["date"].min(),
        "test_end": fold_positions["date"].max(),
        "epochs": int(args.epochs),
        "final_train_sharpe_loss": float(epoch_losses[-1]),
        "avg_abs_position": float(np.abs(fold_positions[POSITION_COL]).mean()),
    }
    return fold_positions, fold_summary


def train_walk_forward(frame, feature_cols, args):
    """Run annual expanding-window LSTM training."""
    rng = np.random.default_rng(args.seed)
    positions = []
    summaries = []
    for year in range(args.first_test_year, args.last_test_year + 1):
        fold_positions, fold_summary = train_fold(frame, feature_cols, year, args, rng)
        if fold_positions is None:
            continue
        positions.append(fold_positions)
        summaries.append(fold_summary)
        print(
            f"{year}: train={fold_summary['train_sequences']:,}, "
            f"test={fold_summary['test_sequences']:,}, "
            f"loss={fold_summary['final_train_sharpe_loss']:.4f}"
        )

    if not positions:
        raise ValueError("No LSTM walk-forward folds were produced.")
    return pd.concat(positions, ignore_index=True), pd.DataFrame(summaries)


def main() -> None:
    args = build_parser().parse_args()
    set_torch_seed(args.seed)
    torch.set_num_threads(max(1, int(args.torch_threads)))

    data_dir = _resolve(args.data_dir)
    panel = load_panel(data_dir, args.panel_file, max_tickers=args.max_tickers)
    feature_cols = list(BASE_FEATURE_COLS)
    if not args.no_cpd:
        cpd = load_cpd_scores(data_dir, args.cpd_file)
        panel = panel.merge(cpd, on=["date", "ticker"], how="left")
        panel[CPD_FEATURE_COL] = panel[CPD_FEATURE_COL].fillna(0.0)
        feature_cols.append(CPD_FEATURE_COL)

    supervised = make_supervised_frame(panel)
    positions, folds = train_walk_forward(supervised, feature_cols, args)

    positions_path = data_dir / args.out_positions
    folds_path = data_dir / args.out_folds
    positions.to_csv(positions_path, index=False)
    folds.to_csv(folds_path, index=False)

    print("LSTM DMN training finished")
    print(f"Rows loaded: {len(panel):,}")
    print(f"Tickers: {panel['ticker'].nunique():,}")
    print(f"Features: {', '.join(feature_cols)}")
    print(f"Position date range: {positions['date'].min().date()} -> {positions['date'].max().date()}")
    print(f"Positions: {positions_path}")
    print(f"Fold summary: {folds_path}")
    print(folds.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
