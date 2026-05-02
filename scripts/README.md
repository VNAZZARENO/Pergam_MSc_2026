# scripts

This folder contains the executable steps of the project pipeline.

## Part 01 - Data loading and feature building

`01_build_dataset.py` uses the static 2025-2026 PRICE ATLAS workbook as the
working universe by default. The yearly CSV files are used only to backfill
prices before the Excel workbook starts, so the backtest can still begin in
2006. This matches the current Presentation 1 policy: static universe first,
bias-free historical universe construction left as a limitation/future step.

Example:

```bash
python scripts/01_build_dataset.py
```

Main outputs:

- `data/processed/stoxx600/prices_clean.csv`
- `data/processed/stoxx600/returns_1d.csv`
- `data/processed/stoxx600/prices_panel.csv`
- `data/processed/stoxx600/features_panel.csv`
- `data/processed/stoxx600/stoxx600_processed.csv`
- `data/processed/stoxx600/sector_returns.csv`
- `data/processed/stoxx600/source_policy_summary.csv`
- `data/processed/stoxx600/static_universe_coverage.csv`

## Part 02 - Changepoint detection

`02_compute_cpd.py` reads the processed panel from part 01 and computes
changepoint scores. The default stock-level series is now `stock_vs_sector`,
because Vincent's feedback makes idiosyncratic shocks the focus. Raw and
market-relative returns remain available with `--stock-series` for diagnostics.

The default method is the fast continuous version:

- CUSUM score
- jump score
- rolling t-test score
- moving-average score
- ensemble score

Example:

```bash
python scripts/02_compute_cpd.py --all-tickers
```

Main outputs:

- `data/processed/stoxx600/cpd_scores_fast.csv`
- `data/processed/stoxx600/cpd_summary_fast.csv`

BOCPD can be added as a targeted continuous-score experiment:

```bash
python scripts/02_compute_cpd.py --include-bocpd --max-tickers 2 --stock-series stock_vs_sector
```

The slower paper-style GP changepoint method can be tested on a small sample:

```bash
python scripts/02_compute_cpd.py --method gp --max-tickers 2 --stride 20
```

## Part 03 - DMN / LSTM training

`03_train_dmn.py` trains the lightweight DMN-lite baseline. It is a ridge
allocation model, not the final paper LSTM, but it keeps the same walk-forward
logic and exports stock-level positions.

`03_train_lstm_dmn.py` trains the first paper-style LSTM with a differentiable
Sharpe loss. It is intentionally CPU-feasible and should be presented as a first
working implementation, not a fully tuned final network.

The no-CPD ablation can be produced with:

```bash
python scripts/03_train_lstm_dmn.py --no-cpd --out-positions dmn_lstm_no_cpd_positions.csv --out-folds dmn_lstm_no_cpd_fold_summary.csv
```

## Part 04 - Backtest

`04_run_backtest.py` evaluates rule-based signals, optional model positions and
raw EW/SXXR benchmarks in one comparable output.

Example:

```bash
python scripts/04_run_backtest.py --positions-file dmn_lite_positions.csv --position-col dmn_lite_position
```

`05_build_final_comparison.py` merges the separate model backtests into one
presentation table:

```bash
python scripts/05_build_final_comparison.py
```
