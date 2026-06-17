# Pergam MSc 2026 - STOXX Europe 600

Clean research pipeline for the Pergam adaptation of *Slow Momentum with Fast
Reversion* on STOXX Europe 600 equities.

Five notebooks form the submission pipeline:

1. `notebooks/01_data_loading.ipynb`
2. `notebooks/02_changepoint_detection.ipynb`
3. `notebooks/03_train_dmn.ipynb` — LSTM model (V1, Edoardo)
4. `notebooks/03_train_dmn_v2.ipynb` — LightGBM model (V2, Justine)
5. `notebooks/04_backtest.ipynb`

## Scope

- Universe: STOXX Europe 600 equities from the static 2025-2026 SXXR workbook.
- Window: 2006 to the latest available workbook date.
- Frequency: daily.
- Data shape: raw workbook prices are wide, then converted to long format
  `(date, ticker, price, features...)`.
- Denoising: NB01 creates cleaned raw returns, market-relative returns, and
  sector-relative returns.
- Known events: macro shocks only, used as chronological references and simple
  sanity checks.
- Transaction costs: 25 bps applied at backtest time (NB04) to all variants.

## Notebook Roles

NB01 builds the data foundation from the SXXR workbook. It documents the
wide-to-long transformation, the static universe policy, the macro event
timeline, the cleaned/denoised return definitions, and the anti-lookahead
checks.

NB02 computes CPD scores. It compares six causal CPD methods, uses known
macro events as a reference metric, and exports the CUSUM CPD feature consumed
by NB03 and NB03-V2.

NB03 (V1) trains the long-only LSTM Deep Momentum Network. It uses an
expanding walk-forward protocol with 3 folds of 5 years (2011-2025), comparing
baseline, CUSUM-CPD, and cost-aware variants.

NB03-V2 (V2) trains an alternative LightGBM model. It uses 22 features
(momentum + region + CPD), calibrates positions via sigmoid(alpha* x score)
with EMA smoothing (halflife=10d) and CPD filter. Walk-forward: 16 annual folds
(2011-2026). Alpha is calibrated to maximise net Sharpe after 25 bps TC.
Includes SHAP interpretability analysis.

NB04 aggregates both models' stock-level out-of-sample positions into portfolio
returns, applies 25 bps transaction costs uniformly, and compares all variants
against EW/SXXR/TSMOM/MACD benchmarks.

## Local Data

The raw workbook is local and must not be pushed. Place it here before running
the notebooks:

```text
data/raw/stoxx600/SXXR.xlsx
```

Generated parquet/csv outputs under `data/processed/` are also ignored by Git.

## Run Order

Run from the repository root in this order:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_loading.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_changepoint_detection.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_train_dmn.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_train_dmn_v2.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_backtest.ipynb
```

To export NB03-V2 as standalone HTML (outputs only, no source code):

```bash
jupyter nbconvert --to html --no-input notebooks/03_train_dmn_v2.ipynb
```

## Pipeline Outputs

Notebook 01 builds the clean modelling panel:

- `data/processed/stoxx600/stoxx600_processed.csv`
- `data/processed/stoxx600/benchmark_stoxx600_ew.csv`

Notebook 02 computes change-point detection scores:

- `data/processed/stoxx600/cpd_scores.parquet`
- `data/processed/stoxx600/cpd_metrics.parquet`
- `data/processed/stoxx600/cpd_features_nb03.parquet`

Notebook 03 (V1 LSTM) trains and compares the DMN variants:

- `data/processed/stoxx600/dmn_positions.parquet`
- `data/processed/stoxx600/dmn_metrics.parquet`
- `data/processed/stoxx600/dmn_diagnostics.parquet`
- `data/processed/models/predictions_fold*_expanding_*.csv`

Notebook 03-V2 (LightGBM) trains the alternative model:

- `data/processed/stoxx600/positions_v2.parquet`
- `data/processed/stoxx600/fold_metrics_v2.parquet`

Notebook 04 aggregates portfolio-level backtest results:

- `data/processed/stoxx600/backtest_portfolio.parquet`
- `data/processed/stoxx600/backtest_metrics.parquet`

## Submission Notes

The repository is intentionally trimmed to the notebook submission flow.
Exploratory work, generated HTML, figures, report exports, and obsolete source
modules were removed from the clean version.
