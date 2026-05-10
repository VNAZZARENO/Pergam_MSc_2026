# Pergam MSc 2026 - STOXX Europe 600

Clean research pipeline for the Pergam adaptation of *Slow Momentum with Fast
Reversion* on STOXX Europe 600 equities.

Only three notebooks are part of the submission pipeline:

1. `notebooks/01_data_loading.ipynb`
2. `notebooks/02_changepoint_detection.ipynb`
3. `notebooks/03_train_dmn.ipynb`

## Scope

- Universe: STOXX Europe 600 equities from the static 2025-2026 SXXR workbook.
- Window: 2006 to the latest available workbook date.
- Frequency: daily.
- Data shape: raw workbook prices are wide, then converted to long format
  `(date, ticker, price, features...)`.
- Denoising: NB01 creates cleaned raw returns, market-relative returns, and
  sector-relative returns.
- Known events: macro shocks only, used as chronological references and simple
  sanity checks. The useful CPD target is idiosyncratic shocks where no macro
  event is known.
- Portfolio constraint in NB03: long-only positions in `[0, 1]`, no short, no leverage.
- Transaction costs in NB03: 25 bps applied directly in the training loss and diagnostics.
- Model comparison in NB03: baseline LSTM, LSTM + CUSUM CPD, LSTM + GP CPD.
- Validation: walk-forward folds from 2019 to 2026.

## Notebook Roles

NB01 builds the data foundation from the SXXR workbook. It documents the
wide-to-long transformation, the static universe policy, the macro event
timeline, the cleaned/denoised return definitions, and the anti-lookahead
checks.

NB02 computes CPD scores. It compares several causal CPD methods, uses known
macro events only as a reference metric, and exports the CPD feature consumed by
NB03.

NB03 trains the long-only LSTM Deep Momentum Network. It compares baseline,
CUSUM-CPD, and GP-CPD variants with 25 bps transaction costs inside the loss.

## Local Data

The raw workbook is local and must not be pushed. Place it here before running
the notebooks:

```text
data/raw/stoxx600/SXXR.xlsx
```

Generated parquet/csv outputs under `data/processed/` are also ignored by Git.

## Run Order

Run from the repository root:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_loading.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_changepoint_detection.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_train_dmn.ipynb
```

## Pipeline Outputs

Notebook 01 builds the clean modelling panel:

- `data/processed/stoxx600/panel.parquet`
- `data/processed/stoxx600/panel_long.parquet`
- `data/processed/stoxx600/features_panel.parquet`
- `data/processed/stoxx600/benchmark_ew.parquet`
- `data/processed/stoxx600/universe_static.parquet`
- `data/processed/stoxx600/universe_pit.parquet`
- `data/processed/stoxx600/static_universe_coverage.csv`
- `data/processed/stoxx600/known_events.csv`

Notebook 02 computes change-point detection scores:

- `data/processed/stoxx600/cpd_scores.parquet`
- `data/processed/stoxx600/cpd_metrics.parquet`
- `data/processed/stoxx600/cpd_features_nb03.parquet`

Notebook 03 trains and compares the DMN variants:

- `data/processed/stoxx600/dmn_positions.parquet`
- `data/processed/stoxx600/dmn_metrics.parquet`
- `data/processed/stoxx600/dmn_diagnostics.parquet`

## Submission Notes

The repository is intentionally trimmed to the three-notebook flow. Exploratory
work, generated HTML, figures, report exports, and obsolete source modules were
removed from the clean version.
