# scripts

This folder contains the executable steps of the project pipeline.

## Part 01 - Data loading and feature building

`01_build_dataset.py` loads the same PRICE ATLAS workbook used in notebook 01 by
default, cleans prices, computes daily returns and creates a tidy feature panel.
The older annual CSV rebuild remains available with `--price-source yearly`.

Example:

```bash
python scripts/01_build_dataset.py --max-tickers 20
```

Main outputs:

- `data/processed/stoxx600/prices_clean.csv`
- `data/processed/stoxx600/returns_1d.csv`
- `data/processed/stoxx600/prices_panel.csv`
- `data/processed/stoxx600/features_panel.csv`
- `data/processed/stoxx600/stoxx600_processed.csv`
- `data/processed/stoxx600/sector_returns.csv`

## Part 02 - Changepoint detection

`02_compute_cpd.py` reads the processed panel from part 01 and computes
changepoint scores on several economically distinct series:

- raw stock returns as the baseline;
- stock returns relative to the STOXX 600 equal-weight benchmark;
- stock returns relative to their own equal-weight sector;
- equal-weight sector return series for sector-level CPD.

The default method is the fast continuous version:

- CUSUM score
- jump score
- rolling t-test score
- moving-average score
- ensemble score

Example:

```bash
python scripts/02_compute_cpd.py
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

## Parts not started yet

`03_train_dmn.py` and `04_run_backtest.py` are still placeholders. They are kept
only to show the next steps of the full research pipeline.
