# Presentation 1 - Current Results Snapshot

This note freezes the current state of the project before adding a heavier
LSTM/Sharpe-loss model. It is meant to support the first monthly presentation:
what is already implemented, what the current results say, and what remains to
be improved.

## 1. Scope Implemented So Far

The current pipeline covers the full research chain:

1. `01_data_loading.ipynb` / `scripts/01_build_dataset.py`
   - STOXX 600 stock prices.
   - Period: 2006-01-02 to 2026-04-10.
   - Long-format panel: `date`, `ticker`, `price`, returns, volatility and metadata.
   - Market-relative and sector-relative returns for idiosyncratic shock analysis.

2. `02_changepoint_detection.ipynb` / `scripts/02_compute_cpd.py`
   - CPD methods: CUSUM, jump score, rolling t-test, BOCPD option, GP-style CPD reference.
   - CPD scores on raw stock returns, market-relative returns, sector-relative returns and sector-level series.

3. `03_train_dmn.ipynb` / `scripts/03_train_dmn.py`
   - First supervised model layer: `DMN-lite`.
   - Expanding walk-forward training by test year.
   - Features: momentum, volatility, market/sector relative returns, CPD score.
   - Output: stock-level positions in `dmn_lite_positions.csv`.

4. `04_run_backtest.ipynb` / `scripts/04_run_backtest.py`
   - Backtest of rule-based baselines and DMN-lite positions.
   - Metrics: annual return, annual volatility, Sharpe, Sortino, Calmar, max drawdown, hit ratio, average assets, turnover.

## 2. Current Backtest Results

Source file: `data/processed/stoxx600/backtest_summary_with_dmn.csv`

| Strategy | Period | Ann. Return | Ann. Vol | Sharpe | Sortino | Calmar | Max Drawdown | Hit Ratio | Avg Assets | Avg Turnover |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DMN-lite | 2010-01-04 to 2026-04-10 | 3.08% | 7.69% | 0.43 | 0.54 | 0.15 | -20.52% | 53.27% | 317.89 | 3.36% |
| Slow momentum | 2006-12-21 to 2026-04-10 | 3.75% | 14.73% | 0.32 | 0.42 | 0.11 | -34.29% | 53.00% | 251.34 | 4.19% |
| CPD-adjusted | 2006-12-21 to 2026-04-10 | 3.69% | 14.64% | 0.32 | 0.41 | 0.11 | -33.86% | 52.96% | 251.29 | 4.25% |
| Slow + fast | 2006-12-21 to 2026-04-10 | 1.68% | 12.91% | 0.19 | 0.24 | 0.05 | -33.68% | 53.30% | 251.29 | 4.53% |

## 3. Main Interpretation

The current results are encouraging but preliminary.

- DMN-lite has the highest Sharpe ratio in the current comparison.
- DMN-lite also has the smallest max drawdown.
- The rule-based CPD-adjusted strategy is close to slow momentum, but does not yet produce a strong improvement.
- This suggests that CPD information is useful as a model feature, but the simple hand-built CPD allocation rule is probably too crude.
- The next important question is whether the paper-style LSTM/DMN can extract more value from the same CPD and momentum features.

## 4. Walk-Forward Status

The current DMN-lite model already uses an expanding walk-forward protocol:

- first test year: 2010;
- train set always ends before the test year starts;
- test folds are yearly;
- latest fold: train 2006-03-27 to 2025-12-31, test 2026-01-01 to 2026-04-10.

This is aligned with Vincent's emphasis on walk-forward validation. The next
improvement is to add a separate validation block inside each fold for
hyperparameter selection and early stopping.

## 5. Important Limitations To State Clearly

1. The current `DMN-lite` is not the full LSTM from the paper.
   - It is a ridge-based supervised allocation layer.
   - It closes the full pipeline, but does not fully reproduce the model architecture.

2. The paper's original universe is futures, while this project uses STOXX 600 equities.
   - Results are not directly comparable to the paper's reported Sharpe ratios.

3. The CPD rule-based strategy is still simple.
   - Better use of CPD may require a learned model or a more persistent regime layer.

4. The current portfolio construction is still simplified.
   - Long-only, benchmark-aware and tracking-error-aware construction remains a future extension.

## 6. Next Technical Step

The next step should be a closer implementation of the paper:

1. LSTM DMN without CPD.
2. LSTM DMN with CPD.
3. Negative Sharpe-ratio loss.
4. Same walk-forward protocol.
5. Compare:
   - slow momentum;
   - CPD-adjusted rule-based;
   - DMN-lite;
   - LSTM without CPD;
   - LSTM with CPD.

The key presentation message is:

> We first built a complete and reproducible pipeline. The next step is to
> replace the DMN-lite model by the paper-style LSTM trained with a Sharpe loss,
> then test whether CPD improves performance out of sample.
