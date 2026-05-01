# Presentation 1 - Progress And Current Results Snapshot

This note freezes the current state of the project before adding a heavier
LSTM/Sharpe-loss model. It is meant to support the first monthly presentation:
what is already implemented, what the current results say, and what remains to
be improved. It also keeps a trace of the main corrections and decisions made
along the way.

## 0. Progress Log

| Step | What changed | Why it mattered | Current status |
|---|---|---|---|
| Git branch | Work moved and pushed on `justine/submission`. | Vincent explicitly asked everyone to work on a Git branch. | Done. |
| Data source | `01_build_dataset.py` now starts in 2006 and uses yearly CSV files by default, then appends the PRICE ATLAS Excel tail for 2025-2026. | The Excel workbook alone starts in 2013, while Vincent asked for a 2006-to-today backtest. | Done and rebuilt. |
| Long-format panel | Built `stoxx600_processed.csv` with price, returns, volatility, metadata and relative returns. | This matches the requested format: `date`, `ticker`, `price`, features. | Done. |
| Idiosyncratic returns | Added market-relative and sector-relative returns. | Vincent highlighted idiosyncratic shocks as the useful detection target, not only macro shocks. | Done, can be improved with earnings dates later. |
| CPD layer | Implemented fast CPD scores and GP-style reference logic. | The paper's key contribution is the CPD signal; the fast layer makes experiments scalable. | Done, but full 2006-2026 CPD refresh should be checked. |
| Backtest | Added a first rule-based backtest in `04_run_backtest.py`. | This created the evaluation layer needed to compare all future models. | Done. |
| DMN-lite | Added a ridge-based supervised allocation model in `03_train_dmn.py`. | It closes the full pipeline before moving to the heavier LSTM. | Done, preliminary baseline. |
| Walk-forward | DMN-lite uses expanding annual walk-forward folds. | Vincent emphasized walk-forward validation; this avoids random temporal leakage. | Done, validation split still to improve. |
| LSTM DMN | Added `03_train_lstm_dmn.py` with a PyTorch LSTM and differentiable Sharpe loss. | This is the first implementation step that moves the model closer to the paper. | Smoke-tested, full run pending. |
| Notebooks | Added notebooks `03` and `04`; converted main visuals to Plotly. | The presentation needs clear, reproducible outputs and graphs. | Done. |
| Current snapshot | This file freezes results, limitations and next steps. | It keeps a clean trace for the report and PowerPoint. | Done. |

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

4. `scripts/03_train_lstm_dmn.py`
   - First PyTorch LSTM implementation closer to the paper.
   - Rolling stock-level feature sequences.
   - Differentiable negative Sharpe-ratio loss.
   - Same annual expanding walk-forward structure.
   - Output: stock-level positions in `dmn_lstm_positions.csv` after a full run.

5. `04_run_backtest.ipynb` / `scripts/04_run_backtest.py`
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

## 3.1 How The Work Evolved

The project started from the paper structure: slow momentum, fast reversion,
CPD, then a DMN-style allocation model. The first implementation effort focused
on making the data reliable and reproducible before optimizing the model.

The main correction was the data source. Initially, the Excel workbook looked
like the natural source, but it only starts in 2013. Because Vincent asked for a
2006-to-today backtest, the pipeline was corrected to use yearly CSV files for
2006-2024 and the Excel workbook only for the 2025-2026 tail.

The interpretation is now:

- `prices_2006.csv` to `prices_2024.csv` are raw historical price files. They
  are read-only inputs and must not be modified.
- `2025_2026_PRICE_ATLAS_data_sxxr_static.xlsx` is used for the 2025-2026 price
  tail and for the static universe / metadata reference.
- All transformations, feature engineering and cleaning outputs are written
  downstream in `data/processed/stoxx600/`.

This is consistent with Vincent's instruction: the static 2025-2026 workbook is
the universe reference, while the annual CSV files provide the historical price
depth needed for a 2006-to-today empirical backtest.

Latest rebuild check:

- `stoxx600_processed.csv` date range: 2006-01-02 to 2026-04-10.
- Feature rows: 1,705,321.
- Unique tickers observed across the full raw history: 828.
- Duplicate `(date, ticker)` rows: 0.
- The raw annual CSV files remain unchanged; only processed outputs are
  regenerated.

## 3.2 Basic Feature Definitions

The main processed file is `stoxx600_processed.csv`. It is a long-format panel:
one row corresponds to one stock on one date. The important columns are not raw
Bloomberg fields; most of them are features computed from the raw prices.

The return columns measure past price performance over different horizons:

- `1d_arith_ret`: one-day return, computed from yesterday's price to today's
  price.
- `21d_arith_ret`: approximately one trading month.
- `63d_arith_ret`: approximately one trading quarter.
- `126d_arith_ret`: approximately six trading months.
- `252d_arith_ret`: approximately one trading year.

Using several horizons matters because the paper's economic idea combines slow
momentum and fast reversion. Long horizons, such as 126d or 252d, capture slow
trend information. Short horizons, such as 1d or 21d, capture recent shocks or
short-term reversal information.

The volatility columns, such as `20d_vol`, `60d_vol` and `252d_vol`, are also
computed from historical returns. For example, `60d_vol` is the annualized
standard deviation of the stock's recent daily returns over roughly 60 trading
days. These variables help the model distinguish a normal move from a large
move relative to the stock's own risk.

The relative-return columns compare each stock to a benchmark or group on the
same day:

- `1d_ret_vs_ew`: stock return minus the equal-weight STOXX 600 return.
- `1d_ret_vs_sxxr`: stock return minus the SXXR benchmark return.
- `1d_ret_vs_sector`: stock return minus its sector return.
- `1d_ret_vs_country` and `1d_ret_vs_region`: stock return minus its local
  group return.

These columns are useful for idiosyncratic shocks. A stock may fall because the
whole market falls, or because something specific happened to that stock. The
relative-return variables help separate stock-specific moves from macro or
sector-wide moves.

Columns ending in `_lag1` are shifted by one trading day. This means the model
uses yesterday's observed information to make today's decision. This is
important to avoid lookahead bias: the strategy must not use information from a
date before that information would have been observable in real time.

For supervised training, the scripts also create next-day targets internally.
The model observes features available at date `t`, predicts a position for date
`t+1`, and the backtest evaluates that position on the realized return at
`t+1`. This keeps the timing of information, prediction and PnL consistent.

The second correction was methodological. A CPD detector alone is not enough:
we need to test whether CPD improves actual portfolio PnL. This is why the
backtest layer was implemented before the final LSTM. It gives us a stable
evaluation framework.

The third correction was validation. Instead of random train/test splits, the
current supervised model uses yearly walk-forward folds. This is closer to how
the strategy would be evaluated in practice and matches Vincent's comments.

The current DMN-lite model should therefore be understood as a bridge: it is
not the final paper-style LSTM, but it proves that the pipeline can transform
daily data and CPD features into out-of-sample positions and measurable
backtest results.

The first LSTM/Sharpe-loss implementation has now been added as a separate
script, not as a replacement for DMN-lite. This keeps the baseline comparable
while allowing the project to move toward the paper's architecture. The next
step is to run the LSTM on the full intended universe/date range, then backtest
`dmn_lstm_positions.csv` next to the existing strategies.

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

The next step is to turn the new LSTM script into a full result:

1. Confirm that `stoxx600_processed.csv` really covers 2006-01-02 to 2026-04-10.
2. Run `03_train_lstm_dmn.py` on the full universe.
3. Backtest `dmn_lstm_positions.csv`.
4. Run an ablation without CPD using `--no-cpd`.
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
