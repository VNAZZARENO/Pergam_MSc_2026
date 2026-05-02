# Slow Momentum with Fast Reversion

A Python implementation and extension of the trading strategy described in:

> **Wood, K., Roberts, S., & Zohren, S. (2022).**
> *Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection.*
> The Journal of Financial Data Science, Winter 2022.

This project is a statistical / quantitative research work carried out at ESILV
(Pergam MSc 2026). The goal is to create investment signals from the paper,
implement them in Python, optimize and improve the model, and explore relevant
extensions.

---

## Current Project Adaptation

The paper is the methodological reference, but the current ESILV/Pergam
implementation is adapted from futures to STOXX Europe 600 equities.

- Universe reference: `data/raw/stoxx600/2025_2026_PRICE_ATLAS_data_sxxr_static.xlsx`.
- Historical backfill: yearly CSV files are used for the pre-2013 history.
- Main price source: the PRICE ATLAS Excel workbook is used from 2013 onward,
  because it is the requested static source and has cleaner coverage for current
  STOXX 600 names.
- Production panel: `data/processed/stoxx600/stoxx600_processed.csv`, in long
  format with `date`, `ticker`, `price` and engineered features.

This keeps one coherent price panel for CPD, model training and backtesting:
CSV files extend the history back to 2006, while the Excel workbook remains the
primary source whenever it is available.

---

## Project Context

Time-series momentum (TSMOM) strategies exploit the empirical fact that strong
price trends tend to persist. They are a core building block of Commodity
Trading Advisors (CTAs) and alternative investment funds.

However, classical momentum strategies (and even recent deep-learning variants)
have **underperformed in recent years**, mostly because they react too slowly
around *momentum turning points* — moments when an uptrend suddenly flips into
a downtrend (or vice versa), such as during the 2020 COVID crash.

The paper proposes a hybrid pipeline that mixes:

1. A **slow momentum** signal that captures persistent trends.
2. A **fast mean-reversion** signal that exploits localized price moves.
3. An **online Changepoint Detection (CPD) module** based on Gaussian Processes
   that tells the model *when* and *how strongly* the regime is changing, so
   that the balance between the two signals is learned in a data-driven way.

The CPD output is fed into a **Deep Momentum Network (DMN)** — an LSTM trained
directly on the Sharpe ratio as loss function — which outputs the position to
hold for each asset.

Reported results: adding the CPD module yields a **+33% improvement in Sharpe
ratio** over the LSTM baseline from 1995–2020, and roughly **+66%** over the
most recent (and more turbulent) 2015–2020 period.

---

## Core Ideas of the Paper

### 1. Changepoint Detection via Gaussian Processes

- Daily returns are standardized over a lookback window (LBW) `l`.
- A Gaussian Process regression is fit with a **Matérn 3/2 kernel** (well suited
  to noisy, non-smooth financial data).
- A **changepoint kernel** is defined as a smooth sigmoid blend between two
  Matérn 3/2 kernels, one on each side of an unknown changepoint location `c`.
- Hyperparameters are fitted by minimizing the negative log marginal likelihood
  (L-BFGS-B via `GPflow` / `scipy.optimize`).
- Two normalized scalars are extracted per asset per day:
  - **Severity** `ν ∈ (0, 1)`: how much the changepoint kernel improves the
    likelihood vs. a single Matérn kernel.
  - **Location** `γ ∈ (0, 1)`: where the detected changepoint sits within the
    lookback window.

### 2. Deep Momentum Network (LSTM)

- Architecture: LSTM + time-distributed dense layer with `tanh` activation,
  directly outputting a position `X ∈ (-1, 1)`.
- **Loss function:** negative annualized Sharpe ratio — the network is trained
  to maximize risk-adjusted return, not to predict direction.
- Inputs per asset per day:
  - Normalized returns at several horizons (1, 21, 63, 126, 252 days).
  - MACD indicators with pairs `{(8, 24), (16, 28), (32, 96)}`.
  - CPD **severity** and **location** for the chosen LBW.
- Volatility scaling targets an annualized vol of **15%** per asset using a
  60-day EWM standard deviation.

### 3. Backtesting Protocol

- **Universe:** 50 liquid continuous futures (commodities, equities, fixed
  income, FX) from Pinnacle Data Corp, 1990–2020.
- **Expanding window:** train on 1990–1995, test 1995–2000, then roll forward
  every 5 years, re-optimizing hyperparameters at each step.
- **Benchmarks:** Long only, MACD, TSMOM (Moskowitz `w=0`, blended `w=0.5`,
  short `w=1`), and a plain LSTM DMN without CPD.
- **Metrics:** annualized return, volatility, Sharpe, Sortino, Calmar, max
  drawdown, % of positive returns, and avg profit / avg loss.

### Reported Benchmark (rescaled to 15% target vol)

| Strategy                 | Return  | Sharpe | Sortino | Calmar |
|--------------------------|--------:|-------:|--------:|-------:|
| Long Only                |  6.62%  | 0.44   | 0.64    | 0.79   |
| MACD                     | 11.08%  | 0.77   | 1.09    | 0.95   |
| TSMOM (w = 0)            | 13.79%  | 0.94   | 1.32    | 1.35   |
| LSTM (DMN, no CPD)       | 21.03%  | 1.62   | 2.46    | 2.79   |
| **LSTM + CPD (21d LBW)** | 30.57%  | 2.04   | 3.07    | 3.75   |
| **LSTM + CPD (opt. LBW)**| **31.52%** | **2.16** | **3.33** | **3.50** |

---

## Repository Layout

```
Pergam_MSc_2026/
├── README.md
├── requirements.txt            # Planned Python dependencies
├── documentation/              # Reference paper (PDF) and notes
├── configs/                    # YAML configs (assets, horizons, hyperparams)
│   └── default.yaml
├── data/                       # Gitignored
│   ├── raw/                    # Original Pinnacle / futures CSVs
│   └── processed/              # Cleaned returns, features, CPD outputs
├── src/                        # Library code
│   ├── data_loader.py          # Load STOXX 600 CSV/Excel prices
│   ├── preprocessing.py        # Returns, EWM vol, vol scaling
│   ├── features.py             # Normalized returns + MACD
│   ├── cpd.py                  # GP Matérn 3/2 + changepoint kernel
│   ├── model.py                # LSTM DMN + Sharpe loss
│   ├── backtest.py             # Expanding-window backtest harness
│   └── metrics.py              # Sharpe, Sortino, Calmar, MDD, hit ratio
├── scripts/                    # Thin CLI entry points
│   ├── 01_build_dataset.py
│   ├── 02_compute_cpd.py
│   ├── 03_train_dmn.py
│   ├── 03_train_lstm_dmn.py
│   ├── 04_run_backtest.py
│   └── 05_build_final_comparison.py
└── notebooks/
    └── 00_exploration.ipynb
```

The main `src/` modules and `scripts/` now contain working implementations for
the first presentation: data loading, feature construction, CPD scores,
DMN-style model experiments and backtesting.

---

## Current Status And Next Work

1. **Data pipeline**
   - Working long-format STOXX 600 panel from 2006-01-02 to 2026-04-10.
   - Static 2025-2026 Excel workbook defines the working universe.
   - Yearly CSV files are used by default only as pre-2013 price backfill.

2. **Changepoint Detection module**
   - Fast scalable CPD scores are computed on stock-vs-sector returns, because
     the first presentation focuses on idiosyncratic shocks.
   - The slower paper-style GP/Matern changepoint method remains available as a
     reference method for targeted samples.

3. **Deep Momentum Network**
   - `DMN-lite` provides a simple walk-forward supervised allocation baseline.
   - `03_train_lstm_dmn.py` provides the first PyTorch LSTM trained with a
     differentiable Sharpe-ratio loss.
   - A no-CPD LSTM ablation is available for comparison.

4. **Backtesting engine**
   - Rule-based strategies, DMN-lite, LSTM with CPD, LSTM without CPD, EW
     benchmark and SXXR benchmark are compared in one final summary.
   - Risk-adjusted metrics include Sharpe, Sortino, Calmar, max drawdown, hit
     ratio, average assets and turnover.

5. **Near-term improvements**
   - Add a validation split and early stopping inside each walk-forward fold.
   - Tune LSTM regularization and turnover control.
   - Test more informative CPD transformations, such as persistence, recent
     maximum score, shock sign or multiple lookback windows.
   - Add transaction costs and benchmark-aware portfolio constraints.

---

## Tech Stack (planned)

- Python 3.11+, virtualenv in `.venv/`
- `numpy`, `pandas`, `scipy`, `matplotlib`
- `gpflow` / `tensorflow` for Gaussian Processes
- `tensorflow` or `pytorch` for the LSTM / DMN
- `scikit-learn` for baselines and utilities

Run everything from the project venv:

```bash
source .venv/bin/activate && python3 <script>.py
```

---

## Reference

Wood, K., Roberts, S., Zohren, S. (2022).
*Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and
Changepoint Detection.* The Journal of Financial Data Science, Winter 2022.
See `documentation/jfds.2021.1.081.full 1.pdf`.
