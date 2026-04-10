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
│   ├── data_loader.py          # Load raw futures data
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
│   └── 04_run_backtest.py
└── notebooks/
    └── 00_exploration.ipynb
```

All `src/` modules and `scripts/` are currently placeholders; concrete
implementations will land in follow-up tasks, mirroring the pipeline
described below.

---

## Planned Work

1. **Data pipeline**
   - Collect continuous futures data (Pinnacle CLC or open alternatives).
   - Compute arithmetic returns, volatility scaling, MACD features.

2. **Changepoint Detection module**
   - Implement the Matérn 3/2 kernel GP fit.
   - Implement the changepoint kernel with sigmoid blending.
   - Precompute `(ν, γ)` for multiple LBWs (10, 21, 63, 126, 252 days).

3. **Deep Momentum Network**
   - LSTM with Sharpe-ratio loss, Adam optimizer, early stopping on validation
     Sharpe.
   - Hyperparameter search over dropout, hidden size, LBW, learning rate.

4. **Backtesting engine**
   - Expanding-window backtest from 1995 to the most recent data.
   - Full benchmarking against Long Only, MACD and TSMOM variants.
   - Risk-adjusted metrics: Sharpe, Sortino, Calmar, MDD.

5. **Improvements and extensions**
   - Transaction cost modeling directly inside the Sharpe loss.
   - Alternative changepoint methods (BOCPD, ruptures, neural CPD).
   - Alternative architectures (Temporal Fusion Transformer, attention-based
     models).
   - Regime-aware ensembling of multiple LBWs.
   - Out-of-sample extension to post-2020 data.

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
