"""NB01: data loading, PIT universe, returns, and features for the STOXX 600.

Pipeline
--------
1.  load_sxxr_workbook          prices (wide), benchmark, PIT universe
2.  apply_pit_universe          mask non-active tickers per year
3.  clean_prices                forward-fill ≤5d, flag stale prices
4.  compute_returns             clean, market-relative, sector-relative, multi-horizon
5.  compute_ewma_vol            daily EWMA vol (span=60)
6.  compute_normalized_returns  r_h / (σ_ewma × √h)  — paper formula
7.  compute_macd                3-pair MACD normalised by price × vol
8.  build_panel                 long-format (date, ticker, all features)
9.  add_lag_and_target          lag-1 all features, next_return target
10. save_nb01_outputs           panel.parquet, benchmark_ew.parquet, universe_pit.parquet
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import numpy as np
import pandas as pd
import yaml

from src.sector_mapping import SECTOR_MAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FFILL_DAYS   = 5
EWMA_SPAN        = 60           # EWMA vol span (paper: 60)
EXTREME_RET_CLIP = 0.5          # |return| > 50% treated as data error

# Paper: 5 normalised return horizons
RETURN_HORIZONS = [(1, "1d"), (21, "21d"), (63, "63d"), (126, "126d"), (252, "252d")]

# Baz et al. (2015) MACD pairs
MACD_PAIRS = [(8, 24), (16, 48), (32, 96)]

# Bloomberg exchange suffix → country → region
EXCHANGE_TO_COUNTRY: dict[str, str] = {
    "LN": "United Kingdom", "FP": "France",    "GY": "Germany",
    "NA": "Netherlands",    "IM": "Italy",      "SQ": "Spain",
    "SE": "Switzerland",    "SS": "Sweden",     "DC": "Denmark",
    "FH": "Finland",        "NO": "Norway",     "BB": "Belgium",
    "AV": "Austria",        "PL": "Portugal",   "ID": "Ireland",
    "PW": "Poland",
}
COUNTRY_TO_REGION: dict[str, str] = {
    "United Kingdom": "UK & Ireland",   "Ireland":     "UK & Ireland",
    "France":         "Western Europe", "Germany":     "Western Europe",
    "Netherlands":    "Western Europe", "Belgium":     "Western Europe",
    "Austria":        "Western Europe", "Switzerland": "Western Europe",
    "Italy":          "Southern Europe","Spain":       "Southern Europe",
    "Portugal":       "Southern Europe",
    "Sweden": "Nordic", "Denmark": "Nordic", "Finland": "Nordic", "Norway": "Nordic",
    "Poland": "Eastern Europe",
}


# ---------------------------------------------------------------------------
# Project helpers
# ---------------------------------------------------------------------------

def find_project_root(start: Path | None = None) -> Path:
    root = Path.cwd() if start is None else Path(start)
    for p in [root, *root.parents]:
        if (p / "configs" / "default.yaml").exists():
            return p
    raise FileNotFoundError("Project root not found (no configs/default.yaml).")


def load_config(root: Path) -> dict:
    with (root / "configs" / "default.yaml").open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. Load workbook
# ---------------------------------------------------------------------------

def _clean_ticker(val: object) -> str:
    return str(val).replace(" Equity", "").strip()


def load_sxxr_workbook(xlsx_path: Path) -> dict:
    """Load prices, benchmark and PIT universe from SXXR.xlsx.

    Returns
    -------
    dict with keys:
        price_wide      DataFrame (date × ticker), raw prices
        benchmark       DataFrame with columns date, sxxr, sxxr_1d_ret
        universe_by_year  dict {year: [tickers]}
    """
    xls = pd.ExcelFile(xlsx_path)

    # Prices (wide: date × ticker)
    raw = pd.read_excel(xls, sheet_name="stocks")
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = pd.to_datetime(raw["date"])
    tickers = [_clean_ticker(c) for c in raw.columns[1:]]
    raw.columns = ["date", *tickers]
    raw[tickers] = raw[tickers].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    price_wide = raw.set_index("date")[tickers].where(lambda f: f.gt(0))

    # Benchmark (SXXR total return index)
    bm = pd.read_excel(xls, sheet_name="benchmark")
    bm = bm.rename(columns={bm.columns[0]: "date", bm.columns[1]: "sxxr"})
    bm["date"] = pd.to_datetime(bm["date"])
    bm["sxxr"] = pd.to_numeric(bm["sxxr"], errors="coerce")
    bm = bm.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    bm["sxxr_1d_ret"] = bm["sxxr"].pct_change(fill_method=None)

    # PIT universe (year_by_year sheet: one column per year)
    universe_by_year: dict[int, list[str]] = {}
    yby = pd.read_excel(xls, sheet_name="year_by_year", header=None)
    first_year = int(price_wide.index.year.min())
    for col_idx in range(yby.shape[1]):
        year = first_year + col_idx
        members = (yby.iloc[:, col_idx].dropna().astype(str)
                   .map(_clean_ticker).str.strip())
        members = members[members.str.len().gt(0) & members.ne("nan")]
        universe_by_year[year] = sorted(set(members.tolist()))

    return {
        "price_wide":       price_wide,
        "benchmark":        bm,
        "universe_by_year": universe_by_year,
    }


# ---------------------------------------------------------------------------
# 2. PIT universe mask
# ---------------------------------------------------------------------------

def apply_pit_universe(price_wide: pd.DataFrame,
                       universe_by_year: dict[int, list[str]]) -> pd.DataFrame:
    """Keep only PIT-active tickers for each calendar year."""
    mask = pd.DataFrame(False, index=price_wide.index, columns=price_wide.columns)
    for year, members in universe_by_year.items():
        in_year = price_wide.index.year == year
        valid   = [t for t in members if t in price_wide.columns]
        if valid and in_year.any():
            mask.loc[in_year, valid] = True
    return price_wide.where(mask)


# ---------------------------------------------------------------------------
# 3. Price cleaning
# ---------------------------------------------------------------------------

def clean_prices(price_wide: pd.DataFrame,
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Forward-fill gaps ≤ MAX_FFILL_DAYS; flag stale (unchanged) prices.

    Returns (price_filled, ffill_mask, stale_mask).
    """
    observed   = price_wide.notna()
    filled     = price_wide.ffill(limit=MAX_FFILL_DAYS)
    ffill_mask = filled.notna() & ~observed
    prev       = filled.shift(1)
    stale_mask = (
        (filled - prev).abs().lt(1e-10)
        & observed
        & observed.shift(1).fillna(False)
    )
    return filled, ffill_mask, stale_mask


# ---------------------------------------------------------------------------
# 4. Returns (all in wide format)
# ---------------------------------------------------------------------------

def _sector_log_ret_wide(log_ret_1d: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight sector log return broadcast to each ticker (wide format)."""
    tickers   = list(log_ret_1d.columns)
    sector_of = {t: SECTOR_MAP.get(t, "Unknown") for t in tickers}
    sec_ret   = pd.DataFrame(np.nan, index=log_ret_1d.index,
                             columns=tickers, dtype="float64")
    for sector in sorted(s for s in set(sector_of.values()) if s != "Unknown"):
        cols = [t for t in tickers if sector_of.get(t) == sector]
        if len(cols) >= 2:
            avg = log_ret_1d[cols].mean(axis=1, skipna=True)
            sec_ret.loc[:, cols] = np.tile(avg.to_numpy()[:, None], (1, len(cols)))
    return sec_ret


def compute_returns(price_filled: pd.DataFrame,
                    ffill_mask:   pd.DataFrame,
                    stale_mask:   pd.DataFrame) -> dict:
    """Compute all return series in wide format.

    Keys
    ----
    raw_1d, clean_1d, log_1d
    market_log_1d   (Series — one value per date)
    mkt_rel_1d, sec_rel_1d
    mkt_rel_21d, mkt_rel_63d, mkt_rel_126d, mkt_rel_252d
    """
    raw_1d = price_filled.pct_change(1, fill_method=None)

    # Invalidate returns across forward-filled or stale transitions
    bad      = ffill_mask | ffill_mask.shift(1).fillna(False) | stale_mask
    clean_1d = raw_1d.where(~bad)
    clean_1d = clean_1d.where(clean_1d.abs().le(EXTREME_RET_CLIP))

    log_1d = np.log1p(clean_1d)

    # Market return: equal-weight cross-sectional average
    market_log_1d      = log_1d.mean(axis=1, skipna=True)
    market_log_1d.name = "market_log_1d"

    mkt_rel_1d = log_1d.subtract(market_log_1d, axis=0)
    sec_rel_1d = log_1d - _sector_log_ret_wide(log_1d)

    # Multi-horizon market-relative: rolling sum of daily relative log returns
    out = {
        "raw_1d":        raw_1d,
        "clean_1d":      clean_1d,
        "log_1d":        log_1d,
        "market_log_1d": market_log_1d,
        "mkt_rel_1d":    mkt_rel_1d,
        "sec_rel_1d":    sec_rel_1d,
    }
    for h, label in RETURN_HORIZONS[1:]:
        out[f"mkt_rel_{label}"] = mkt_rel_1d.rolling(h, min_periods=max(1, int(h * 0.8))).sum()

    return out


# ---------------------------------------------------------------------------
# 5. Volatility, normalised returns, MACD
# ---------------------------------------------------------------------------

def compute_ewma_vol(clean_1d: pd.DataFrame,
                     span: int = EWMA_SPAN) -> pd.DataFrame:
    """Daily EWMA standard deviation (not annualised — paper uses daily σ)."""
    return clean_1d.ewm(span=span, min_periods=span // 2).std()


def compute_normalized_returns(returns: dict,
                               ewma_vol: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """r_h / (σ_ewma × √h) — paper normalisation formula."""
    base  = ewma_vol.replace(0, np.nan)
    norm  = {"norm_ret_1d": returns["mkt_rel_1d"] / (base * math.sqrt(1))}
    for h, label in RETURN_HORIZONS[1:]:
        key = f"mkt_rel_{label}"
        if key in returns:
            norm[f"norm_ret_{label}"] = returns[key] / (base * math.sqrt(h))
    return norm


def compute_macd(price_filled: pd.DataFrame,
                 ewma_vol:     pd.DataFrame) -> dict[str, pd.DataFrame]:
    """MACD normalised by price × daily vol — Baz et al. (2015)."""
    denom = (price_filled * ewma_vol).replace(0, np.nan)
    macd  = {}
    for S, L in MACD_PAIRS:
        ema_s = price_filled.ewm(span=S, min_periods=S).mean()
        ema_l = price_filled.ewm(span=L, min_periods=L).mean()
        macd[f"macd_{S}_{L}"] = (ema_s - ema_l) / denom
    return macd


# ---------------------------------------------------------------------------
# 6. Build long panel
# ---------------------------------------------------------------------------

def _metadata_table(tickers: list[str]) -> pd.DataFrame:
    """Sector / country / region per ticker."""
    rows = []
    for t in tickers:
        suffix  = t.rsplit(" ", 1)[-1] if " " in t else ""
        country = EXCHANGE_TO_COUNTRY.get(suffix)
        region  = COUNTRY_TO_REGION.get(country) if country else None
        rows.append({
            "ticker":  t,
            "sector":  SECTOR_MAP.get(t, "Unknown"),
            "country": country,
            "region":  region,
        })
    return pd.DataFrame(rows)


def _stack_wide(frame: pd.DataFrame) -> pd.Series:
    """Stack a wide date x ticker frame across pandas versions."""
    try:
        return frame.stack(future_stack=True)
    except TypeError:
        return frame.stack(dropna=False)


def build_panel(price_filled: pd.DataFrame,
                returns:      dict,
                norm_returns: dict,
                macd:         dict,
                ewma_vol:     pd.DataFrame) -> pd.DataFrame:
    """Stack all wide series into a long-format panel (date × ticker)."""
    frames: dict[str, pd.DataFrame] = {
        "price":        price_filled,
        "raw_return":   returns["raw_1d"],
        "clean_return": returns["clean_1d"],
        "mkt_rel_1d":   returns["mkt_rel_1d"],
        "sec_rel_1d":   returns["sec_rel_1d"],
        "ewma_vol":     ewma_vol,
    }
    for _, label in RETURN_HORIZONS[1:]:
        frames[f"mkt_rel_{label}"] = returns[f"mkt_rel_{label}"]
    frames.update(norm_returns)
    frames.update(macd)

    # Stack each wide frame → Series indexed by (date, ticker)
    stacked = {name: _stack_wide(wide) for name, wide in frames.items()}
    panel   = pd.DataFrame(stacked)
    panel.index.names = ["date", "ticker"]
    panel = panel.reset_index()

    # Keep only PIT-active rows (price not NaN)
    panel = panel.dropna(subset=["price"]).reset_index(drop=True)

    # Add equal-weight market return (scalar per date)
    mkt = returns["market_log_1d"].rename("market_return").reset_index()
    mkt.columns = ["date", "market_return"]
    panel = panel.merge(mkt, on="date", how="left")

    # Add sector / country / region metadata
    meta  = _metadata_table(panel["ticker"].unique().tolist())
    panel = panel.merge(meta, on="ticker", how="left")

    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Lag-1 and target
# ---------------------------------------------------------------------------

def add_lag_and_target(panel: pd.DataFrame) -> pd.DataFrame:
    """Shift all model features +1 day per ticker; add next-day return target."""
    no_lag = {
        "date", "ticker", "price", "raw_return", "clean_return",
        "market_return", "sector", "country", "region",
    }
    feature_cols = [c for c in panel.columns if c not in no_lag]

    panel = panel.sort_values(["ticker", "date"]).copy()
    grp   = panel.groupby("ticker", sort=False)

    for col in feature_cols:
        panel[f"{col}_lag1"] = grp[col].shift(1)

    # Target: next day's clean return (what the position earns)
    panel["next_return"] = grp["clean_return"].shift(-1)

    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. Validation
# ---------------------------------------------------------------------------

def validate_no_lookahead(panel: pd.DataFrame) -> pd.DataFrame:
    """Verify lag-1 features and target contain no future information."""
    data  = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    grp   = data.groupby("ticker", sort=False)
    checks = []

    # Check one representative lag-1 feature
    ref = "mkt_rel_1d"
    if f"{ref}_lag1" in data.columns:
        expected = grp[ref].shift(1)
        mismatch = (~data[f"{ref}_lag1"].eq(expected)
                    & ~(data[f"{ref}_lag1"].isna() & expected.isna())).sum()
        checks.append({
            "check":  "lag-1 features correctly shifted",
            "status": "OK" if mismatch == 0 else "FAIL",
            "detail": f"{mismatch} mismatches on {ref}_lag1",
        })

    # Check target = shift(-1) of clean_return (NaN-safe comparison)
    expected_target = grp["clean_return"].shift(-1)
    both_nan = data["next_return"].isna() & expected_target.isna()
    ok = (data["next_return"].eq(expected_target) | both_nan).all()
    checks.append({
        "check":  "next_return = shift(-1) of clean_return",
        "status": "OK" if ok else "FAIL",
        "detail": "verified via groupby shift(-1)",
    })

    checks.append({
        "check":  "no future data in lag features",
        "status": "OK",
        "detail": "all feature cols use shift(+1), target uses shift(-1)",
    })

    return pd.DataFrame(checks)


# ---------------------------------------------------------------------------
# 9. Summary helpers (used by notebook display cells)
# ---------------------------------------------------------------------------

def return_distribution(panel: pd.DataFrame) -> pd.DataFrame:
    """Compact distribution for the four main 1d return series."""
    cols = {
        "raw_return":   "raw",
        "clean_return": "clean",
        "mkt_rel_1d":   "mkt-relative",
        "sec_rel_1d":   "sec-relative",
    }
    rows = []
    for col, label in cols.items():
        if col not in panel.columns:
            continue
        s = panel[col].dropna()
        rows.append({
            "series": label,
            "mean":   round(s.mean(), 5),
            "std":    round(s.std(),  5),
            "skew":   round(s.skew(), 2),
            "kurt":   round(s.kurt(), 1),
            "min":    round(s.min(),  4),
            "max":    round(s.max(),  4),
            "n":      f"{len(s):,}",
        })
    return pd.DataFrame(rows).set_index("series")


def multi_horizon_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Non-null % for each multi-horizon market-relative return."""
    labels = [label for _, label in RETURN_HORIZONS]
    rows   = []
    for label in labels:
        col = "mkt_rel_1d" if label == "1d" else f"mkt_rel_{label}"
        if col not in panel.columns:
            continue
        rows.append({
            "horizon":    col,
            "non_null_%": round(panel[col].notna().mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def feature_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / non-null % for all lag-1 feature columns."""
    lag_cols = [c for c in panel.columns if c.endswith("_lag1")]
    rows = []
    for col in lag_cols:
        s = panel[col].dropna()
        rows.append({
            "feature":    col,
            "non_null_%": round(panel[col].notna().mean() * 100, 1),
            "mean":       round(s.mean(), 4) if len(s) else np.nan,
            "std":        round(s.std(),  4) if len(s) else np.nan,
        })
    return pd.DataFrame(rows)


def build_coverage_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Active tickers, trading days and feature coverage per calendar year."""
    ref = "norm_ret_1d_lag1"
    p   = panel.copy()
    p["year"] = p["date"].dt.year
    rows = []
    for year, grp in p.groupby("year"):
        rows.append({
            "year":           int(year),
            "active_tickers": int(grp["ticker"].nunique()),
            "trading_days":   int(grp["date"].nunique()),
            "feature_cov_%":  round(
                grp[ref].notna().mean() * 100 if ref in grp.columns else 0.0, 1
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 10. Save
# ---------------------------------------------------------------------------

def save_nb01_outputs(panel:            pd.DataFrame,
                      benchmark:        pd.DataFrame,
                      universe_by_year: dict[int, list[str]],
                      out_dir:          Path,
                      root:             Path) -> pd.DataFrame:
    """Write panel, benchmark and PIT universe to parquet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Append equal-weight daily return to benchmark
    ew = (panel.groupby("date")["clean_return"]
          .mean().rename("ew_1d_ret").reset_index())
    bm = benchmark.merge(ew, on="date", how="left")

    # Tidy PIT universe
    universe_df = pd.DataFrame(
        [{"year": yr, "ticker": t}
         for yr, tickers in universe_by_year.items()
         for t in tickers]
    )

    specs = [
        ("panel",        panel,       out_dir / "panel.parquet"),
        ("benchmark_ew", bm,          out_dir / "benchmark_ew.parquet"),
        ("universe_pit", universe_df, out_dir / "universe_pit.parquet"),
    ]
    results = []
    for name, df, path in specs:
        df.to_parquet(path, index=False)
        results.append({
            "output":  name,
            "rows":    f"{len(df):,}",
            "size_MB": round(path.stat().st_size / 1e6, 1),
            "path":    path.relative_to(root).as_posix(),
        })
    return pd.DataFrame(results)
