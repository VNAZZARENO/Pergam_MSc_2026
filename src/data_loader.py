"""Data loading helpers for the STOXX 600 project.

The raw files are yearly CSV files in wide format:
one row per date, one column per stock ticker.

The functions below keep the data pipeline simple:
1. load yearly prices,
2. clean obvious bad values,
3. convert the wide table to a tidy panel.
"""

from __future__ import annotations

import sys
import importlib.util
import json
from pathlib import Path

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import pandas as pd


EXCHANGE_TO_COUNTRY = {
    "LN": "United Kingdom",
    "FP": "France",
    "GY": "Germany",
    "NA": "Netherlands",
    "IM": "Italy",
    "SQ": "Spain",
    "SM": "Spain",
    "SE": "Switzerland",
    "SW": "Switzerland",
    "SS": "Sweden",
    "DC": "Denmark",
    "FH": "Finland",
    "NO": "Norway",
    "BB": "Belgium",
    "AV": "Austria",
    "PL": "Portugal",
    "ID": "Ireland",
    "PW": "Poland",
}


COUNTRY_TO_REGION = {
    "United Kingdom": "UK & Ireland",
    "Ireland": "UK & Ireland",
    "France": "Western Europe",
    "Germany": "Western Europe",
    "Netherlands": "Western Europe",
    "Belgium": "Western Europe",
    "Austria": "Western Europe",
    "Switzerland": "Western Europe",
    "Italy": "Southern Europe",
    "Spain": "Southern Europe",
    "Portugal": "Southern Europe",
    "Sweden": "Nordic",
    "Denmark": "Nordic",
    "Finland": "Nordic",
    "Norway": "Nordic",
    "Poland": "Eastern Europe",
}

# Organisation du fichier :
# chargement des donnees, nettoyage simple, puis passage au format panel.


# [LOAD] load_stoxx600_prices
# Charge les fichiers annuels de prix et les concatene par date.
def load_stoxx600_prices(raw_dir, start_year=None, end_year=None):
    """Load and concatenate yearly STOXX 600 price CSV files.

    Parameters
    ----------
    raw_dir : str or Path
        Folder containing files named ``prices_YYYY.csv``.
    start_year, end_year : int or None
        Optional year filter.

    Returns
    -------
    pandas.DataFrame
        Wide price table indexed by date.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("prices_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prices_YYYY.csv files found in {raw_dir}")

    selected = []
    for file in files:
        year = int(file.stem.split("_")[-1])
        if start_year is not None and year < start_year:
            continue
        if end_year is not None and year > end_year:
            continue
        selected.append(file)

    if not selected:
        raise ValueError("No raw price files match the requested year range.")

    frames = []
    for file in selected:
        frame = pd.read_csv(file, parse_dates=["Date"])
        frame = frame.rename(columns={"Date": "date"})
        frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.drop_duplicates(subset="date")
    prices = prices.set_index("date").sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    return prices


# [LOAD] load_price_atlas_prices
# Charge les prix depuis le classeur PRICE ATLAS.
def load_price_atlas_prices(path, start_year=None, end_year=None, sheet_name="price"):
    """Load wide stock prices from the PRICE ATLAS Excel workbook."""
    path = Path(path)
    prices = pd.read_excel(path, sheet_name=sheet_name)
    prices = prices.rename(columns={"Ticker": "date", "Date": "date"})
    if "date" not in prices.columns:
        raise ValueError(f"No date column found in {path} sheet {sheet_name!r}.")

    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index("date").sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")

    if start_year is not None:
        prices = prices.loc[prices.index.year >= start_year]
    if end_year is not None:
        prices = prices.loc[prices.index.year <= end_year]
    return prices


# [LOAD] append_price_atlas_tail
# Ajoute les dates recentes de l'Excel en recalant les niveaux sur l'historique CSV.
def append_price_atlas_tail(prices, atlas_prices):
    """Append the PRICE ATLAS tail after rescaling it to the CSV history.

    The yearly CSV archive is the continuous historical source through 2024,
    while the PRICE ATLAS workbook provides the recent 2025-2026 tail. Their
    raw price levels can differ on the overlapping dates, so common tickers are
    rescaled on the latest shared date before appending. This keeps returns
    around the source switch from being dominated by an artificial level jump.
    """
    if prices.empty:
        return atlas_prices.sort_index()

    tail = atlas_prices.loc[atlas_prices.index > prices.index.max()]
    if tail.empty:
        return prices.sort_index()

    common_cols = prices.columns.intersection(atlas_prices.columns)
    if len(common_cols):
        scale_values = {}
        for col in common_cols:
            overlap = pd.DataFrame({
                "csv": prices[col],
                "atlas": atlas_prices[col],
            }).dropna()
            overlap = overlap.loc[overlap["atlas"] != 0]
            if overlap.empty:
                continue
            anchor = overlap.iloc[-1]
            scale_values[col] = anchor["csv"] / anchor["atlas"]

        scale = pd.Series(scale_values, dtype="float64").replace(
            [float("inf"), float("-inf")],
            pd.NA,
        ).dropna()
        tail = tail.copy()
        tail.loc[:, scale.index] = tail.loc[:, scale.index].multiply(
            scale,
            axis=1,
        )

    combined = pd.concat([prices, tail], axis=0, sort=True)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_index()


# [LOAD] load_universe_by_year
# Charge le mapping annee -> tickers actifs.
def load_universe_by_year(path):
    """Load yearly universe membership from JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        universe = json.load(f)
    return {str(year): list(tickers) for year, tickers in universe.items()}


# [VALIDATION] universe_coverage
# Compare les tickers de prix avec les tickers d'univers par annee.
def universe_coverage(prices, universe_by_year):
    """Return a compact per-year coverage report for a wide price table."""
    rows = []
    price_columns = set(prices.columns)
    available_years = sorted(set(prices.index.year))
    for year in available_years:
        key = str(year)
        if key not in universe_by_year:
            rows.append({
                "year": year,
                "universe_tickers": 0,
                "price_tickers": int(prices.loc[prices.index.year == year].notna().any().sum()),
                "matched_tickers": 0,
                "missing_in_prices": 0,
                "extra_in_prices": len(price_columns),
            })
            continue

        universe = set(universe_by_year[key])
        year_prices = set(prices.loc[prices.index.year == year].dropna(axis=1, how="all").columns)
        rows.append({
            "year": year,
            "universe_tickers": len(universe),
            "price_tickers": len(year_prices),
            "matched_tickers": len(universe & year_prices),
            "missing_in_prices": len(universe - year_prices),
            "extra_in_prices": len(year_prices - universe),
        })
    return pd.DataFrame(rows)


# [FILTER] filter_prices_by_yearly_universe
# Garde seulement les valeurs de prix appartenant a l'univers de l'annee.
def filter_prices_by_yearly_universe(prices, universe_by_year):
    """Mask prices outside each calendar year's active universe."""
    filtered = prices.copy()
    for year in sorted(set(filtered.index.year)):
        key = str(year)
        mask = filtered.index.year == year
        if key not in universe_by_year:
            continue
        keep = set(universe_by_year[key])
        drop_cols = [col for col in filtered.columns if col not in keep]
        filtered.loc[mask, drop_cols] = pd.NA
    return filtered


# [CLEAN] clean_prices
# Supprime les prix aberrants et bouche seulement les petits trous.
def clean_prices(prices, min_price=0.01, ffill_limit=5):
    """Clean a wide price table.

    Prices below ``min_price`` are treated as missing values. Short missing
    gaps are forward-filled stock by stock.
    """
    clean = prices.copy()
    clean = clean.where(clean >= min_price)
    clean = clean.ffill(limit=ffill_limit)
    return clean


# [FORMAT] prices_to_panel
# Convertit le format large en panel : date, ticker, prix.
def prices_to_panel(prices):
    """Convert wide prices to a tidy panel with columns date, ticker, price."""
    panel = (
        prices
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="price")
        .dropna(subset=["price"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    return panel


# [METADATA] add_geography
# Ajoute exchange, country et region a partir du suffixe Bloomberg.
def add_geography(panel):
    """Add exchange, country and region columns from Bloomberg-like tickers."""
    out = panel.copy()
    out["exchange"] = out["ticker"].astype(str).str.split().str[-1]
    out["country"] = out["exchange"].map(EXCHANGE_TO_COUNTRY)
    out["region"] = out["country"].map(COUNTRY_TO_REGION)
    return out


# [METADATA] load_sector_mapping
# Charge une table optionnelle ticker -> secteur quand elle sera disponible.
def load_sector_mapping(path):
    """Load an optional ticker-sector mapping from CSV or Excel.

    The file must contain a ticker column and a sector-like column. Column names
    are matched case-insensitively to keep the handoff with the team flexible.
    """
    path = Path(path)
    if path.suffix.lower() == ".py":
        spec = importlib.util.spec_from_file_location("sector_mapping", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot import sector mapping from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "SECTOR_MAP"):
            raise ValueError(f"{path} must define SECTOR_MAP.")
        mapping = pd.DataFrame(
            sorted(module.SECTOR_MAP.items()),
            columns=["ticker", "sector"],
        )
        return mapping.drop_duplicates(subset=["ticker"], keep="last")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        mapping = pd.read_excel(path)
    else:
        mapping = pd.read_csv(path)

    normalized = {col.lower().strip(): col for col in mapping.columns}
    ticker_col = normalized.get("ticker") or normalized.get("security") or normalized.get("symbol")
    sector_col = (
        normalized.get("sector")
        or normalized.get("supersector")
        or normalized.get("industry")
        or normalized.get("gics_sector")
        or normalized.get("icb_supersector")
    )
    if ticker_col is None or sector_col is None:
        raise ValueError("Sector mapping must contain ticker and sector columns.")

    return (
        mapping[[ticker_col, sector_col]]
        .rename(columns={ticker_col: "ticker", sector_col: "sector"})
        .dropna(subset=["ticker"])
        .drop_duplicates(subset=["ticker"], keep="last")
    )


# [COMPAT] load_futures
# Garde l'ancienne API du projet pour ne pas casser les vieux notebooks.
def load_futures(path, assets=None):
    """Backward-compatible alias for the old project API.

    The original paper used futures. In this project we use the STOXX 600
    PRICE ATLAS Excel workbook, but keeping this function avoids breaking old
    notebooks.
    """
    path = Path(path)
    prices = pd.read_excel(path, sheet_name="price")
    prices = prices.rename(columns={"Ticker": "date"})
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.set_index("date").sort_index()
    if assets is not None:
        available = [asset for asset in assets if asset in prices.columns]
        prices = prices[available]
    return prices


# [LOAD] load_benchmark
# Charge le benchmark marche SXXR utilise pour les rendements relatifs.
def load_benchmark(path):
    """Load the SXXR benchmark from the PRICE ATLAS Excel file."""
    path = Path(path)
    bench = pd.read_excel(path, sheet_name="benchmark")
    bench["Date"] = pd.to_datetime(bench["Date"])
    bench = bench.set_index("Date").sort_index()
    return bench["price"].rename("SXXR")


# [LOAD] load_processed
# Charge directement les CSV deja prepares par les scripts.
def load_processed(processed_dir):
    """Load the processed stock panel and benchmark CSVs."""
    processed_dir = Path(processed_dir)
    stocks = pd.read_csv(processed_dir / "stoxx600_processed.csv", parse_dates=["date"])
    benchmarks = pd.read_csv(
        processed_dir / "benchmark_stoxx600_ew.csv",
        parse_dates=["date"],
    )
    return stocks, benchmarks
