# STOXX 600 Raw Price Data

This folder contains the raw annual CSV price snapshots for the STOXX 600 dataset.
These files are the input data used to build the cleaned dataset and to compute
features for the strategy pipeline.

## Folder structure

- `data/raw/stoxx600/`
  - `prices_2006.csv`
  - `prices_2007.csv`
  - `prices_2008.csv`
  - `prices_2009.csv`
  - `prices_2010.csv`
  - `prices_2011.csv`
  - `prices_2012.csv`
  - `prices_2013.csv`
  - `prices_2014.csv`
  - `prices_2015.csv`
  - `prices_2016.csv`
  - `prices_2017.csv`
  - `prices_2018.csv`
  - `prices_2019.csv`
  - `prices_2020.csv`
  - `prices_2021.csv`
  - `prices_2022.csv`
  - `prices_2023.csv`
  - `prices_2024.csv`
  - `README.md`

## File naming convention

Each CSV file is named as `prices_YYYY.csv`, where `YYYY` is the calendar year.
Files are intended to be loaded in chronological order and concatenated to form a
continuous price history across years.

## Recommended usage

1. Load each file using a CSV reader such as `pandas.read_csv()`.
2. Parse the `Date` column as a datetime index.
3. Concatenate the yearly files along the time axis.
4. Clean and normalize the resulting dataset before feature engineering.

Example with pandas:

```python
import pandas as pd
from pathlib import Path

folder = Path('data/raw/stoxx600')
files = sorted(folder.glob('prices_*.csv'))
df = pd.concat(
    [pd.read_csv(f, parse_dates=['Date']) for f in files],
    ignore_index=True,
)
```

## Purpose

This raw data folder is the initial source for the project's data pipeline. The
cleaned outputs should be written to `data/processed/` after preprocessing,
feature generation, and any dataset validation steps.

## Notes

- The raw files are expected to be kept unchanged.
- Any aggregation, filtering, or transformation should happen downstream in the
  preprocessing pipeline.
