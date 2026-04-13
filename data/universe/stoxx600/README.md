# STOXX 600 Universe Data

This folder contains the asset universe definitions used by the project.
It specifies the full STOXX 600 stock universe and the per-year universe selections.

## Folder structure

- `data/universe/stoxx600/`
  - `all_sxxp_stocks.csv`       # Complete list of STOXX 600 tickers / identifiers
  - `universe_by_year.json`    # Yearly mappings of active universe symbols
  - `README.md`

## File descriptions

### `all_sxxp_stocks.csv`
- Contains the full set of STOXX 600 universe symbols.
- Use this file as the master universe reference when building backtests,
  filtering assets, or validating symbol coverage.

### `universe_by_year.json`
- Contains the universe membership for each calendar year.
- Use this file to load the active stock list for a specific year when
  backtesting or generating historical feature sets.
- The JSON format is typically a mapping from year keys to symbol lists, for
  example:

```json
{
  "2006": ["AAA", "AAB", ...],
  "2007": ["AAC", "AAD", ...],
  ...
}
```

## Recommended usage

1. Load `all_sxxp_stocks.csv` to obtain the full STOXX 600 symbol universe.
2. Load `universe_by_year.json` to retrieve the active universe for each year.
3. Use the year-specific list when constructing historical simulations,
   ensuring only the assets available in that year are included.

Example with pandas and json:

```python
import json
import pandas as pd
from pathlib import Path

folder = Path('data/universe/stoxx600')
all_stocks = pd.read_csv(folder / 'all_sxxp_stocks.csv')
with open(folder / 'universe_by_year.json', 'r') as f:
    universe_by_year = json.load(f)

# Example: symbols active in 2015
symbols_2015 = universe_by_year['2015']
```

## Purpose

This folder defines the eligible STOXX 600 stock universe and its yearly
changes. It is used to constrain the dataset, maintain consistency across
historical periods, and support time-aware backtesting.

## Notes

- Keep the source universe files unchanged unless the asset list is updated.
- If the universe changes by year, use `universe_by_year.json` rather than the
  full master list.
- Treat `all_sxxp_stocks.csv` as the complete reference universe, and
  `universe_by_year.json` as the active membership for each historical year.
