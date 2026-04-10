"""Load raw continuous futures data into tidy DataFrames.

Placeholder module — no implementation yet.
"""

from __future__ import annotations

from typing import Iterable


def load_futures(path: str, assets: Iterable[str]):
    """Load continuous futures prices for the given assets.

    Parameters
    ----------
    path : str
        Directory or file path containing raw futures data.
    assets : Iterable[str]
        Tickers / identifiers of the assets to load.

    Returns
    -------
    pandas.DataFrame
        A wide DataFrame indexed by date with one column per asset.
    """
    raise NotImplementedError
