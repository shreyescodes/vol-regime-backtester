"""Optional Yahoo Finance loader with an on-disk CSV cache (requires ``quantvol[data]``)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantvol.data.loaders import DataError, load_csv_data


def fetch_yahoo(
    ticker: str,
    start: str,
    end: str | None = None,
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
) -> pd.DataFrame:
    """Download daily bars for ``ticker`` or reuse the cached CSV.

    Only the first call (or ``refresh=True``) touches the network.
    """
    cache = Path(cache_dir) / f"{ticker.upper()}_{start}_{end or 'latest'}.csv"
    if cache.exists() and not refresh:
        return load_csv_data(cache)

    try:
        import yfinance as yf
    except ImportError as exc:
        raise DataError("yfinance is not installed; run `pip install quantvol[data]`") from exc

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise DataError(f"no data returned for {ticker!r}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(cache, index=False)
    return load_csv_data(cache)
