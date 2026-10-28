"""Loading and validating real-world OHLCV data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataError(ValueError):
    """Raised when market data cannot be interpreted safely."""


def standardize_ohlcv(
    df: pd.DataFrame,
    date_col: str | None = "Date",
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize a raw price table to a sorted, de-duplicated OHLCV frame.

    * Column names are matched case-insensitively; ``column_map`` (source -> standard)
      overrides that for vendor-specific headers.
    * ``Close`` is mandatory. Missing ``Open/High/Low`` fall back to ``Close`` and a
      missing ``Volume`` becomes NaN-free zeros; none of the strategies depend on them.
    * The date column defaults to ``Date`` and falls back to the first column.
    """
    out = df.copy()
    if date_col is None or date_col not in out.columns:
        date_col = str(out.columns[0])
    try:
        out[date_col] = pd.to_datetime(out[date_col])
    except (ValueError, TypeError) as exc:
        raise DataError(f"could not parse dates in column {date_col!r}") from exc
    out = out.set_index(date_col)
    out.index.name = "Date"

    mapping = {c: str(c).strip().capitalize() for c in out.columns}
    if column_map:
        mapping.update({src: dst for src, dst in column_map.items() if src in out.columns})
    out = out.rename(columns=mapping)

    if "Close" not in out.columns:
        raise DataError(f"missing required column 'Close'; found {list(out.columns)}")
    for col in ("Open", "High", "Low"):
        if col not in out.columns:
            out[col] = out["Close"]
    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    out = out[OHLCV].apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.dropna(subset=["Close"])
    if len(out) < 2:
        raise DataError("need at least 2 valid rows of data")
    if (out["Close"] <= 0).any():
        raise DataError("Close prices must be strictly positive")
    return out


def load_csv_data(
    source: Any,
    date_col: str | None = "Date",
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Read a CSV path or file-like object and standardize it."""
    return standardize_ohlcv(pd.read_csv(source), date_col=date_col, column_map=column_map)
