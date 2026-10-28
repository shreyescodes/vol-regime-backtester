"""Volatility-regime classification.

Regimes are derived from the trailing annualized realized volatility of log returns:
``0`` = low, ``1`` = medium, ``2`` = high. The value at bar *t* only uses prices up to and
including bar *t*; bars inside the warm-up window default to the medium regime rather than
back-filling from the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_LABELS: dict[int, str] = {0: "Low Vol", 1: "Medium Vol", 2: "High Vol"}


def detect_regimes(
    data: pd.DataFrame,
    vol_window: int = 20,
    low_thresh: float = 0.15,
    high_thresh: float = 0.25,
    trading_days: int = 252,
) -> pd.Series:
    """Classify each bar into a volatility regime (0 low, 1 medium, 2 high)."""
    if not 0 < low_thresh < high_thresh:
        raise ValueError("thresholds must satisfy 0 < low_thresh < high_thresh")
    close = data["Close"]
    log_ret = pd.Series(np.log(close / close.shift(1)), index=close.index)
    rolling_vol = log_ret.rolling(window=vol_window).std() * np.sqrt(trading_days)

    regimes = pd.Series(1, index=data.index, dtype=int)
    known = rolling_vol.notna()
    regimes[known & (rolling_vol < low_thresh)] = 0
    regimes[known & (rolling_vol >= high_thresh)] = 2
    regimes.name = "Regime"
    return regimes


def transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """Empirical one-step regime transition probabilities (rows sum to 1)."""
    counts = pd.crosstab(regimes.shift(1).dropna().astype(int), regimes.iloc[1:].astype(int))
    counts = counts.reindex(index=list(REGIME_LABELS), columns=list(REGIME_LABELS), fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    probs.index = pd.Index([REGIME_LABELS[i] for i in probs.index], name="From")
    probs.columns = pd.Index([REGIME_LABELS[i] for i in probs.columns], name="To")
    return probs
