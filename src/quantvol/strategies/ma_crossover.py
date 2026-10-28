"""Regime-filtered dual moving-average crossover."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quantvol.strategies.base import BaseStrategy


class RegimeFilteredMovingAverageCrossover(BaseStrategy):
    """Long when the fast SMA is above the slow SMA, flat in the listed regimes."""

    name = "Regime-Filtered Crossover"

    def __init__(
        self,
        fast_window: int = 20,
        slow_window: int = 50,
        flat_regimes: Sequence[int] | None = None,
    ) -> None:
        if not 0 < fast_window < slow_window:
            raise ValueError("require 0 < fast_window < slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.flat_regimes = tuple(flat_regimes) if flat_regimes is not None else (2,)

    def generate_signals(self, data: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        close = data["Close"]
        fast = close.rolling(self.fast_window).mean()
        slow = close.rolling(self.slow_window).mean()
        trend = (fast > slow).astype(float)
        return trend.where(~regimes.isin(self.flat_regimes), 0.0)
