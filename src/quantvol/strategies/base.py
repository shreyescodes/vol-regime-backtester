"""Strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    """A strategy maps market data and regimes to target portfolio weights in ``[0, 1]``.

    Implementations must be causal: the weight at bar *t* may only depend on information
    available at the close of bar *t*. The engine then applies the weight to bar *t+lag*.
    """

    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        """Return target weights indexed like ``data``."""
