"""Trend filter with volatility-targeted position sizing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantvol.strategies.base import BaseStrategy


class VolatilityTargetingStrategy(BaseStrategy):
    """Long above the momentum SMA, sized as ``min(1, target_vol / realized_vol)``.

    The weight is capped at 1 (no leverage) and is zero until realized volatility is
    computable, so the strategy never takes a blind position during warm-up.
    """

    name = "Volatility Targeting"

    def __init__(
        self,
        momentum_window: int = 50,
        vol_window: int = 20,
        target_vol: float = 0.15,
        trading_days: int = 252,
    ) -> None:
        if momentum_window < 2 or vol_window < 2:
            raise ValueError("windows must be >= 2")
        if target_vol <= 0:
            raise ValueError("target_vol must be positive")
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.trading_days = trading_days

    def generate_signals(self, data: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        close = data["Close"]
        trend = (close > close.rolling(self.momentum_window).mean()).astype(float)

        log_ret = pd.Series(np.log(close / close.shift(1)), index=close.index)
        realized = log_ret.rolling(self.vol_window).std() * np.sqrt(self.trading_days)
        weight = (self.target_vol / realized.where(realized > 0)).clip(upper=1.0).fillna(0.0)
        return trend * weight
