# strategies.py
# Implementation of systematic, non-leveraged trading strategies.

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Optional


class BaseStrategy(ABC):
    """
    Abstract class for strategies. Subclasses must implement generate_signals.
    """
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        pass


class RegimeFilteredMovingAverageCrossover(BaseStrategy):
    """
    Standard Dual MA Crossover strategy.
    Positions are set to 0 (flat) during specified regimes to protect capital.
    """
    def __init__(self, fast_window: int = 20, slow_window: int = 50, flat_regimes: Optional[List[int]] = None):
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.flat_regimes = flat_regimes if flat_regimes is not None else [2]
        
    def generate_signals(self, data: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        close_prices = data["Close"]
        
        fast_sma = close_prices.rolling(window=self.fast_window).mean()
        slow_sma = close_prices.rolling(window=self.slow_window).mean()
        
        crossover_signal = pd.Series(0.0, index=data.index)
        crossover_signal[fast_sma > slow_sma] = 1.0
        
        final_signal = crossover_signal.copy()
        final_signal[regimes.isin(self.flat_regimes)] = 0.0
        
        return final_signal.fillna(0.0)


class VolatilityTargetingStrategy(BaseStrategy):
    """
    Dynamic position sizing based on rolling asset volatility.
    """
    def __init__(self, momentum_window: int = 50, vol_window: int = 20, 
                 target_vol: float = 0.15, trading_days: int = 252):
        self.momentum_window = momentum_window
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.trading_days = trading_days
        
    def generate_signals(self, data: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        close_prices = data["Close"]
        
        sma_filter = close_prices.rolling(window=self.momentum_window).mean()
        base_signal = pd.Series(0.0, index=data.index)
        base_signal[close_prices > sma_filter] = 1.0
        
        daily_returns = np.log(close_prices / close_prices.shift(1))
        rolling_vol = daily_returns.rolling(window=self.vol_window).std() * np.sqrt(self.trading_days)
        
        # Default to 0.0 (cash) when volatility is unknown (initial vol_window days)
        # Prevents taking blind 100% unleveraged positions before vol is computable
        scaling_factor = pd.Series(0.0, index=data.index)
        
        valid_vol_mask = (rolling_vol > 0.0) & (~rolling_vol.isna())
        scaling_factor[valid_vol_mask] = self.target_vol / rolling_vol[valid_vol_mask]
        
        weights = scaling_factor.clip(upper=1.0)
        
        final_signal = base_signal * weights
        return final_signal.fillna(0.0)
