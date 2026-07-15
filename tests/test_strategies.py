# tests/test_strategies.py
# Unit tests for backtesting trading strategies.

import unittest
import pandas as pd
import numpy as np
from strategies import RegimeFilteredMovingAverageCrossover, VolatilityTargetingStrategy


class TestStrategies(unittest.TestCase):
    
    def setUp(self):
        # Create a basic rising price DataFrame for strategy tests
        dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
        prices = [100.0]
        for i in range(1, 100):
            # Slow upward trend
            prices.append(prices[-1] * 1.002)
        self.data = pd.DataFrame(index=dates, data={"Close": prices})
        
    def test_crossover_regime_filtering(self):
        # In low volatility (Regime 0), signals should be 1.0 since prices are consistently rising
        regimes_low = pd.Series(0, index=self.data.index)
        strategy = RegimeFilteredMovingAverageCrossover(fast_window=5, slow_window=10)
        signals = strategy.generate_signals(self.data, regimes_low)
        
        # Once SMA window clears, signal should be 1.0 (long)
        self.assertEqual(signals.iloc[-1], 1.0)
        
        # If we switch to high volatility (Regime 2), crossover should override to 0.0 (cash)
        regimes_high = pd.Series(2, index=self.data.index)
        signals_high = strategy.generate_signals(self.data, regimes_high)
        self.assertEqual(signals_high.iloc[-1], 0.0)

    def test_volatility_targeting_bounds(self):
        regimes = pd.Series(0, index=self.data.index)
        strategy = VolatilityTargetingStrategy(momentum_window=10, vol_window=10, target_vol=0.10)
        signals = strategy.generate_signals(self.data, regimes)
        
        # Weights must never exceed 1.0 (non-leveraged constraint)
        self.assertTrue((signals >= 0.0).all())
        self.assertTrue((signals <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
