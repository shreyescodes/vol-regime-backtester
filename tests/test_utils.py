# tests/test_utils.py
# Unit tests for performance calculations and regime detection.

import unittest
import pandas as pd
import numpy as np
from utils import calculate_performance, detect_regimes


class TestUtils(unittest.TestCase):
    
    def test_calculate_performance_flat_returns(self):
        # Case where there are no returns
        equity = pd.Series([100000.0, 100000.0, 100000.0])
        returns = pd.Series([0.0, 0.0, 0.0])
        
        metrics = calculate_performance(equity, returns)
        self.assertEqual(metrics["Total Return"], 0.0)
        self.assertEqual(metrics["CAGR"], 0.0)
        self.assertEqual(metrics["Sharpe Ratio"], 0.0)
        self.assertEqual(metrics["Max Drawdown"], 0.0)
        self.assertEqual(metrics["Calmar Ratio"], 0.0)

    def test_calculate_performance_normal(self):
        # 252 business days (exactly 1 year) with variation
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
        
        # Calculate resulting equity curve
        equity = 100.0 * (1.0 + returns).cumprod()
        equity_curve = pd.concat([pd.Series([100.0]), equity]).reset_index(drop=True)
        
        metrics = calculate_performance(equity_curve, returns)
        
        self.assertAlmostEqual(metrics["Total Return"], (equity_curve.iloc[-1] - 100.0) / 100.0)
        self.assertTrue(metrics["CAGR"] > -1.0)
        self.assertTrue(metrics["Sharpe Ratio"] != 0.0)
        
    def test_detect_regimes_bounds(self):
        # Generate dummy prices with specific volatility levels
        dates = pd.date_range(start="2020-01-01", periods=100, freq="B")
        
        # Segment 1: Low vol (returns close to 0)
        prices_low = [100.0]
        for _ in range(49):
            prices_low.append(prices_low[-1] * (1 + np.random.normal(0, 0.001)))
            
        # Segment 2: High vol (returns highly spread)
        prices_high = [prices_low[-1]]
        for _ in range(50):
            prices_high.append(prices_high[-1] * (1 + np.random.normal(0, 0.05)))
            
        prices = prices_low + prices_high[1:]
        df = pd.DataFrame(index=dates, data={"Close": prices})
        
        regimes = detect_regimes(df, vol_window=10, low_thresh=0.05, high_thresh=0.15)
        
        self.assertEqual(len(regimes), len(df))
        # Initial regime should be Medium Vol (1) due to NaN fallback for lookahead prevention
        self.assertEqual(regimes.iloc[0], 1)
        # Final regime should be High Vol (2)
        self.assertEqual(regimes.iloc[-1], 2)


if __name__ == "__main__":
    unittest.main()
