# backtester.py
# Core vectorized backtest execution engine and stress testing suite.

import numpy as np
import pandas as pd
from typing import Dict, Any, Callable
from utils import detect_regimes, calculate_performance


class Backtester:
    """
    Vectorized backtester that handles execution delays, commissions, and slippage.
    """
    def __init__(self, initial_capital: float = 100000.0, 
                 commission_rate: float = 0.0010, 
                 slippage_rate: float = 0.0005,
                 risk_free_rate: float = 0.0):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate
        
    def run(self, data: pd.DataFrame, signals: pd.Series) -> Dict[str, Any]:
        close_prices = data["Close"]
        daily_returns = close_prices.pct_change().fillna(0.0)
        
        # Shift signals by 1 day to prevent lookahead bias (execute on next day's returns)
        execution_positions = signals.shift(1).fillna(0.0)
        
        # Calculate daily trades to estimate transaction costs
        trades = execution_positions.diff().abs().fillna(0.0)
        trades.iloc[0] = execution_positions.iloc[0]  # First trade cost
        
        # Round-trip/One-way frictional costs (commission + slippage)
        frictions = trades * (self.commission_rate + self.slippage_rate)
        
        # Daily portfolio returns
        portfolio_returns = (execution_positions * daily_returns) - frictions
        
        # Compute equity curves
        equity_curve = self.initial_capital * (1.0 + portfolio_returns).cumprod()
        
        # Compute Buy & Hold Benchmark
        benchmark_returns = daily_returns.copy()
        benchmark_returns.iloc[0] = benchmark_returns.iloc[0] - (self.commission_rate + self.slippage_rate)
        benchmark_equity = self.initial_capital * (1.0 + benchmark_returns).cumprod()
        
        # Generate metrics dictionaries
        metrics = calculate_performance(equity_curve, portfolio_returns, risk_free_rate=self.risk_free_rate)
        benchmark_metrics = calculate_performance(benchmark_equity, daily_returns, risk_free_rate=self.risk_free_rate)
        
        return {
            "equity_curve": equity_curve,
            "benchmark_curve": benchmark_equity,
            "returns": portfolio_returns,
            "metrics": metrics,
            "benchmark_metrics": benchmark_metrics,
            "positions": execution_positions,
            "trades": trades
        }
        
    def stress_test(self, data: pd.DataFrame, 
                    strategy_factory: Callable[[], Any], 
                    scenarios: Dict[str, Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Runs strategy under shock scenarios.
        Regenerates signals on the stressed data to reflect adaptive logic.
        """
        if scenarios is None:
            scenarios = {
                "vol_spike": {"factor": 2.0},
                "flash_crash": {"drop": -0.15, "day_idx": len(data) // 2},
                "cost_surge": {"cost_multiplier": 5.0}
            }
            
        stress_results = {}
        
        # Scenario 1: Volatility Spike
        if "vol_spike" in scenarios:
            factor = scenarios["vol_spike"]["factor"]
            stressed_data = data.copy()
            returns = stressed_data["Close"].pct_change().fillna(0.0)
            mean_ret = returns.mean()
            
            stressed_returns = mean_ret + factor * (returns - mean_ret)
            stressed_close = data["Close"].iloc[0] * (1.0 + stressed_returns).cumprod()
            stressed_data["Close"] = stressed_close
            
            # Scale high and low bounds roughly
            stressed_data["High"] = stressed_data["Close"] * (1.0 + (data["High"]/data["Close"] - 1.0) * factor)
            stressed_data["Low"] = stressed_data["Close"] * (1.0 - (1.0 - data["Low"]/data["Close"]) * factor)
            
            # Re-generate signals on the stressed path
            stressed_regimes = detect_regimes(stressed_data)
            strat = strategy_factory()
            stressed_signals = strat.generate_signals(stressed_data, stressed_regimes)
            
            res = self.run(stressed_data, stressed_signals)
            stress_results["vol_spike"] = res["metrics"]
            
        # Scenario 2: Flash Crash
        if "flash_crash" in scenarios:
            drop = scenarios["flash_crash"]["drop"]
            day_idx = scenarios["flash_crash"]["day_idx"]
            
            stressed_data = data.copy()
            close_arr = stressed_data["Close"].values.copy()
            
            # Apply shock on specific day
            close_arr[day_idx] = close_arr[day_idx] * (1.0 + drop)
            for t in range(day_idx + 1, len(close_arr)):
                ret = data["Close"].iloc[t] / data["Close"].iloc[t - 1]
                close_arr[t] = close_arr[t - 1] * ret
                
            stressed_data["Close"] = close_arr
            stressed_data["High"] = stressed_data["Close"] * (data["High"] / data["Close"])
            stressed_data["Low"] = stressed_data["Close"] * (data["Low"] / data["Close"])
            
            stressed_regimes = detect_regimes(stressed_data)
            strat = strategy_factory()
            stressed_signals = strat.generate_signals(stressed_data, stressed_regimes)
            
            res = self.run(stressed_data, stressed_signals)
            stress_results["flash_crash"] = res["metrics"]
            
        # Scenario 3: Transaction Cost Surge
        if "cost_surge" in scenarios:
            multiplier = scenarios["cost_surge"]["cost_multiplier"]
            
            # Modify frictional settings
            original_comm = self.commission_rate
            original_slip = self.slippage_rate
            self.commission_rate = original_comm * multiplier
            self.slippage_rate = original_slip * multiplier
            
            strat = strategy_factory()
            normal_regimes = detect_regimes(data)
            signals = strat.generate_signals(data, normal_regimes)
            
            res = self.run(data, signals)
            stress_results["cost_surge"] = res["metrics"]
            
            # Reset values
            self.commission_rate = original_comm
            self.slippage_rate = original_slip
            
        return stress_results
