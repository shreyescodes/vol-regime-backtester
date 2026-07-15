# utils.py
# Helper functions for data generation, volatility regime detection, and performance calculations.

import numpy as np
import pandas as pd
from typing import Dict, Optional


def load_sample_data(n_days: int = 1000, 
                     seed: int = 42,
                     start_date: str = "2020-01-01",
                     trading_days: int = 252,
                     base_vol: int = 1000000,
                     transition_matrix: Optional[np.ndarray] = None,
                     regime_params: Optional[Dict[int, Dict[str, float]]] = None) -> pd.DataFrame:
    """
    Generates synthetic daily OHLCV market data using a regime-switching
    Geometric Brownian Motion (GBM) model.
    """
    np.random.seed(seed)
    dt = 1 / float(trading_days)
    
    if transition_matrix is None:
        transition_matrix = np.array([
            [0.97, 0.025, 0.005],  # Low Vol transitions
            [0.02, 0.95, 0.03],   # Med Vol transitions
            [0.015, 0.045, 0.94]   # High Vol transitions
        ])
        
    if regime_params is None:
        regime_params = {
            0: {"drift": 0.12, "vol": 0.10, "vol_scale": 0.8},
            1: {"drift": 0.02, "vol": 0.20, "vol_scale": 1.2},
            2: {"drift": -0.30, "vol": 0.45, "vol_scale": 2.5}
        }
        
    # Simulate regime states over time
    current_state = 0
    states = np.zeros(n_days, dtype=int)
    for t in range(n_days):
        states[t] = current_state
        current_state = np.random.choice(len(transition_matrix), p=transition_matrix[current_state])
        
    # Generate price path using daily returns
    prices = np.zeros(n_days)
    prices[0] = 100.0
    for t in range(1, n_days):
        state = states[t]
        drift = regime_params[state]["drift"]
        vol = regime_params[state]["vol"]
        z = np.random.normal()
        
        return_t = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
        prices[t] = prices[t - 1] * np.exp(return_t)
        
    date_index = pd.date_range(start=start_date, periods=n_days, freq="B")
    df = pd.DataFrame(index=date_index)
    df["Close"] = prices
    
    # Construct Open price with some gap noise
    df["Open"] = df["Close"].shift(1)
    df.iloc[0, df.columns.get_loc("Open")] = 99.8
    gap_noise = np.random.normal(0, 0.001, n_days)
    df["Open"] = df["Open"] * np.exp(gap_noise)
    df.iloc[0, df.columns.get_loc("Open")] = 99.8
    
    # Generate High, Low and Volume based on current regime volatility
    high_h = np.zeros(n_days)
    low_l = np.zeros(n_days)
    volumes = np.zeros(n_days)
    
    for t in range(n_days):
        state = states[t]
        vol = regime_params[state]["vol"]
        vol_scale = regime_params[state]["vol_scale"]
        
        op = df["Open"].iloc[t]
        cl = df["Close"].iloc[t]
        
        high_dev = np.abs(np.random.normal(0, vol * np.sqrt(dt) * 0.7))
        low_dev = np.abs(np.random.normal(0, vol * np.sqrt(dt) * 0.7))
        
        high_h[t] = max(op, cl) * (1.0 + high_dev)
        low_l[t] = min(op, cl) * (1.0 - low_dev)
        
        vol_noise = np.random.lognormal(0, 0.25)
        volumes[t] = int(base_vol * vol_scale * vol_noise)
        
    df["High"] = high_h
    df["Low"] = low_l
    df["Volume"] = volumes
    df["TrueRegime"] = states
    
    df = df[["Open", "High", "Low", "Close", "Volume", "TrueRegime"]]
    return df


def load_csv_data(filepath: str, date_col: str = "Date", price_cols: Dict[str, str] = None) -> pd.DataFrame:
    """
    Loads real market data from a CSV file (e.g., Binance, Yahoo Finance).
    Maps custom column names to the standard framework names (Open, High, Low, Close, Volume).
    """
    if price_cols is None:
        price_cols = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
        
    df = pd.read_csv(filepath)
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)
    
    # Rename custom columns to standard framework names
    rename_map = {v: k.capitalize() for k, v in price_cols.items()}
    df.rename(columns=rename_map, inplace=True)
    
    # Ensure required columns exist
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            if col == "Volume":
                df[col] = 1.0  # Mock volume if not provided
            else:
                df[col] = df["Close"]  # Fallback to Close price
                
    return df[required_cols]



def detect_regimes(data: pd.DataFrame, vol_window: int = 20, 
                   low_thresh: float = 0.15, high_thresh: float = 0.25,
                   trading_days: int = 252) -> pd.Series:
    """
    Detects market volatility regimes based on rolling annualized volatility.
    """
    close_prices = data["Close"]
    daily_returns = np.log(close_prices / close_prices.shift(1))
    
    rolling_vol = daily_returns.rolling(window=vol_window).std() * np.sqrt(trading_days)
    
    regimes = pd.Series(index=data.index, dtype=float)
    regimes[rolling_vol < low_thresh] = 0
    regimes[(rolling_vol >= low_thresh) & (rolling_vol < high_thresh)] = 1
    regimes[rolling_vol >= high_thresh] = 2
    
    # Forward fill is safe (uses past knowledge).
    regimes = regimes.ffill()
    
    # Replace initial NaNs (caused by rolling window) with default Medium Volatility (1) to prevent lookahead bias.
    # Previously, bfill() was propagating future volatility backwards, which is invalid.
    regimes = regimes.fillna(1).astype(int)
    
    return regimes


def calculate_performance(equity_curve: pd.Series, returns: pd.Series, 
                          trading_days: int = 252, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Calculates portfolio performance statistics.
    """
    if len(equity_curve) < 2:
        return {
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Sharpe Ratio": 0.0,
            "Max Drawdown": 0.0,
            "Calmar Ratio": 0.0
        }
        
    initial_val = equity_curve.iloc[0]
    final_val = equity_curve.iloc[-1]
    
    total_return = (final_val - initial_val) / initial_val
    
    n_days = len(equity_curve) - 1
    years = n_days / float(trading_days)
    if initial_val <= 0 or final_val <= 0 or years == 0:
        cagr = 0.0
    else:
        cagr = (final_val / initial_val) ** (1.0 / years) - 1.0
        
    std_returns = returns.std()
    if std_returns > 0:
        # Subtract daily risk-free rate from average daily returns
        daily_rf = risk_free_rate / float(trading_days)
        sharpe = ((returns.mean() - daily_rf) / std_returns) * np.sqrt(trading_days)
    else:
        sharpe = 0.0
        
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max
    max_dd = drawdowns.min()
    
    abs_max_dd = abs(max_dd)
    if abs_max_dd > 0.0001:
        calmar = cagr / abs_max_dd
    else:
        calmar = 0.0
        
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar
    }
