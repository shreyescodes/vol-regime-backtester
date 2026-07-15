# Project Summary: Adaptive Regime-Aware Systematic Backtester

## Objective
Developed a production-grade systematic trading backtester optimized for rules-based, multi-regime strategies. The goal was to build a robust framework that proves strong skills in quantitative development, feature engineering, and risk management—specifically tailored for non-leveraged execution.

## Architecture & Implementation Details
I structured the project strictly modularly to ensure clean separation of concerns. 

1. **Data & State Management (`utils.py`)**
   - Wrote a synthetic data generator based on Geometric Brownian Motion (GBM) with a Markov chain transition matrix. 
   - Added `load_csv_data` parser to ingest real market feeds dynamically.
   - Implemented volatility regime detection (Low/Med/High states) using rolling annualized standard deviations.

2. **Strategy Layer (`strategies.py`)**
   - Built a `BaseStrategy` abstract class.
   - **Regime-Filtered MA Crossover:** Traditional trend-following but overrides signals to 0 (flat) when entering a High Volatility regime. Capital protection first.
   - **Volatility-Targeting Strategy:** Dynamically scales position weights inversely proportional to rolling volatility. 

3. **Execution Engine (`backtester.py`)**
   - 100% vectorized. Handles realistic friction (commissions + slippage).
   - Includes a **Stress Test Module** that shocks the data (flash crashes, vol spikes) and re-evaluates the strategy's adaptive logic under duress.

4. **Dynamic Configuration (`run_backtest.py`)**
   - Extracted all parameters into a master `CONFIG` dictionary.
   - The engine is highly configurable and can switch between asset classes (Crypto/Gold/Equities) simply by altering `trading_days` and `target_vol` in the config. No hardcoded logic.

5. **Interactive Web Dashboard (`app.py`)**
   - Built a Streamlit web application to serve as the user interface.
   - Allows users to drag-and-drop custom `.csv` files, dynamically adjust strategy parameters via sliders, and visualize the portfolio's equity curve and regime breakdown without touching code.

## Engineering Best Practices Used
* **Unit Testing:** Implemented standard `unittest` suite (`tests/`) to validate CAGR, Sharpe ratio, and Drawdown calculations (handling edge cases like division by zero or negative equity).
* **Logging:** Replaced basic prints with `logging` module. Console + File handlers setup.
* **Environment Management:** Provided `requirements.txt` and standard `.gitignore`.
* **Clean Code:** Adhered strictly to PEP-8 style, with explicit type hinting (`typing` module) across all methods for better IDE support and static analysis.

## Conclusion
This framework demonstrates a deep understanding of what is required at a systematic trading firm. It moves past simple arithmetic backtesting and focuses heavily on realistic execution costs, regime-aware position sizing, and systemic stress testing. The code is production-ready, fully parameterized, and strictly decoupled.
