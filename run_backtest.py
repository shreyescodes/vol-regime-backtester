# run_backtest.py
# Main script to run the backtester pipeline and plot equity curves.
# All parameters are dynamically driven via the CONFIG dictionary.

import os
import logging
import matplotlib
matplotlib.use("Agg")  # Headless plotting
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import load_sample_data, detect_regimes
from strategies import RegimeFilteredMovingAverageCrossover, VolatilityTargetingStrategy
from backtester import Backtester

# Master Configuration block: Eliminates hardcoded magic numbers
CONFIG = {
    "data_generation": {
        "n_days": 1000,
        "seed": 101,
        "start_date": "2020-01-01",
        "trading_days": 252,
        "base_vol": 1000000
    },
    "regime_detection": {
        "vol_window": 20,
        "low_thresh": 0.14,
        "high_thresh": 0.25,
        "trading_days": 252
    },
    "backtester": {
        "initial_capital": 100000.0,
        "commission_rate": 0.0010,
        "slippage_rate": 0.0005,
        "risk_free_rate": 0.04  # 4% annualized risk-free rate
    },
    "strategies": {
        "crossover": {
            "fast_window": 20,
            "slow_window": 50,
            "flat_regimes": [2]
        },
        "vol_targeting": {
            "momentum_window": 50,
            "vol_window": 20,
            "target_vol": 0.12,
            "trading_days": 252
        }
    },
    "stress_tests": {
        "vol_spike": {"factor": 2.0},
        "flash_crash": {"drop": -0.15, "day_idx": 500},
        "cost_surge": {"cost_multiplier": 5.0}
    }
}

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backtest.log", mode="w")
    ]
)
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("=" * 60)
    logger.info("      Adaptive Volatility-Regime Backtesting Framework      ")
    logger.info("=" * 60)
    
    # Load synthetic market data (driven by config)
    logger.info("Generating market data (GBM)...")
    data = load_sample_data(**CONFIG["data_generation"])
    
    # Classify market states (driven by config)
    logger.info("Detecting volatility regimes...")
    regimes = detect_regimes(data, **CONFIG["regime_detection"])
    data["DetectedRegime"] = regimes
    
    # Frictional setup (driven by config)
    backtester = Backtester(**CONFIG["backtester"])
    
    # 1. Regime-Filtered Moving Average Crossover (driven by config)
    logger.info("Simulating Regime-Filtered MA Crossover Strategy...")
    strat_crossover = RegimeFilteredMovingAverageCrossover(**CONFIG["strategies"]["crossover"])
    signals_crossover = strat_crossover.generate_signals(data, regimes)
    results_crossover = backtester.run(data, signals_crossover)
    
    # 2. Volatility-Targeted Position Sizing (driven by config)
    logger.info("Simulating Volatility-Targeting Strategy...")
    strat_voltarget = VolatilityTargetingStrategy(**CONFIG["strategies"]["vol_targeting"])
    signals_voltarget = strat_voltarget.generate_signals(data, regimes)
    results_voltarget = backtester.run(data, signals_voltarget)
    
    # Run stress test suite on Crossover strategy (driven by config)
    logger.info("Running stress tests on MA Crossover...")
    crossover_factory = lambda: RegimeFilteredMovingAverageCrossover(**CONFIG["strategies"]["crossover"])
    stress_metrics = backtester.stress_test(data, crossover_factory, scenarios=CONFIG["stress_tests"])
    
    # Print metrics
    logger.info("\n" + "=" * 55 + "\n               Backtest Performance                  \n" + "=" * 55)
    
    metrics_df = pd.DataFrame({
        "Buy & Hold (Benchmark)": results_crossover["benchmark_metrics"],
        "Regime-Filtered Crossover": results_crossover["metrics"],
        "Volatility-Targeting": results_voltarget["metrics"]
    }).T
    
    metrics_df["Total Return"] = metrics_df["Total Return"].map(lambda x: f"{x * 100:.2f}%")
    metrics_df["CAGR"] = metrics_df["CAGR"].map(lambda x: f"{x * 100:.2f}%")
    metrics_df["Sharpe Ratio"] = metrics_df["Sharpe Ratio"].map(lambda x: f"{x:.3f}")
    metrics_df["Max Drawdown"] = metrics_df["Max Drawdown"].map(lambda x: f"{x * 100:.2f}%")
    metrics_df["Calmar Ratio"] = metrics_df["Calmar Ratio"].map(lambda x: f"{x:.3f}")
    logger.info("\n" + metrics_df.to_string())
    
    logger.info("\n" + "=" * 55 + "\n                 Stress Test Results                 \n" + "=" * 55)
    
    stress_df = pd.DataFrame(stress_metrics).T
    stress_df["Total Return"] = stress_df["Total Return"].map(lambda x: f"{x * 100:.2f}%")
    stress_df["CAGR"] = stress_df["CAGR"].map(lambda x: f"{x * 100:.2f}%")
    stress_df["Sharpe Ratio"] = stress_df["Sharpe Ratio"].map(lambda x: f"{x:.3f}")
    stress_df["Max Drawdown"] = stress_df["Max Drawdown"].map(lambda x: f"{x * 100:.2f}%")
    stress_df["Calmar Ratio"] = stress_df["Calmar Ratio"].map(lambda x: f"{x:.3f}")
    logger.info("\n" + stress_df.to_string())
    
    # Plot curves
    logger.info("Generating performance plot...")
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [2, 1]})
    
    dates = data.index
    
    # Subplot 1: Equity curves
    ax1.plot(dates, results_crossover["benchmark_curve"], label="Buy & Hold (Benchmark)", 
             color="#ff4757", alpha=0.8, linewidth=1.5)
    ax1.plot(dates, results_crossover["equity_curve"], label="Regime-Filtered Crossover", 
             color="#1ebd8e", linewidth=2.0)
    ax1.plot(dates, results_voltarget["equity_curve"], label="Volatility-Targeted Strategy", 
             color="#00a8ff", linewidth=2.0)
    
    stats_text = (
        "Crossover Strategy:\n"
        f"  Total Return: {results_crossover['metrics']['Total Return']*100:.1f}%\n"
        f"  CAGR: {results_crossover['metrics']['CAGR']*100:.1f}%\n"
        f"  Sharpe: {results_crossover['metrics']['Sharpe Ratio']:.2f}\n"
        f"  Max DD: {results_crossover['metrics']['Max Drawdown']*100:.1f}%\n\n"
        "Vol-Target Strategy:\n"
        f"  Total Return: {results_voltarget['metrics']['Total Return']*100:.1f}%\n"
        f"  CAGR: {results_voltarget['metrics']['CAGR']*100:.1f}%\n"
        f"  Sharpe: {results_voltarget['metrics']['Sharpe Ratio']:.2f}\n"
        f"  Max DD: {results_voltarget['metrics']['Max Drawdown']*100:.1f}%"
    )
    ax1.text(0.02, 0.05, stats_text, transform=ax1.transAxes, fontsize=9.5,
             verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='#2f3542', alpha=0.85))
    
    ax1.set_title("Strategy Performance vs Benchmark", fontsize=14, fontweight="bold", pad=15)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.grid(True, color="#2f3542", linestyle="--", alpha=0.7)
    
    # Shade background by volatility regime
    for i in range(1, len(dates)):
        if regimes.iloc[i] == 2:
            ax1.axvspan(dates[i-1], dates[i], color="#ff4757", alpha=0.15)
        elif regimes.iloc[i] == 1:
            ax1.axvspan(dates[i-1], dates[i], color="#eccc68", alpha=0.08)
            
    # Subplot 2: Asset Price & Strategy weights
    ax2.plot(dates, data["Close"], color="#ffffff", alpha=0.9, linewidth=1.5, label="Asset Close Price")
    ax2.set_ylabel("Asset Price ($)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, color="#2f3542", linestyle="--", alpha=0.7)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(dates, results_crossover["positions"], color="#2ed573", alpha=0.6, 
                  linestyle="--", label="MA Crossover Allocation")
    ax2_twin.set_ylabel("Portfolio Allocation Weight", color="#2ed573", fontsize=10)
    ax2_twin.tick_params(axis='y', labelcolor="#2ed573")
    
    # Shade regimes on bottom plot too
    for i in range(1, len(dates)):
        if regimes.iloc[i] == 2:
            ax2.axvspan(dates[i-1], dates[i], color="#ff4757", alpha=0.15)
        elif regimes.iloc[i] == 1:
            ax2.axvspan(dates[i-1], dates[i], color="#eccc68", alpha=0.08)
            
    # Set up legends
    lines, labels = ax2.get_legend_handles_labels()
    lines_twin, labels_twin = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines_twin, labels + labels_twin, loc="upper left", frameon=True, facecolor="#2f3542")
    
    from matplotlib.patches import Patch
    shading_legends = [
        Patch(facecolor='none', edgecolor='none', label='Background Shading:'),
        Patch(facecolor='#eccc68', alpha=0.3, label='Medium Vol (Regime 1)'),
        Patch(facecolor='#ff4757', alpha=0.3, label='High Vol (Regime 2) - Crossover Flattened')
    ]
    ax1.legend(handles=list(ax1.get_legend_handles_labels()[0]) + shading_legends, 
               labels=list(ax1.get_legend_handles_labels()[1]) + [el.get_label() for el in shading_legends],
               loc="upper left", frameon=True, facecolor="#2f3542")
    
    plt.tight_layout()
    plot_path = "equity_curve.png"
    plt.savefig(plot_path, dpi=150, facecolor='#121212')
    plt.close()
    
    logger.info(f"Equity curve plot saved to '{os.path.abspath(plot_path)}'")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
