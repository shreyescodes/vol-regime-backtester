"""Vectorized backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantvol.metrics import calculate_performance


@dataclass(frozen=True)
class BacktestResult:
    """Output of a single backtest run."""

    equity_curve: pd.Series
    benchmark_curve: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series
    positions: pd.Series
    trades: pd.Series
    metrics: dict[str, float]
    benchmark_metrics: dict[str, float]


@dataclass(frozen=True)
class Backtester:
    """Long-only, unleveraged daily backtester.

    Target weights from a strategy are delayed by ``execution_lag`` bars (>= 1) before they
    earn returns, which rules out same-bar look-ahead. Each unit of turnover pays
    ``commission_rate + slippage_rate``. The buy-and-hold benchmark pays the entry cost once.
    """

    initial_capital: float = 100_000.0
    commission_rate: float = 0.0010
    slippage_rate: float = 0.0005
    risk_free_rate: float = 0.0
    execution_lag: int = 1
    trading_days: int = 252

    def __post_init__(self) -> None:
        if self.execution_lag < 1:
            raise ValueError("execution_lag must be >= 1; 0 would introduce look-ahead bias")

    @property
    def friction(self) -> float:
        return self.commission_rate + self.slippage_rate

    def run(self, data: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        asset_returns = data["Close"].pct_change().fillna(0.0)
        positions = signals.reindex(data.index).shift(self.execution_lag).fillna(0.0)

        trades = positions.diff().abs()
        trades.iloc[0] = abs(positions.iloc[0])
        net_returns = positions * asset_returns - trades * self.friction
        equity = self.initial_capital * (1.0 + net_returns).cumprod()

        bench_returns = asset_returns.copy()
        bench_returns.iloc[0] = -self.friction
        bench_equity = self.initial_capital * (1.0 + bench_returns).cumprod()

        td, rf = self.trading_days, self.risk_free_rate
        return BacktestResult(
            equity_curve=equity,
            benchmark_curve=bench_equity,
            returns=net_returns,
            benchmark_returns=bench_returns,
            positions=positions,
            trades=trades,
            metrics=calculate_performance(
                equity, net_returns, td, rf, positions=positions, trades=trades
            ),
            benchmark_metrics=calculate_performance(bench_equity, bench_returns, td, rf),
        )
