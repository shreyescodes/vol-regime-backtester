"""Parameter sweeps and walk-forward validation for the crossover strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from quantvol.engine import Backtester
from quantvol.metrics import performance_from_returns
from quantvol.strategies.ma_crossover import RegimeFilteredMovingAverageCrossover


def _sharpe_of_window(
    backtester: Backtester,
    data: pd.DataFrame,
    regimes: pd.Series,
    fast: int,
    slow: int,
    flat_regimes: Sequence[int],
    start: int,
    end: int,
) -> float:
    """Sharpe of bars ``[start, end)`` using only data before ``end``.

    The strategy is run on ``data[:end]`` so earlier bars warm up the indicators, but nothing
    after ``end`` can influence the result.
    """
    strategy = RegimeFilteredMovingAverageCrossover(fast, slow, flat_regimes)
    window = data.iloc[:end]
    result = backtester.run(window, strategy.generate_signals(window, regimes.iloc[:end]))
    stats = performance_from_returns(
        result.returns.iloc[start:end], backtester.trading_days, backtester.risk_free_rate
    )
    return stats["Sharpe Ratio"]


def parameter_sweep(
    backtester: Backtester,
    data: pd.DataFrame,
    regimes: pd.Series,
    fast_grid: Sequence[int],
    slow_grid: Sequence[int],
    flat_regimes: Sequence[int] = (2,),
) -> pd.DataFrame:
    """Full-sample Sharpe for every valid ``(fast, slow)`` pair (NaN where fast >= slow)."""
    grid = pd.DataFrame(index=pd.Index(fast_grid, name="Fast MA"), columns=list(slow_grid))
    grid.columns.name = "Slow MA"
    grid = grid.astype(float)
    for fast in fast_grid:
        for slow in slow_grid:
            if fast < slow:
                grid.loc[fast, slow] = _sharpe_of_window(
                    backtester, data, regimes, fast, slow, flat_regimes, 0, len(data)
                )
    return grid


@dataclass(frozen=True)
class WalkForwardResult:
    folds: pd.DataFrame
    oos_returns: pd.Series
    oos_equity: pd.Series
    oos_metrics: dict[str, float]


def walk_forward(
    backtester: Backtester,
    data: pd.DataFrame,
    regimes: pd.Series,
    train_days: int,
    test_days: int,
    fast_grid: Sequence[int],
    slow_grid: Sequence[int],
    flat_regimes: Sequence[int] = (2,),
) -> WalkForwardResult:
    """Rolling walk-forward: pick the best in-sample Sharpe, then trade it out-of-sample.

    Each fold selects ``(fast, slow)`` on ``train_days`` of history and applies it to the next
    ``test_days`` bars. Stitched out-of-sample returns give an honest estimate of how the
    parameter-selection procedure itself would have performed.
    """
    candidates = [(f, s) for f in fast_grid for s in slow_grid if f < s]
    if not candidates:
        raise ValueError("parameter grid contains no valid fast < slow pair")
    if train_days + test_days > len(data):
        raise ValueError("train_days + test_days exceeds the available history")

    rows = []
    pieces = []
    start = 0
    while start + train_days + test_days <= len(data):
        train_end = start + train_days
        test_end = train_end + test_days
        scores = {
            (f, s): _sharpe_of_window(
                backtester, data, regimes, f, s, flat_regimes, start, train_end
            )
            for f, s in candidates
        }
        best = max(scores, key=lambda k: scores[k])
        full = backtester.run(
            data.iloc[:test_end],
            RegimeFilteredMovingAverageCrossover(*best, flat_regimes).generate_signals(
                data.iloc[:test_end], regimes.iloc[:test_end]
            ),
        )
        oos = full.returns.iloc[train_end:test_end]
        pieces.append(oos)
        rows.append(
            {
                "Train Start": data.index[start],
                "Test Start": data.index[train_end],
                "Test End": data.index[test_end - 1],
                "Fast": best[0],
                "Slow": best[1],
                "In-Sample Sharpe": scores[best],
                "OOS Sharpe": performance_from_returns(
                    oos, backtester.trading_days, backtester.risk_free_rate
                )["Sharpe Ratio"],
            }
        )
        start += test_days

    oos_returns = pd.concat(pieces)
    equity = backtester.initial_capital * (1.0 + oos_returns).cumprod()
    return WalkForwardResult(
        folds=pd.DataFrame(rows),
        oos_returns=oos_returns,
        oos_equity=equity,
        oos_metrics=performance_from_returns(
            oos_returns, backtester.trading_days, backtester.risk_free_rate
        ),
    )
