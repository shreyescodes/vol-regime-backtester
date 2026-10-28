import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantvol import Backtester, RegimeFilteredMovingAverageCrossover, VolatilityTargetingStrategy


def _frame(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"Close": prices}, index=idx)


def test_hand_computed_pnl_and_costs() -> None:
    data = _frame([100.0, 110.0, 99.0, 99.0])
    signals = pd.Series([1.0, 1.0, 0.0, 0.0], index=data.index)
    engine = Backtester(initial_capital=1000.0, commission_rate=0.01, slippage_rate=0.0)
    res = engine.run(data, signals)
    # positions after 1-bar lag: [0, 1, 1, 0]; trades: [0, 1, 0, 1]
    assert list(res.positions) == [0.0, 1.0, 1.0, 0.0]
    assert list(res.trades) == [0.0, 1.0, 0.0, 1.0]
    expected = 1000.0 * (1 - 0.0 + 0.1 - 0.01) * (1 - 0.1) * (1 - 0.01)
    assert res.equity_curve.iloc[-1] == pytest.approx(expected)


def test_flat_strategy_earns_nothing() -> None:
    data = _frame(list(100 + np.arange(50.0)))
    res = Backtester().run(data, pd.Series(0.0, index=data.index))
    assert (res.equity_curve == 100_000.0).all()
    assert res.metrics["Total Return"] == 0.0


def test_always_long_matches_benchmark_when_costless_after_warmup() -> None:
    data = _frame(list(100 * 1.001 ** np.arange(60)))
    engine = Backtester(commission_rate=0.0, slippage_rate=0.0)
    res = engine.run(data, pd.Series(1.0, index=data.index))
    # the strategy's first bar is flat because of the lag; afterwards returns are identical
    pd.testing.assert_series_equal(
        res.returns.iloc[1:], res.benchmark_returns.iloc[1:], check_names=False
    )


def test_benchmark_pays_entry_cost() -> None:
    data = _frame([100.0, 100.0, 100.0])
    engine = Backtester(initial_capital=1000.0, commission_rate=0.001, slippage_rate=0.0005)
    res = engine.run(data, pd.Series(0.0, index=data.index))
    assert res.benchmark_curve.iloc[0] == pytest.approx(1000.0 * (1 - 0.0015))
    assert res.benchmark_metrics["Total Return"] == pytest.approx(0.0, abs=0.0016)


def test_run_does_not_mutate_inputs(market: pd.DataFrame, regimes: pd.Series) -> None:
    snapshot = market.copy()
    signals = RegimeFilteredMovingAverageCrossover().generate_signals(market, regimes)
    sig_snapshot = signals.copy()
    Backtester().run(market, signals)
    pd.testing.assert_frame_equal(market, snapshot)
    pd.testing.assert_series_equal(signals, sig_snapshot)


def test_higher_costs_reduce_returns(market: pd.DataFrame, regimes: pd.Series) -> None:
    signals = RegimeFilteredMovingAverageCrossover(5, 20).generate_signals(market, regimes)
    cheap = Backtester(commission_rate=0.0, slippage_rate=0.0).run(market, signals)
    pricey = Backtester(commission_rate=0.005, slippage_rate=0.005).run(market, signals)
    assert pricey.metrics["Total Return"] < cheap.metrics["Total Return"]


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    fast=st.integers(3, 15),
    slow=st.integers(20, 60),
    target=st.floats(0.05, 0.4),
)
def test_weights_bounded_and_equity_positive(
    seed: int, fast: int, slow: int, target: float
) -> None:
    from quantvol.data import generate_synthetic_data
    from quantvol.regimes import detect_regimes

    data = generate_synthetic_data(n_days=300, seed=seed)
    regimes = detect_regimes(data)
    for strat in (
        RegimeFilteredMovingAverageCrossover(fast, slow),
        VolatilityTargetingStrategy(slow, 20, target),
    ):
        sig = strat.generate_signals(data, regimes)
        assert sig.between(0.0, 1.0).all()
        assert not sig.isna().any()
        res = Backtester().run(data, sig)
        assert (res.equity_curve > 0).all()
        assert res.metrics["Max Drawdown"] <= 0.0
