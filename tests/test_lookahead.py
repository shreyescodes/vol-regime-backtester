"""Causality guarantees: nothing observed after bar t may influence anything decided at t."""

import numpy as np
import pandas as pd
import pytest

from quantvol import Backtester, RegimeFilteredMovingAverageCrossover, VolatilityTargetingStrategy
from quantvol.regimes import detect_regimes

CUT = 400


def _corrupt_future(df: pd.DataFrame, cut: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(0)
    for col in ("Open", "High", "Low", "Close"):
        out.iloc[cut + 1 :, out.columns.get_loc(col)] *= rng.uniform(0.5, 1.5, len(out) - cut - 1)
    return out


def test_regimes_ignore_future(market: pd.DataFrame) -> None:
    base = detect_regimes(market, low_thresh=0.14, high_thresh=0.25)
    pert = detect_regimes(_corrupt_future(market, CUT), low_thresh=0.14, high_thresh=0.25)
    pd.testing.assert_series_equal(base.iloc[: CUT + 1], pert.iloc[: CUT + 1])


@pytest.mark.parametrize(
    "strategy",
    [RegimeFilteredMovingAverageCrossover(10, 40), VolatilityTargetingStrategy(40, 20, 0.12)],
    ids=lambda s: s.name,
)
def test_signals_ignore_future(market: pd.DataFrame, strategy) -> None:  # type: ignore[no-untyped-def]
    pert = _corrupt_future(market, CUT)
    s1 = strategy.generate_signals(market, detect_regimes(market))
    s2 = strategy.generate_signals(pert, detect_regimes(pert))
    pd.testing.assert_series_equal(s1.iloc[: CUT + 1], s2.iloc[: CUT + 1])


def test_position_at_t_never_uses_return_at_t(market: pd.DataFrame, regimes: pd.Series) -> None:
    strat = RegimeFilteredMovingAverageCrossover(10, 40)
    signals = strat.generate_signals(market, regimes)
    result = Backtester().run(market, signals)
    # target weight decided at close t is held during t+1, never during t
    expected = signals.shift(1).fillna(0.0)
    pd.testing.assert_series_equal(result.positions, expected, check_names=False)


def test_future_shock_does_not_change_past_equity(market: pd.DataFrame, regimes: pd.Series) -> None:
    strat = RegimeFilteredMovingAverageCrossover(10, 40)
    engine = Backtester()
    base = engine.run(market, strat.generate_signals(market, regimes))
    pert_data = _corrupt_future(market, CUT)
    pert = engine.run(pert_data, strat.generate_signals(pert_data, detect_regimes(pert_data)))
    np.testing.assert_allclose(base.equity_curve.iloc[: CUT + 1], pert.equity_curve.iloc[: CUT + 1])


def test_zero_lag_is_rejected() -> None:
    with pytest.raises(ValueError, match="look-ahead"):
        Backtester(execution_lag=0)
