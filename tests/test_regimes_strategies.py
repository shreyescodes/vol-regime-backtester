import numpy as np
import pandas as pd
import pytest

from quantvol import (
    RegimeFilteredMovingAverageCrossover,
    VolatilityTargetingStrategy,
    detect_regimes,
)
from quantvol.regimes import transition_matrix


def test_regime_classification_follows_volatility() -> None:
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.001, 60)
    wild = rng.normal(0, 0.05, 60)
    close = 100 * np.exp(np.cumsum(np.concatenate([calm, wild])))
    df = pd.DataFrame({"Close": close}, index=pd.date_range("2020-01-01", periods=120, freq="B"))
    regimes = detect_regimes(df, vol_window=10, low_thresh=0.05, high_thresh=0.15)
    assert regimes.iloc[0] == 1  # warm-up defaults to medium, never back-filled from the future
    assert regimes.iloc[50] == 0
    assert regimes.iloc[-1] == 2
    assert set(regimes.unique()) <= {0, 1, 2}


def test_invalid_thresholds_rejected(trending: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        detect_regimes(trending, low_thresh=0.3, high_thresh=0.2)


def test_transition_matrix_rows_sum_to_one(regimes: pd.Series) -> None:
    probs = transition_matrix(regimes)
    sums = probs.sum(axis=1)
    assert ((sums - 1).abs() < 1e-9).all() or (sums == 0).any()
    assert probs.shape == (3, 3)


def test_crossover_long_in_trend_flat_in_high_vol(trending: pd.DataFrame) -> None:
    strat = RegimeFilteredMovingAverageCrossover(5, 10)
    calm = pd.Series(0, index=trending.index)
    stressed = pd.Series(2, index=trending.index)
    assert strat.generate_signals(trending, calm).iloc[-1] == 1.0
    assert (strat.generate_signals(trending, stressed) == 0.0).all()


def test_crossover_is_flat_during_warmup(trending: pd.DataFrame) -> None:
    sig = RegimeFilteredMovingAverageCrossover(5, 30).generate_signals(
        trending, pd.Series(0, index=trending.index)
    )
    assert (sig.iloc[:29] == 0.0).all()


def test_crossover_param_validation() -> None:
    with pytest.raises(ValueError):
        RegimeFilteredMovingAverageCrossover(50, 20)


def test_vol_target_scales_inversely_with_volatility() -> None:
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(5)
    drift = np.linspace(0, 0.5, 200)
    quiet = pd.DataFrame(
        {"Close": 100 * np.exp(drift + np.cumsum(rng.normal(0, 0.003, 200)))}, index=idx
    )
    noisy = pd.DataFrame(
        {"Close": 100 * np.exp(drift + np.cumsum(rng.normal(0, 0.03, 200)))}, index=idx
    )
    strat = VolatilityTargetingStrategy(20, 20, 0.15)
    reg = pd.Series(1, index=idx)
    assert (
        strat.generate_signals(quiet, reg).iloc[-30:].mean()
        > strat.generate_signals(noisy, reg).iloc[-30:].mean()
    )


def test_vol_target_zero_before_vol_is_known(trending: pd.DataFrame) -> None:
    sig = VolatilityTargetingStrategy(10, 20, 0.1).generate_signals(
        trending, pd.Series(0, index=trending.index)
    )
    assert (sig.iloc[:20] == 0.0).all()


def test_vol_target_param_validation() -> None:
    with pytest.raises(ValueError):
        VolatilityTargetingStrategy(target_vol=0)
