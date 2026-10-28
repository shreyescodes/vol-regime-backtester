import dataclasses
from dataclasses import asdict

import pandas as pd
import pytest

from quantvol import Backtester, RegimeFilteredMovingAverageCrossover
from quantvol.config import RegimeConfig, StressConfig
from quantvol.stress import apply_flash_crash, apply_vol_spike, run_stress_tests
from quantvol.walkforward import parameter_sweep, walk_forward


def test_stress_does_not_mutate_backtester_or_data(market: pd.DataFrame) -> None:
    engine = Backtester()
    before = dataclasses.replace(engine)
    snapshot = market.copy()
    run_stress_tests(
        engine,
        market,
        lambda: RegimeFilteredMovingAverageCrossover(10, 40),
        RegimeConfig(),
        StressConfig(),
    )
    assert engine == before
    pd.testing.assert_frame_equal(market, snapshot)


def test_stress_covers_all_scenarios(market: pd.DataFrame) -> None:
    out = run_stress_tests(
        Backtester(),
        market,
        lambda: RegimeFilteredMovingAverageCrossover(10, 40),
        RegimeConfig(),
        StressConfig(),
    )
    assert list(out.index) == ["Baseline", "Vol Spike", "Flash Crash", "Cost Surge"]
    assert out.loc["Cost Surge", "Total Return"] < out.loc["Baseline", "Total Return"]


def test_flash_crash_applies_one_day_drop_and_persists(market: pd.DataFrame) -> None:
    out = apply_flash_crash(market, -0.2, 300)
    ratio = out["Close"] / market["Close"]
    assert (ratio.iloc[:300] == 1.0).all()
    assert ratio.iloc[300:].to_numpy() == pytest.approx(0.8)
    assert out["Close"].pct_change().iloc[300] == pytest.approx(
        0.8 * market["Close"].iloc[300] / market["Close"].iloc[299] - 1
    )


def test_flash_crash_rejects_out_of_range_day(market: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        apply_flash_crash(market, -0.1, len(market))


def test_vol_spike_scales_volatility(market: pd.DataFrame) -> None:
    out = apply_vol_spike(market, 2.0)
    ratio = out["Close"].pct_change().std() / market["Close"].pct_change().std()
    assert ratio == pytest.approx(2.0, rel=0.05)


def test_parameter_sweep_masks_invalid_pairs(market: pd.DataFrame, regimes: pd.Series) -> None:
    grid = parameter_sweep(Backtester(), market, regimes, [10, 50], [20, 50])
    assert pd.isna(grid.loc[50, 20]) and pd.isna(grid.loc[50, 50])
    assert not pd.isna(grid.loc[10, 20])


def test_walk_forward_is_out_of_sample_and_contiguous(
    market: pd.DataFrame, regimes: pd.Series
) -> None:
    res = walk_forward(Backtester(), market, regimes, 300, 100, [5, 10], [30, 60])
    assert len(res.folds) == 5
    assert res.oos_returns.index.is_monotonic_increasing
    assert res.oos_returns.index.is_unique
    assert res.oos_returns.index[0] == market.index[300]
    assert (res.folds["Test Start"] > res.folds["Train Start"]).all()


def test_walk_forward_training_cannot_see_the_test_window(
    market: pd.DataFrame, regimes: pd.Series
) -> None:
    base = walk_forward(Backtester(), market, regimes, 300, 100, [5, 10], [30, 60])
    pert = market.copy()
    pert.iloc[400:, pert.columns.get_loc("Close")] *= 1.7  # shock inside later test windows
    other = walk_forward(Backtester(), pert, regimes, 300, 100, [5, 10], [30, 60])
    first = base.folds.iloc[0]
    assert (first[["Fast", "Slow"]] == other.folds.iloc[0][["Fast", "Slow"]]).all()
    assert first["In-Sample Sharpe"] == pytest.approx(other.folds.iloc[0]["In-Sample Sharpe"])


def test_walk_forward_validates_inputs(market: pd.DataFrame, regimes: pd.Series) -> None:
    with pytest.raises(ValueError):
        walk_forward(Backtester(), market, regimes, 700, 700, [5], [30])
    with pytest.raises(ValueError):
        walk_forward(Backtester(), market, regimes, 100, 50, [60], [30])


def test_regime_config_asdict_matches_detect_signature() -> None:
    from quantvol.regimes import detect_regimes

    detect_regimes(pd.DataFrame({"Close": [1.0, 1.1, 1.2]}), **asdict(RegimeConfig()))
