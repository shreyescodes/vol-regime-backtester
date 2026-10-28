import numpy as np
import pandas as pd
import pytest

from quantvol.metrics import (
    calculate_performance,
    drawdown_series,
    performance_from_returns,
    regime_attribution,
)


def test_flat_returns_give_zeros() -> None:
    equity = pd.Series([100.0, 100.0, 100.0])
    m = calculate_performance(equity, pd.Series([0.0, 0.0, 0.0]))
    assert all(m[k] == 0.0 for k in ("Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown"))


def test_too_short_series_is_safe() -> None:
    m = calculate_performance(pd.Series([100.0]), pd.Series([0.0]))
    assert m["Total Return"] == 0.0


def test_golden_values() -> None:
    equity = pd.Series([100.0, 110.0, 99.0, 108.9])
    returns = equity.pct_change().fillna(0.0)
    m = calculate_performance(equity, returns, trading_days=252)
    assert m["Total Return"] == pytest.approx(0.089)
    assert m["Max Drawdown"] == pytest.approx(-0.10)
    years = 3 / 252
    assert m["CAGR"] == pytest.approx(1.089 ** (1 / years) - 1)
    expected_sharpe = returns.mean() / returns.std() * np.sqrt(252)
    assert m["Sharpe Ratio"] == pytest.approx(expected_sharpe)
    assert m["Calmar Ratio"] == pytest.approx(m["CAGR"] / 0.10)


def test_risk_free_rate_lowers_sharpe() -> None:
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0008, 0.01, 500))
    no_rf = performance_from_returns(r, risk_free_rate=0.0)["Sharpe Ratio"]
    with_rf = performance_from_returns(r, risk_free_rate=0.05)["Sharpe Ratio"]
    assert with_rf < no_rf


def test_sortino_ignores_upside_volatility() -> None:
    down = pd.Series([0.01, -0.01] * 100)
    up_heavy = pd.Series([0.05, -0.01] * 100)
    assert (
        performance_from_returns(up_heavy)["Sortino Ratio"]
        > performance_from_returns(down)["Sortino Ratio"]
    )


def test_var_cvar_ordering() -> None:
    r = pd.Series(np.random.default_rng(3).normal(0, 0.01, 1000))
    m = performance_from_returns(r)
    assert m["CVaR 95%"] <= m["VaR 95%"] < 0


def test_exposure_and_turnover() -> None:
    equity = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
    returns = equity.pct_change().fillna(0.0)
    pos = pd.Series([0.0, 1.0, 1.0, 0.5, 0.5])
    trades = pos.diff().abs().fillna(0.0)
    m = calculate_performance(equity, returns, positions=pos, trades=trades)
    assert m["Exposure"] == pytest.approx(0.6)
    assert m["Annual Turnover"] == pytest.approx(1.5 / (4 / 252))


def test_drawdown_never_positive() -> None:
    dd = drawdown_series(pd.Series([1.0, 2.0, 1.5, 3.0, 2.0]))
    assert (dd <= 0).all() and dd.iloc[-1] == pytest.approx(-1 / 3)


def test_regime_attribution_sharpe_undefined_for_flat_book() -> None:
    idx = pd.date_range("2022-01-01", periods=6, freq="B")
    ret = pd.Series([0.0, 0.0, 0.0, 0.01, -0.01, 0.02], index=idx)
    pos = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], index=idx)
    regimes = pd.Series([2, 2, 2, 0, 0, 0], index=idx)
    table = regime_attribution(ret, pos, regimes)
    assert np.isnan(table.loc["High Vol", "Sharpe"])
    assert table.loc["High Vol", "Avg Exposure"] == 0.0
    assert np.isfinite(table.loc["Low Vol", "Sharpe"])


def test_regime_attribution_partitions_all_days() -> None:
    idx = pd.date_range("2022-01-01", periods=9, freq="B")
    ret = pd.Series(np.linspace(-0.01, 0.01, 9), index=idx)
    pos = pd.Series(1.0, index=idx)
    regimes = pd.Series([0, 0, 0, 1, 1, 1, 2, 2, 2], index=idx)
    table = regime_attribution(ret, pos, regimes)
    assert table["Days"].sum() == 9
    assert table["% of Time"].sum() == pytest.approx(1.0)
    assert list(table.index) == ["Low Vol", "Medium Vol", "High Vol"]
