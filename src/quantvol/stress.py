"""Stress scenarios.

Each shock is a pure function that returns a new price frame. Signals are regenerated on
the shocked path so the adaptive logic (regime detection included) is what gets tested.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace

import pandas as pd

from quantvol.config import RegimeConfig, StressConfig
from quantvol.engine import Backtester
from quantvol.regimes import detect_regimes
from quantvol.strategies.base import BaseStrategy


def apply_vol_spike(data: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Scale deviations of daily returns around their mean by ``factor``."""
    out = data.copy()
    ret = data["Close"].pct_change().fillna(0.0)
    mean = ret.mean()
    shocked = mean + factor * (ret - mean)
    shocked.iloc[0] = 0.0
    out["Close"] = data["Close"].iloc[0] * (1.0 + shocked).cumprod()
    out["High"] = out["Close"] * (1.0 + (data["High"] / data["Close"] - 1.0) * factor)
    out["Low"] = out["Close"] * (1.0 - (1.0 - data["Low"] / data["Close"]) * factor)
    return out


def apply_flash_crash(data: pd.DataFrame, drop: float, day_idx: int) -> pd.DataFrame:
    """Apply a one-day price shock at ``day_idx`` that persists afterwards."""
    if not 0 <= day_idx < len(data):
        raise ValueError(f"flash-crash day {day_idx} is outside the sample (0..{len(data) - 1})")
    out = data.copy()
    scale = pd.Series(1.0, index=data.index)
    scale.iloc[day_idx:] = 1.0 + drop
    for col in ("Open", "High", "Low", "Close"):
        out[col] = data[col] * scale
    return out


def run_stress_tests(
    backtester: Backtester,
    data: pd.DataFrame,
    strategy_factory: Callable[[], BaseStrategy],
    regime_config: RegimeConfig,
    stress_config: StressConfig,
) -> pd.DataFrame:
    """Run baseline plus vol-spike, flash-crash and cost-surge scenarios.

    Returns one row of metrics per scenario. The supplied ``backtester`` is never mutated;
    the cost scenario runs on a modified copy.
    """
    regime_kwargs = asdict(regime_config)
    crash_day = (
        stress_config.flash_crash_day
        if stress_config.flash_crash_day is not None
        else len(data) // 2
    )
    cost_engine = replace(
        backtester,
        commission_rate=backtester.commission_rate * stress_config.cost_multiplier,
        slippage_rate=backtester.slippage_rate * stress_config.cost_multiplier,
    )
    scenarios: dict[str, tuple[Backtester, pd.DataFrame]] = {
        "Baseline": (backtester, data),
        "Vol Spike": (backtester, apply_vol_spike(data, stress_config.vol_spike_factor)),
        "Flash Crash": (
            backtester,
            apply_flash_crash(data, stress_config.flash_crash_drop, crash_day),
        ),
        "Cost Surge": (cost_engine, data),
    }
    rows = {}
    for name, (engine, frame) in scenarios.items():
        regimes = detect_regimes(frame, **regime_kwargs)
        signals = strategy_factory().generate_signals(frame, regimes)
        rows[name] = engine.run(frame, signals).metrics
    return pd.DataFrame(rows).T
