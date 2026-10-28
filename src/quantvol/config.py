"""Typed, validated configuration for QuantVol runs.

Every tunable lives in a frozen dataclass so a run is fully described by one
YAML file. Unknown keys and out-of-range values fail fast with a clear message.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration file or value is invalid."""


@dataclass(frozen=True)
class DataConfig:
    source: str = "synthetic"  # synthetic | csv | yahoo
    n_days: int = 1000
    seed: int = 101
    start_date: str = "2020-01-01"
    base_volume: int = 1_000_000
    csv_path: str | None = None
    ticker: str | None = None
    end_date: str | None = None

    def __post_init__(self) -> None:
        if self.source not in {"synthetic", "csv", "yahoo"}:
            raise ConfigError(f"data.source must be synthetic|csv|yahoo, got {self.source!r}")
        if self.source == "synthetic" and self.n_days < 100:
            raise ConfigError("data.n_days must be >= 100")
        if self.source == "csv" and not self.csv_path:
            raise ConfigError("data.csv_path is required when data.source is 'csv'")
        if self.source == "yahoo" and not self.ticker:
            raise ConfigError("data.ticker is required when data.source is 'yahoo'")


@dataclass(frozen=True)
class RegimeConfig:
    vol_window: int = 20
    low_thresh: float = 0.14
    high_thresh: float = 0.25
    trading_days: int = 252

    def __post_init__(self) -> None:
        if self.vol_window < 2:
            raise ConfigError("regime.vol_window must be >= 2")
        if not 0 < self.low_thresh < self.high_thresh:
            raise ConfigError("regime thresholds must satisfy 0 < low_thresh < high_thresh")


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_rate: float = 0.0010
    slippage_rate: float = 0.0005
    risk_free_rate: float = 0.04
    execution_lag: int = 1
    trading_days: int = 252

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ConfigError("backtest.initial_capital must be positive")
        if self.commission_rate < 0 or self.slippage_rate < 0:
            raise ConfigError("backtest costs must be non-negative")
        if self.execution_lag < 1:
            raise ConfigError("backtest.execution_lag must be >= 1 (0 would be look-ahead)")


@dataclass(frozen=True)
class CrossoverConfig:
    fast_window: int = 20
    slow_window: int = 50
    flat_regimes: tuple[int, ...] = (2,)


@dataclass(frozen=True)
class VolTargetConfig:
    momentum_window: int = 50
    vol_window: int = 20
    target_vol: float = 0.12
    trading_days: int = 252


@dataclass(frozen=True)
class StressConfig:
    vol_spike_factor: float = 2.0
    flash_crash_drop: float = -0.15
    flash_crash_day: int | None = None  # None -> middle of the sample
    cost_multiplier: float = 5.0


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 378
    test_days: int = 126
    fast_grid: tuple[int, ...] = (5, 10, 20, 30)
    slow_grid: tuple[int, ...] = (40, 60, 90, 120)


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    crossover: CrossoverConfig = field(default_factory=CrossoverConfig)
    vol_target: VolTargetConfig = field(default_factory=VolTargetConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)


def _build(cls: type[T], raw: Mapping[str, Any] | None, section: str) -> T:
    raw = dict(raw or {})
    valid = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(raw) - set(valid))
    if unknown:
        raise ConfigError(f"unknown key(s) in '{section}': {', '.join(unknown)}")
    for name, value in raw.items():
        if isinstance(value, list):
            raw[name] = tuple(value)
    try:
        return cls(**raw)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ConfigError(f"invalid '{section}' section: {exc}") from exc


def config_from_dict(raw: Mapping[str, Any]) -> Config:
    """Build a :class:`Config` from a nested mapping (e.g. parsed YAML)."""
    sections = {f.name: f.type for f in dataclasses.fields(Config)}
    unknown = sorted(set(raw) - set(sections))
    if unknown:
        raise ConfigError(f"unknown top-level section(s): {', '.join(unknown)}")
    return Config(
        data=_build(DataConfig, raw.get("data"), "data"),
        regime=_build(RegimeConfig, raw.get("regime"), "regime"),
        backtest=_build(BacktestConfig, raw.get("backtest"), "backtest"),
        crossover=_build(CrossoverConfig, raw.get("crossover"), "crossover"),
        vol_target=_build(VolTargetConfig, raw.get("vol_target"), "vol_target"),
        stress=_build(StressConfig, raw.get("stress"), "stress"),
        walk_forward=_build(WalkForwardConfig, raw.get("walk_forward"), "walk_forward"),
    )


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML configuration file."""
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a mapping")
    return config_from_dict(raw)
