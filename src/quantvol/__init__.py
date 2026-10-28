"""QuantVol: regime-aware, vectorized systematic backtesting."""

from quantvol.config import Config, ConfigError, load_config
from quantvol.engine import Backtester, BacktestResult
from quantvol.regimes import detect_regimes
from quantvol.strategies import (
    BaseStrategy,
    RegimeFilteredMovingAverageCrossover,
    VolatilityTargetingStrategy,
)

__version__ = "1.0.0"

__all__ = [
    "BacktestResult",
    "Backtester",
    "BaseStrategy",
    "Config",
    "ConfigError",
    "RegimeFilteredMovingAverageCrossover",
    "VolatilityTargetingStrategy",
    "__version__",
    "detect_regimes",
    "load_config",
]
