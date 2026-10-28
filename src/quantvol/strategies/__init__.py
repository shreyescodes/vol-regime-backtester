"""Trading strategies."""

from quantvol.strategies.base import BaseStrategy
from quantvol.strategies.ma_crossover import RegimeFilteredMovingAverageCrossover
from quantvol.strategies.vol_target import VolatilityTargetingStrategy

__all__ = [
    "BaseStrategy",
    "RegimeFilteredMovingAverageCrossover",
    "VolatilityTargetingStrategy",
]
