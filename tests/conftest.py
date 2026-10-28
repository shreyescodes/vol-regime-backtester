import numpy as np
import pandas as pd
import pytest

from quantvol.data import generate_synthetic_data
from quantvol.regimes import detect_regimes


@pytest.fixture(scope="session")
def market() -> pd.DataFrame:
    return generate_synthetic_data(n_days=800, seed=7)


@pytest.fixture(scope="session")
def regimes(market: pd.DataFrame) -> pd.Series:
    return detect_regimes(market, low_thresh=0.14, high_thresh=0.25)


@pytest.fixture
def trending() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    close = 100.0 * 1.002 ** np.arange(120)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close}, index=idx)
