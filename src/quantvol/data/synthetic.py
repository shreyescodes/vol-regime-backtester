"""Synthetic OHLCV generator: regime-switching geometric Brownian motion."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

DEFAULT_TRANSITIONS = np.array(
    [
        [0.970, 0.025, 0.005],  # low vol
        [0.020, 0.950, 0.030],  # medium vol
        [0.015, 0.045, 0.940],  # high vol
    ]
)

DEFAULT_REGIME_PARAMS: dict[int, dict[str, float]] = {
    0: {"drift": 0.12, "vol": 0.10, "vol_scale": 0.8},
    1: {"drift": 0.02, "vol": 0.20, "vol_scale": 1.2},
    2: {"drift": -0.30, "vol": 0.45, "vol_scale": 2.5},
}


def generate_synthetic_data(
    n_days: int = 1000,
    seed: int = 42,
    start_date: str = "2020-01-01",
    trading_days: int = 252,
    base_volume: int = 1_000_000,
    transition_matrix: np.ndarray | None = None,
    regime_params: Mapping[int, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """Generate daily OHLCV bars from a Markov-switching GBM.

    The latent regime (``TrueRegime``) follows a Markov chain; each regime has its own
    annualized drift and volatility. Output is deterministic for a given ``seed`` and
    uses a local RNG, so no global NumPy state is touched.
    """
    trans = DEFAULT_TRANSITIONS if transition_matrix is None else np.asarray(transition_matrix)
    params = DEFAULT_REGIME_PARAMS if regime_params is None else regime_params
    if trans.ndim != 2 or trans.shape[0] != trans.shape[1]:
        raise ValueError("transition_matrix must be square")
    if not np.allclose(trans.sum(axis=1), 1.0):
        raise ValueError("each row of transition_matrix must sum to 1")
    if set(params) != set(range(trans.shape[0])):
        raise ValueError("regime_params keys must be 0..n_states-1")

    rng = np.random.default_rng(seed)
    dt = 1.0 / trading_days

    cum = trans.cumsum(axis=1)
    draws = rng.random(n_days)
    states = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        states[t] = min(int(np.searchsorted(cum[states[t - 1]], draws[t])), trans.shape[0] - 1)

    drift = np.array([params[s]["drift"] for s in states])
    vol = np.array([params[s]["vol"] for s in states])
    vol_scale = np.array([params[s]["vol_scale"] for s in states])

    log_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * rng.standard_normal(n_days)
    log_ret[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(log_ret))

    prev_close = np.concatenate(([100.0], close[:-1]))
    open_ = prev_close * np.exp(rng.normal(0.0, 0.001, n_days))
    wick = vol * np.sqrt(dt) * 0.7
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, wick)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, wick)))
    volume = np.floor(base_volume * vol_scale * rng.lognormal(0.0, 0.25, n_days))

    index = pd.date_range(start=start_date, periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "TrueRegime": states,
        },
        index=index,
    )
