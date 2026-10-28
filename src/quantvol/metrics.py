"""Performance and risk statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantvol.regimes import REGIME_LABELS

MIN_ANNUAL_VOL = 0.005
CORE_KEYS = ["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio"]


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak (<= 0)."""
    return equity_curve / equity_curve.cummax() - 1.0


def calculate_performance(
    equity_curve: pd.Series,
    returns: pd.Series,
    trading_days: int = 252,
    risk_free_rate: float = 0.0,
    positions: pd.Series | None = None,
    trades: pd.Series | None = None,
) -> dict[str, float]:
    """Summarize a return stream.

    Core keys (always present): Total Return, CAGR, Sharpe Ratio, Max Drawdown, Calmar Ratio.
    Extended keys: Annual Volatility, Sortino Ratio, VaR 95%, CVaR 95%, Win Rate and, when
    ``positions`` / ``trades`` are supplied, Exposure and Annual Turnover.
    """
    metrics = dict.fromkeys(CORE_KEYS, 0.0)
    metrics.update(
        {"Annual Volatility": 0.0, "Sortino Ratio": 0.0, "VaR 95%": 0.0, "CVaR 95%": 0.0}
    )
    metrics["Win Rate"] = 0.0
    if len(equity_curve) < 2:
        return metrics

    initial, final = float(equity_curve.iloc[0]), float(equity_curve.iloc[-1])
    years = (len(equity_curve) - 1) / float(trading_days)
    metrics["Total Return"] = (final - initial) / initial
    if initial > 0 and final > 0 and years > 0:
        metrics["CAGR"] = (final / initial) ** (1.0 / years) - 1.0

    daily_rf = risk_free_rate / float(trading_days)
    std = float(returns.std())
    if std > 0:
        metrics["Sharpe Ratio"] = (float(returns.mean()) - daily_rf) / std * np.sqrt(trading_days)
        metrics["Annual Volatility"] = std * np.sqrt(trading_days)

    downside = np.minimum(returns - daily_rf, 0.0)
    downside_dev = float(np.sqrt((downside**2).mean()))
    if downside_dev > 0:
        metrics["Sortino Ratio"] = (
            (float(returns.mean()) - daily_rf) / downside_dev * np.sqrt(trading_days)
        )

    max_dd = float(drawdown_series(equity_curve).min())
    metrics["Max Drawdown"] = max_dd
    if abs(max_dd) > 1e-4:
        metrics["Calmar Ratio"] = metrics["CAGR"] / abs(max_dd)

    var = float(returns.quantile(0.05))
    metrics["VaR 95%"] = var
    tail = returns[returns <= var]
    metrics["CVaR 95%"] = float(tail.mean()) if len(tail) else var

    active = returns[returns != 0.0]
    if len(active):
        metrics["Win Rate"] = float((active > 0).mean())

    if positions is not None:
        metrics["Exposure"] = float(positions.abs().mean())
    if trades is not None and years > 0:
        metrics["Annual Turnover"] = float(trades.sum() / years)
    return metrics


def performance_from_returns(
    returns: pd.Series, trading_days: int = 252, risk_free_rate: float = 0.0
) -> dict[str, float]:
    """Convenience wrapper: build a unit equity curve from ``returns`` and summarize it."""
    equity = pd.concat([pd.Series([1.0]), (1.0 + returns.reset_index(drop=True)).cumprod()])
    return calculate_performance(
        equity.reset_index(drop=True),
        returns,
        trading_days=trading_days,
        risk_free_rate=risk_free_rate,
    )


def regime_attribution(
    returns: pd.Series,
    positions: pd.Series,
    regimes: pd.Series,
    trading_days: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Break strategy performance down by the volatility regime of each realized bar."""
    frame = pd.DataFrame({"ret": returns, "pos": positions.abs(), "regime": regimes}).dropna()
    daily_rf = risk_free_rate / float(trading_days)
    rows = {}
    for regime, label in REGIME_LABELS.items():
        sub = frame[frame["regime"] == regime]
        if sub.empty:
            rows[label] = {
                "Days": 0,
                "% of Time": 0.0,
                "Ann. Return": 0.0,
                "Ann. Vol": 0.0,
                "Sharpe": 0.0,
                "Avg Exposure": 0.0,
            }
            continue
        std = float(sub["ret"].std()) if len(sub) > 1 else 0.0
        ann_vol = std * np.sqrt(trading_days)
        # A flat book has ~zero volatility, which would make Sharpe a meaningless huge number.
        sharpe = (
            (float(sub["ret"].mean()) - daily_rf) / std * np.sqrt(trading_days)
            if ann_vol >= MIN_ANNUAL_VOL
            else float("nan")
        )
        rows[label] = {
            "Days": len(sub),
            "% of Time": len(sub) / len(frame),
            "Ann. Return": float(sub["ret"].mean()) * trading_days,
            "Ann. Vol": ann_vol,
            "Sharpe": sharpe,
            "Avg Exposure": float(sub["pos"].mean()),
        }
    return pd.DataFrame(rows).T
