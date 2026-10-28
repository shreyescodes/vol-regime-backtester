"""Matplotlib figures for backtest reports (headless-safe)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from quantvol.engine import BacktestResult
from quantvol.metrics import drawdown_series
from quantvol.regimes import REGIME_LABELS
from quantvol.walkforward import WalkForwardResult

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
RED = "#e34948"
SERIES = [BLUE, ORANGE, "#1baf7a"]
DIVERGING = LinearSegmentedColormap.from_list("blue_red", [RED, "#f0efec", BLUE])


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "sans-serif",
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def _shade_regimes(ax: Axes, regimes: pd.Series) -> None:
    """Shade contiguous high-volatility spans (one patch per span, not per bar)."""
    mask = (regimes == 2).to_numpy()
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    for start, stop in zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True):
        ax.axvspan(
            regimes.index[start],
            regimes.index[min(stop, len(regimes) - 1)],
            color=RED,
            alpha=0.14,
            lw=0,
        )


def plot_performance(
    regimes: pd.Series,
    results: Mapping[str, BacktestResult],
    title: str = "Strategy performance vs buy & hold",
) -> Figure:
    """Equity curves, drawdowns and exposure with volatility regimes shaded behind them."""
    _style()
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.4, 1.4]}
    )
    ax_eq, ax_dd, ax_pos = axes
    first = next(iter(results.values()))

    ax_eq.plot(
        first.benchmark_curve, color=GRAY, lw=1.6, label="Buy & hold", zorder=2, ls=(0, (4, 2))
    )
    ax_dd.plot(drawdown_series(first.benchmark_curve) * 100, color=GRAY, lw=1.4, ls=(0, (4, 2)))
    for color, (name, res) in zip(SERIES, results.items(), strict=False):
        ax_eq.plot(res.equity_curve, color=color, lw=2.0, label=name, zorder=3)
        ax_dd.plot(drawdown_series(res.equity_curve) * 100, color=color, lw=1.6)
        ax_pos.plot(res.positions, color=color, lw=1.4, drawstyle="steps-post")
    for ax in axes:
        _shade_regimes(ax, regimes)

    ax_eq.set_title(title, loc="left")
    ax_eq.set_ylabel("Portfolio value ($)")
    ax_eq.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    handles, labels = ax_eq.get_legend_handles_labels()
    handles.append(Patch(color=RED, alpha=0.3))
    labels.append("High-vol regime")
    ax_eq.legend(handles, labels, loc="upper left", ncol=2, fontsize=9)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_pos.set_ylabel("Exposure")
    ax_pos.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    return fig


def plot_stress(stress: pd.DataFrame) -> Figure:
    """Total return and max drawdown per stress scenario."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    panels = (("Total Return", "Total return (%)", BLUE), ("Max Drawdown", "Max drawdown (%)", RED))
    for ax, (col, label, color) in zip(axes, panels, strict=True):
        values = stress[col] * 100
        bars = ax.bar(stress.index, values, color=color, width=0.55)
        ax.bar_label(bars, fmt="%.1f", padding=3, color=INK_2, fontsize=9)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_title(label, loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Stress scenarios (signals regenerated on each shocked path)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_sweep(
    grid: pd.DataFrame, title: str = "Sharpe ratio across crossover parameters"
) -> Figure:
    """Annotated heatmap of Sharpe by (fast, slow) window, centred on zero."""
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    values = grid.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(values))) or 1.0
    im = ax.imshow(values, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), [str(c) for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [str(i) for i in grid.index])
    ax.grid(False)
    for (i, j), v in np.ndenumerate(values):
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=INK, fontsize=10)
    ax.set_xlabel("Slow MA window")
    ax.set_ylabel("Fast MA window")
    ax.set_title(title, loc="left")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.colorbar(im, ax=ax, label="Sharpe").outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_regime_attribution(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Annualized return and Sharpe by volatility regime for each named series."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = list(REGIME_LABELS.values())
    width = 0.8 / max(len(tables), 1)
    colors = [GRAY, *SERIES]
    for ax, (col, title, scale) in zip(
        axes,
        (("Ann. Return", "Annualized return (%)", 100), ("Sharpe", "Sharpe ratio", 1)),
        strict=True,
    ):
        for k, (name, table) in enumerate(tables.items()):
            x = np.arange(len(labels)) + (k - (len(tables) - 1) / 2) * width
            values = table.loc[labels, col] * scale
            ax.bar(x, values.fillna(0.0), width * 0.9, color=colors[k], label=name)
            for xi, missing in zip(x, values.isna(), strict=True):
                if missing:
                    ax.text(xi, 0, "n/a", ha="center", va="bottom", color=MUTED, fontsize=8)
        ax.set_xticks(range(len(labels)), labels)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def plot_walkforward(result: WalkForwardResult, benchmark_returns: pd.Series) -> Figure:
    """Stitched out-of-sample equity curve versus buy & hold over the same dates."""
    _style()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bench = (1.0 + benchmark_returns.loc[result.oos_returns.index]).cumprod()
    scale = float(result.oos_equity.iloc[0] / (1.0 + result.oos_returns.iloc[0]))
    ax.plot(bench * scale, color=GRAY, lw=1.6, ls=(0, (4, 2)), label="Buy & hold")
    ax.plot(result.oos_equity, color=BLUE, lw=2.0, label="Walk-forward (out-of-sample)")
    for date in result.folds["Test Start"]:
        ax.axvline(date, color=GRID, lw=1, zorder=0)
    ax.set_title("Walk-forward out-of-sample equity (vertical lines = re-optimisation)", loc="left")
    ax.set_ylabel("Portfolio value ($)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def save(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
