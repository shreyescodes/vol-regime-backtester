"""End-to-end orchestration: data -> regimes -> strategies -> stress -> attribution -> report."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from quantvol import reporting
from quantvol.config import Config, DataConfig
from quantvol.data import generate_synthetic_data, load_csv_data
from quantvol.engine import Backtester, BacktestResult
from quantvol.metrics import regime_attribution
from quantvol.regimes import detect_regimes, transition_matrix
from quantvol.resources import resolve_data_path
from quantvol.strategies import (
    RegimeFilteredMovingAverageCrossover,
    VolatilityTargetingStrategy,
)
from quantvol.stress import run_stress_tests
from quantvol.walkforward import WalkForwardResult, parameter_sweep, walk_forward

logger = logging.getLogger(__name__)

BENCHMARK = "Buy & Hold"


@dataclass(frozen=True)
class PipelineResult:
    config: Config
    data: pd.DataFrame
    regimes: pd.Series
    results: dict[str, BacktestResult]
    metrics: pd.DataFrame
    stress: pd.DataFrame
    attribution: dict[str, pd.DataFrame]
    transitions: pd.DataFrame
    sweep: pd.DataFrame
    walk_forward: WalkForwardResult | None


def load_market_data(cfg: DataConfig, trading_days: int = 252) -> pd.DataFrame:
    """Resolve the configured data source into a standardized OHLCV frame."""
    if cfg.source == "csv":
        assert cfg.csv_path is not None
        return load_csv_data(resolve_data_path(cfg.csv_path))
    if cfg.source == "yahoo":
        from quantvol.data.yahoo import fetch_yahoo

        assert cfg.ticker is not None
        return fetch_yahoo(cfg.ticker, cfg.start_date, cfg.end_date)
    return generate_synthetic_data(
        n_days=cfg.n_days,
        seed=cfg.seed,
        start_date=cfg.start_date,
        trading_days=trading_days,
        base_volume=cfg.base_volume,
    )


def run_pipeline(cfg: Config, data: pd.DataFrame | None = None) -> PipelineResult:
    """Run every analysis described by ``cfg`` and return all artefacts in memory."""
    if data is None:
        data = load_market_data(cfg.data, cfg.backtest.trading_days)
    regimes = detect_regimes(data, **asdict(cfg.regime))
    engine = Backtester(**asdict(cfg.backtest))

    crossover = RegimeFilteredMovingAverageCrossover(**asdict(cfg.crossover))
    vol_target = VolatilityTargetingStrategy(**asdict(cfg.vol_target))
    results = {
        s.name: engine.run(data, s.generate_signals(data, regimes)) for s in (crossover, vol_target)
    }
    first = next(iter(results.values()))
    metrics = pd.DataFrame(
        {BENCHMARK: first.benchmark_metrics, **{n: r.metrics for n, r in results.items()}}
    ).T

    logger.info("Running stress scenarios")
    stress = run_stress_tests(
        engine,
        data,
        lambda: RegimeFilteredMovingAverageCrossover(**asdict(cfg.crossover)),
        cfg.regime,
        cfg.stress,
    )

    # Group each bar's return by the regime known when the position was held (prior close).
    held_regime = regimes.shift(1).fillna(1).astype(int)
    ann = (cfg.backtest.trading_days, cfg.backtest.risk_free_rate)
    attribution = {
        BENCHMARK: regime_attribution(
            first.benchmark_returns, pd.Series(1.0, index=data.index), held_regime, *ann
        )
    }
    for name, res in results.items():
        attribution[name] = regime_attribution(res.returns, res.positions, held_regime, *ann)

    wf_cfg = cfg.walk_forward
    sweep = parameter_sweep(
        engine, data, regimes, wf_cfg.fast_grid, wf_cfg.slow_grid, cfg.crossover.flat_regimes
    )
    wf: WalkForwardResult | None = None
    if len(data) >= wf_cfg.train_days + wf_cfg.test_days:
        logger.info("Running walk-forward validation")
        wf = walk_forward(
            engine,
            data,
            regimes,
            wf_cfg.train_days,
            wf_cfg.test_days,
            wf_cfg.fast_grid,
            wf_cfg.slow_grid,
            cfg.crossover.flat_regimes,
        )
    return PipelineResult(
        cfg, data, regimes, results, metrics, stress, attribution,
        transition_matrix(regimes), sweep, wf,
    )  # fmt: skip


def write_report(result: PipelineResult, out_dir: str | Path) -> list[Path]:
    """Persist tables (CSV/JSON) and figures (PNG) under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    tables = {
        "metrics.csv": result.metrics,
        "stress.csv": result.stress,
        "regime_transitions.csv": result.transitions,
        "parameter_sweep.csv": result.sweep,
    }
    for name, frame in tables.items():
        frame.to_csv(out / name)
        written.append(out / name)
    for name, frame in result.attribution.items():
        path = out / f"regime_attribution_{name.lower().replace(' ', '_').replace('&', 'and')}.csv"
        frame.to_csv(path)
        written.append(path)

    summary = {
        "metrics": result.metrics.to_dict(orient="index"),
        "stress": result.stress.to_dict(orient="index"),
        "walk_forward_oos": result.walk_forward.oos_metrics if result.walk_forward else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(out / "summary.json")

    written.append(
        reporting.save(
            reporting.plot_performance(result.regimes, result.results),
            out / "performance.png",
        )
    )
    written.append(reporting.save(reporting.plot_stress(result.stress), out / "stress.png"))
    written.append(reporting.save(reporting.plot_sweep(result.sweep), out / "parameter_sweep.png"))
    written.append(
        reporting.save(
            reporting.plot_regime_attribution(result.attribution), out / "regime_attribution.png"
        )
    )
    if result.walk_forward is not None:
        first = next(iter(result.results.values()))
        written.append(
            reporting.save(
                reporting.plot_walkforward(result.walk_forward, first.benchmark_returns),
                out / "walk_forward.png",
            )
        )
        result.walk_forward.folds.to_csv(out / "walk_forward_folds.csv", index=False)
        written.append(out / "walk_forward_folds.csv")
    return written
