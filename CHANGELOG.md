# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0]

### Added
- Installable `quantvol` package (`src/` layout) with a `quantvol` CLI: `run`, `sweep`,
  `walkforward`, `dashboard`.
- Validated, frozen YAML configuration with clear error messages (`configs/spy.yaml`,
  `configs/default.yaml`).
- Real-data support: bundled SPY 2005-2024 sample, standardizing CSV/Excel loader, and an
  optional Yahoo Finance loader with an on-disk cache.
- Metrics: Sortino, annual volatility, historical VaR/CVaR, win rate, exposure, turnover.
- Per-regime performance attribution and an empirical regime transition matrix.
- Parameter sweep and rolling walk-forward validation.
- Reporting: equity/drawdown/exposure with regime shading, stress, sweep, attribution and
  walk-forward figures.
- Streamlit dashboard rebuilt on the package API with Overview, Equity, Regimes, Stress tests and
  Robustness tabs.
- Test suite of 59 pytest and Hypothesis tests (95% coverage), including look-ahead and
  walk-forward leakage checks.
- CI on Python 3.10-3.12 with ruff, mypy (strict), coverage, wheel build and Docker build;
  multi-stage non-root Dockerfile; pre-commit hooks.
- `docs/METHODOLOGY.md` and `docs/ARCHITECTURE.md`.

### Changed
- `Backtester` and all config sections are immutable dataclasses; results are returned as a typed
  `BacktestResult` instead of a dict.
- Stress scenarios no longer mutate engine state during the cost-surge test, and the flash-crash
  shock is vectorized.
- The buy-and-hold benchmark now pays the entry cost in its metrics as well as its equity curve.
- Synthetic data uses a local `numpy.random.Generator` instead of seeding global state.
- Regime detection is fully vectorized and rejects inconsistent thresholds.
- Dependencies are declared once in `pyproject.toml` extras (`app`, `data`, `docs`, `dev`);
  minimum Python is now 3.10.

### Removed
- Root-level scripts (`run_backtest.py`, `utils.py`, `backtester.py`, `strategies.py`, `app.py`),
  `requirements*.txt` and `PROJECT_SUMMARY.md`.
