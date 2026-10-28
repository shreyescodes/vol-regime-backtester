"""Command-line interface: ``quantvol run|sweep|walkforward|dashboard``."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from quantvol import __version__
from quantvol.config import Config, ConfigError, load_config
from quantvol.data import DataError
from quantvol.pipeline import PipelineResult, run_pipeline, write_report

PERCENT_COLS = {"Total Return", "CAGR", "Max Drawdown", "Annual Volatility", "VaR 95%", "CVaR 95%"}
PERCENT_COLS |= {"Win Rate", "Exposure"}


def format_table(frame: pd.DataFrame) -> str:
    """Human-readable metrics table (percentages for return-like columns)."""
    shown = frame.copy().astype(float)
    for col in shown.columns:
        if col in PERCENT_COLS:
            shown[col] = shown[col].map(lambda v: f"{v * 100:.2f}%")
        else:
            shown[col] = shown[col].map(lambda v: f"{v:.3f}")
    return shown.to_string()


def _banner(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}"


def _print_run(res: PipelineResult) -> None:
    cols = ["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Calmar Ratio"]
    print(_banner("Performance"))
    print(format_table(res.metrics[cols]))
    print(_banner("Stress scenarios (crossover strategy)"))
    print(format_table(res.stress[["Total Return", "Sharpe Ratio", "Max Drawdown"]]))
    for name, table in res.attribution.items():
        print(_banner(f"Regime attribution - {name}"))
        print(table.round(3).to_string())
    if res.walk_forward is not None:
        print(_banner("Walk-forward (out-of-sample)"))
        oos = pd.DataFrame({"Walk-forward OOS": res.walk_forward.oos_metrics}).T
        print(format_table(oos[["Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown"]]))


def _load(args: argparse.Namespace) -> Config:
    return load_config(args.config) if args.config else Config()


def _cmd_run(args: argparse.Namespace) -> int:
    res = run_pipeline(_load(args))
    _print_run(res)
    files = write_report(res, args.out)
    print(_banner(f"Wrote {len(files)} files to {Path(args.out).resolve()}"))
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    res = run_pipeline(_load(args))
    print(_banner("Sharpe ratio by (fast, slow) crossover window"))
    print(res.sweep.round(2).to_string(na_rep="-"))
    return 0


def _cmd_walkforward(args: argparse.Namespace) -> int:
    res = run_pipeline(_load(args))
    if res.walk_forward is None:
        print("Not enough history for the configured train/test windows.", file=sys.stderr)
        return 1
    print(_banner("Walk-forward folds"))
    folds = res.walk_forward.folds.copy()
    for col in ("Train Start", "Test Start", "Test End"):
        folds[col] = folds[col].dt.strftime("%Y-%m-%d")
    print(folds.round(3).to_string(index=False))
    return 0


def _cmd_dashboard(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve().parents[2] / "app" / "dashboard.py"
    if not script.exists():
        print("dashboard.py not found; run from a source checkout.", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(script)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantvol", description=__doc__)
    parser.add_argument("--version", action="version", version=f"quantvol {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("-c", "--config", help="YAML config file (defaults are used if omitted)")

    run = sub.add_parser("run", help="full backtest, stress tests, attribution and report")
    common(run)
    run.add_argument("-o", "--out", default="reports", help="output directory (default: reports)")
    run.set_defaults(func=_cmd_run)

    sweep = sub.add_parser("sweep", help="Sharpe grid over crossover windows")
    common(sweep)
    sweep.set_defaults(func=_cmd_sweep)

    wf = sub.add_parser("walkforward", help="rolling walk-forward validation")
    common(wf)
    wf.set_defaults(func=_cmd_walkforward)

    dash = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    dash.set_defaults(func=_cmd_dashboard)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, DataError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
