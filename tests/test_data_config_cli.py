import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantvol.cli import main
from quantvol.config import Config, ConfigError, config_from_dict, load_config
from quantvol.data import DataError, generate_synthetic_data, load_csv_data, standardize_ohlcv

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_is_deterministic_and_valid() -> None:
    a = generate_synthetic_data(500, seed=3)
    b = generate_synthetic_data(500, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert (a["High"] >= a[["Open", "Close"]].max(axis=1)).all()
    assert (a["Low"] <= a[["Open", "Close"]].min(axis=1)).all()
    assert (a["Close"] > 0).all() and set(a["TrueRegime"]) <= {0, 1, 2}


def test_synthetic_does_not_touch_global_rng_state() -> None:
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    generate_synthetic_data(200, seed=1)
    assert np.random.random() == expected


def test_synthetic_rejects_bad_transition_matrix() -> None:
    with pytest.raises(ValueError):
        generate_synthetic_data(200, transition_matrix=np.array([[0.5, 0.2], [0.5, 0.5]]))


def test_csv_loader_handles_messy_input() -> None:
    raw = "date,OPEN,high,Low,close\n2020-01-03,3,3,3,3\n2020-01-01,1,1,1,1\n2020-01-01,1,1,1,1.5\n"
    df = load_csv_data(io.StringIO(raw), date_col="date")
    assert list(df.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-01-03"]
    assert df.loc["2020-01-01", "Close"] == 1.5  # last duplicate wins
    assert (df["Volume"] == 0).all()


def test_csv_loader_requires_close_and_positive_prices() -> None:
    with pytest.raises(DataError, match="Close"):
        standardize_ohlcv(pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"], "Px": [1, 2]}))
    with pytest.raises(DataError, match="positive"):
        standardize_ohlcv(
            pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"], "Close": [1.0, -2.0]})
        )


def test_bundled_sample_loads() -> None:
    df = load_csv_data(ROOT / "data" / "sample" / "spy_2005_2024.csv")
    assert len(df) > 5000 and df.index.is_monotonic_increasing


def test_default_and_shipped_configs_validate() -> None:
    assert isinstance(Config(), Config)
    for name in ("default.yaml", "spy.yaml"):
        assert isinstance(load_config(ROOT / "configs" / name), Config)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"bogus": {}}, "unknown top-level"),
        ({"regime": {"nope": 1}}, "unknown key"),
        ({"regime": {"low_thresh": 0.3, "high_thresh": 0.2}}, "low_thresh"),
        ({"backtest": {"execution_lag": 0}}, "look-ahead"),
        ({"backtest": {"commission_rate": -1}}, "non-negative"),
        ({"data": {"source": "csv"}}, "csv_path"),
        ({"data": {"source": "ftp"}}, "data.source"),
    ],
)
def test_invalid_configs_fail_loudly(raw: dict, message: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(ConfigError, match=message):
        config_from_dict(raw)


def test_cli_run_writes_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "data: {n_days: 700, seed: 5}\nwalk_forward: {train_days: 250, test_days: 100,"
        " fast_grid: [5, 10], slow_grid: [30, 60]}\n"
    )
    out = tmp_path / "rep"
    assert main(["run", "-c", str(cfg), "-o", str(out)]) == 0
    assert (out / "performance.png").stat().st_size > 10_000
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["metrics"]) == {
        "Buy & Hold",
        "Regime-Filtered Crossover",
        "Volatility Targeting",
    }
    assert "Performance" in capsys.readouterr().out


def test_cli_reports_config_errors_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("backtest: {execution_lag: 0}\n")
    assert main(["run", "-c", str(bad)]) == 2
    assert "look-ahead" in capsys.readouterr().err


def test_cli_sweep_and_walkforward(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "data: {n_days: 700}\nwalk_forward: {train_days: 250, test_days: 100,"
        " fast_grid: [5, 10], slow_grid: [30, 60]}\n"
    )
    assert main(["sweep", "-c", str(cfg)]) == 0
    assert main(["walkforward", "-c", str(cfg)]) == 0
    assert "OOS Sharpe" in capsys.readouterr().out
