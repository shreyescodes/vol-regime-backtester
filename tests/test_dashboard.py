from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import quantvol

DASHBOARD = Path(quantvol.__file__).parent / "dashboard.py"


def test_dashboard_ships_inside_the_package() -> None:
    assert DASHBOARD.is_file()


def test_dashboard_renders_without_errors_on_bundled_data() -> None:
    app = AppTest.from_file(str(DASHBOARD), default_timeout=120).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "Overview",
        "Equity & drawdown",
        "Regimes",
        "Stress tests",
        "Robustness",
    ]
    assert len(app.metric) == 12  # 4 KPIs x 3 strategies


def test_dashboard_runs_on_synthetic_data() -> None:
    app = AppTest.from_file(str(DASHBOARD), default_timeout=120).run()
    app.radio[0].set_value("Synthetic regime-switching").run()
    assert not app.exception
