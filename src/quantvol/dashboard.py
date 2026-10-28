"""QuantVol dashboard: interactive regime-aware backtesting on top of the quantvol package."""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import pandas as pd
import streamlit as st

from quantvol import reporting
from quantvol.config import (
    BacktestConfig,
    Config,
    ConfigError,
    CrossoverConfig,
    DataConfig,
    RegimeConfig,
    StressConfig,
    VolTargetConfig,
    WalkForwardConfig,
)
from quantvol.data import DataError, load_csv_data
from quantvol.pipeline import BENCHMARK, PipelineResult, run_pipeline
from quantvol.resources import resource_path

SAMPLE = resource_path("spy_2005_2024.csv")
TEMPLATE = "Date,Open,High,Low,Close,Volume\n2020-01-02,100,101,99,100.5,1000000\n"

st.set_page_config(page_title="QuantVol", page_icon=":chart_with_upwards_trend:", layout="wide")


@st.cache_data(show_spinner=False)
def cached_run(cfg: Config, data: pd.DataFrame) -> PipelineResult:
    return run_pipeline(cfg, data)


@dataclass(frozen=True)
class Defaults:
    low: float
    high: float
    commission: float
    slippage: float
    slow: int


EQUITY_DEFAULTS = Defaults(low=0.10, high=0.20, commission=0.0002, slippage=0.0003, slow=100)
SYNTHETIC_DEFAULTS = Defaults(low=0.14, high=0.25, commission=0.0010, slippage=0.0005, slow=50)


def sidebar() -> tuple[Config, pd.DataFrame | None]:
    st.sidebar.title("QuantVol")
    source = (
        st.sidebar.radio(
            "Market data", ["SPY 2005-2024 (bundled)", "Synthetic regime-switching", "Upload CSV"]
        )
        or "SPY"
    )
    data: pd.DataFrame | None = None
    data_cfg = DataConfig()
    defaults = EQUITY_DEFAULTS
    if source.startswith("SPY"):
        data = load_csv_data(SAMPLE)
    elif source.startswith("Synthetic"):
        n_days = st.sidebar.slider("Trading days", 500, 4000, 1500, step=100)
        seed = st.sidebar.number_input("Random seed", value=101, step=1)
        data_cfg = DataConfig(n_days=n_days, seed=int(seed))
        defaults = SYNTHETIC_DEFAULTS
    else:
        st.sidebar.download_button("CSV template", TEMPLATE, "quantvol_template.csv", "text/csv")
        upload = st.sidebar.file_uploader("Daily OHLC(V) file", type=["csv", "xlsx"])
        if upload is not None:
            try:
                raw = pd.read_excel(upload) if upload.name.endswith("xlsx") else pd.read_csv(upload)
                from quantvol.data import standardize_ohlcv

                data = standardize_ohlcv(raw, date_col=None)
            except (DataError, ValueError) as exc:
                st.sidebar.error(str(exc))

    with st.sidebar.expander("Execution costs", expanded=False):
        capital = st.number_input("Initial capital ($)", value=100_000.0, step=10_000.0)
        commission = st.number_input(
            "Commission (per unit turnover)", value=defaults.commission, format="%.4f"
        )
        slippage = st.number_input(
            "Slippage (per unit turnover)", value=defaults.slippage, format="%.4f"
        )
        rf = st.number_input("Risk-free rate", value=0.02, format="%.3f")
        days = st.number_input("Trading days / year", value=252, step=1)
    with st.sidebar.expander("Regime detection", expanded=False):
        window = st.slider("Realized-vol window", 5, 100, 20)
        low = st.slider("Low / medium threshold", 0.03, 0.40, defaults.low, 0.01)
        high = st.slider("Medium / high threshold", 0.08, 0.80, defaults.high, 0.01)
    with st.sidebar.expander("Strategies", expanded=True):
        fast = st.slider("Crossover fast MA", 5, 60, 20)
        slow = st.slider("Crossover slow MA", 30, 250, defaults.slow)
        target = st.slider("Volatility target (annual)", 0.05, 0.40, 0.10, 0.01)
    with st.sidebar.expander("Stress scenarios", expanded=False):
        spike = st.slider("Vol-spike multiplier", 1.1, 3.0, 1.5, 0.1)
        crash = st.slider("Flash-crash drop", -0.40, -0.02, -0.10, 0.01)
        mult = st.slider("Cost multiplier", 1.0, 10.0, 5.0, 0.5)

    try:
        cfg = Config(
            data=data_cfg,
            regime=RegimeConfig(window, low, high, int(days)),
            backtest=BacktestConfig(capital, commission, slippage, rf, 1, int(days)),
            crossover=CrossoverConfig(fast, max(slow, fast + 1)),
            vol_target=VolTargetConfig(slow, window, target, int(days)),
            stress=StressConfig(spike, crash, None, mult),
            walk_forward=WalkForwardConfig(756, 252, (10, 20, 30, 50), (60, 100, 150, 200))
            if source.startswith("SPY")
            else WalkForwardConfig(),
        )
    except ConfigError as exc:
        st.sidebar.error(str(exc))
        st.stop()
    if data is None and source.startswith("Synthetic"):
        from quantvol.pipeline import load_market_data

        data = load_market_data(cfg.data, cfg.backtest.trading_days)
    return cfg, data


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def kpis(res: PipelineResult) -> None:
    cols = st.columns(len(res.metrics))
    for col, (name, row) in zip(cols, res.metrics.iterrows(), strict=True):
        with col:
            st.markdown(f"**{name}**")
            st.metric("CAGR", pct(row["CAGR"]))
            st.metric("Sharpe", f"{row['Sharpe Ratio']:.2f}")
            st.metric("Max drawdown", pct(row["Max Drawdown"]))
            st.metric("Calmar", f"{row['Calmar Ratio']:.2f}")


def main() -> None:
    st.title("QuantVol: regime-aware systematic backtesting")
    st.caption(
        "Signals are lagged one bar (no look-ahead), costs are charged on every unit of turnover, "
        "and stress scenarios regenerate signals on the shocked path."
    )
    cfg, data = sidebar()
    if data is None:
        st.info("Upload a daily OHLC file in the sidebar to begin.")
        return
    if len(data) < max(cfg.crossover.slow_window, cfg.regime.vol_window) + 30:
        st.warning("Not enough history for the selected windows.")
        return

    with st.spinner("Running backtests, stress tests and walk-forward validation..."):
        res = cached_run(cfg, data)

    overview, equity, regimes, stress, robustness = st.tabs(
        ["Overview", "Equity & drawdown", "Regimes", "Stress tests", "Robustness"]
    )
    with overview:
        kpis(res)
        st.subheader("Full metrics")
        st.dataframe(res.metrics.style.format("{:.3f}"), use_container_width=True)
    with equity:
        st.pyplot(reporting.plot_performance(res.regimes, res.results))
    with regimes:
        counts = res.regimes.value_counts().reindex([0, 1, 2], fill_value=0)
        left, right = st.columns(2)
        left.subheader("Days in each regime")
        order = ["Low vol", "Medium vol", "High vol"]
        frame = pd.DataFrame({"Regime": order, "Days": counts.to_numpy()})
        left.altair_chart(
            alt.Chart(frame)
            .mark_bar(color="#2a78d6", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(x=alt.X("Regime", sort=order, axis=alt.Axis(labelAngle=0)), y="Days"),
            use_container_width=True,
        )
        right.subheader("Regime transition probabilities")
        right.dataframe(res.transitions.style.format("{:.1%}"), use_container_width=True)
        st.subheader("Performance by regime")
        st.pyplot(reporting.plot_regime_attribution(res.attribution))
        for name in (BENCHMARK, *res.results):
            with st.expander(name):
                st.dataframe(res.attribution[name].style.format("{:.3f}"))
    with stress:
        st.pyplot(reporting.plot_stress(res.stress))
        st.dataframe(res.stress.style.format("{:.3f}"), use_container_width=True)
    with robustness:
        st.subheader("Parameter sweep")
        st.pyplot(reporting.plot_sweep(res.sweep))
        st.subheader("Walk-forward validation")
        if res.walk_forward is None:
            st.info("Not enough history for the configured train/test windows.")
        else:
            first = next(iter(res.results.values()))
            st.pyplot(reporting.plot_walkforward(res.walk_forward, first.benchmark_returns))
            folds = res.walk_forward.folds.copy()
            for col in ("Train Start", "Test Start", "Test End"):
                folds[col] = folds[col].dt.strftime("%Y-%m-%d")
            st.dataframe(folds.round(3), use_container_width=True)


main()
