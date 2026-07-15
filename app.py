import pandas as pd
import streamlit as st
from backtester import Backtester
from strategies import RegimeFilteredMovingAverageCrossover, VolatilityTargetingStrategy
from utils import detect_regimes

st.set_page_config(page_title="QuantVol Backtester", layout="wide")

st.title("📈 QuantVol Systematic Backtester")
st.markdown("Upload your custom market data to run the multi-regime backtester.")

# Provide a template download and hint
template_csv = "Date,Open,High,Low,Close,Volume\n2020-01-01,100,105,95,102,10000\n2020-01-02,102,108,100,105,12000\n"
st.download_button(
    label="📄 Download Sample CSV Template",
    data=template_csv,
    file_name="quantvol_template.csv",
    mime="text/csv",
    help="Download this template to see the required column format."
)
st.caption("Your file must contain `Open`, `High`, `Low`, and `Close` columns. `Volume` and `Date` will be auto-detected.")

# Sidebar Config
st.sidebar.header("Execution Friction")
initial_capital = st.sidebar.number_input("Initial Capital ($)", value=100000.0)
commission_rate = st.sidebar.number_input("Commission Rate", value=0.0010, format="%.4f")
slippage_rate = st.sidebar.number_input("Slippage Rate", value=0.0005, format="%.4f")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.04, format="%.2f")

st.sidebar.header("Regime Detection Parameters")
vol_window = st.sidebar.slider("Vol Rolling Window", 10, 100, 20)
low_thresh = st.sidebar.slider("Low Vol Thresh", 0.05, 0.50, 0.14)
high_thresh = st.sidebar.slider("High Vol Thresh", 0.15, 1.00, 0.25)
trading_days = st.sidebar.number_input("Trading Days/Year", value=252)

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
        
    # Standardize column names (assuming first column is Date)
    if "Date" not in df.columns:
        # Try to infer date column, or just use index
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    # Capitalize standard columns
    rename_map = {col: col.capitalize() for col in df.columns}
    df.rename(columns=rename_map, inplace=True)
    
    required_cols = ["Open", "High", "Low", "Close"]
    valid = True
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            valid = False
            
    if valid:
        st.success("Data loaded successfully!")
        
        with st.spinner("Detecting Volatility Regimes..."):
            regimes = detect_regimes(df, vol_window=vol_window, low_thresh=low_thresh, 
                                     high_thresh=high_thresh, trading_days=trading_days)
            df["DetectedRegime"] = regimes
            
        st.subheader("Regime Breakdown (Days)")
        # 0: Low, 1: Med, 2: High
        regime_counts = regimes.value_counts().sort_index()
        regime_counts.index = regime_counts.index.map({0: "Low Vol", 1: "Medium Vol", 2: "High Vol"})
        st.bar_chart(regime_counts)
        
        # Strategies configuration
        st.sidebar.header("Strategy Configurations")
        fast_window = st.sidebar.slider("Crossover Fast MA", 5, 50, 20)
        slow_window = st.sidebar.slider("Crossover Slow MA", 20, 200, 50)
        target_vol = st.sidebar.slider("Volatility Target", 0.05, 0.50, 0.12)
        
        with st.spinner("Running Backtests..."):
            backtester = Backtester(initial_capital, commission_rate, slippage_rate, risk_free_rate)
            
            strat_crossover = RegimeFilteredMovingAverageCrossover(fast_window=fast_window, slow_window=slow_window)
            signals_crossover = strat_crossover.generate_signals(df, regimes)
            results_crossover = backtester.run(df, signals_crossover)
            
            strat_voltarget = VolatilityTargetingStrategy(target_vol=target_vol, trading_days=trading_days)
            signals_voltarget = strat_voltarget.generate_signals(df, regimes)
            results_voltarget = backtester.run(df, signals_voltarget)
            
        st.subheader("Performance Metrics")
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Buy & Hold")
            st.metric("Total Return", f"{results_crossover['benchmark_metrics']['Total Return']*100:.2f}%")
            st.metric("Sharpe Ratio", f"{results_crossover['benchmark_metrics']['Sharpe Ratio']:.2f}")
            st.metric("Max Drawdown", f"{results_crossover['benchmark_metrics']['Max Drawdown']*100:.2f}%")
            
        with col2:
            st.markdown("### Regime-Filtered Crossover")
            st.metric("Total Return", f"{results_crossover['metrics']['Total Return']*100:.2f}%")
            st.metric("Sharpe Ratio", f"{results_crossover['metrics']['Sharpe Ratio']:.2f}")
            st.metric("Max Drawdown", f"{results_crossover['metrics']['Max Drawdown']*100:.2f}%")
            
        with col3:
            st.markdown("### Volatility Targeting")
            st.metric("Total Return", f"{results_voltarget['metrics']['Total Return']*100:.2f}%")
            st.metric("Sharpe Ratio", f"{results_voltarget['metrics']['Sharpe Ratio']:.2f}")
            st.metric("Max Drawdown", f"{results_voltarget['metrics']['Max Drawdown']*100:.2f}%")
            
        st.subheader("Equity Curve ($)")
        
        # Combine curves for plotting
        curves_df = pd.DataFrame({
            "Buy & Hold": results_crossover["benchmark_curve"],
            "MA Crossover": results_crossover["equity_curve"],
            "Vol Targeting": results_voltarget["equity_curve"]
        })
        st.line_chart(curves_df)
        
        st.subheader("Raw Data Preview")
        st.dataframe(df.tail(10))
