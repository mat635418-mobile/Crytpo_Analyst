import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import date, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Hedge Fund Terminal", layout="wide", page_icon="🏦")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 5px;
        padding: 15px;
        color: white;
    }
    .risk-alert {
        color: #ff4b4b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏦 Institutional Crypto Terminal")
st.markdown("Advanced Portfolio Optimization, Risk Management (VaR/CVaR), and Macro-Regime Filtering.")

# --- SIDEBAR & CONFIGURATION ---
st.sidebar.header("⚙️ Fund Settings")

# 1. UNIVERSE SELECTION
TOP_CRYPTO = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 
    'ADA-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD',
    'LINK-USD', 'UNI-USD', 'LTC-USD', 'ATOM-USD', 'XLM-USD'
]

# Allow selecting multiple assets for Portfolio Mode
selected_assets = st.sidebar.multiselect("Portfolio Universe", TOP_CRYPTO, default=['BTC-USD', 'ETH-USD', 'SOL-USD'])
benchmark_asset = 'BTC-USD' # For Beta calculation

# 2. TIMEFRAME
years_back = st.sidebar.slider("Lookback Period (Years)", 1, 5, 2)
start_date = date.today() - timedelta(days=years_back*365)
end_date = date.today()

# 3. MACRO DATA
MACRO_TICKERS = {
    'DXY (Dollar Index)': 'DX-Y.NYB',
    'US10Y (Treasury Yield)': '^TNX',
    'S&P 500': '^GSPC'
}

# --- DATA ENGINE ---
@st.cache_data
def load_data(tickers, macro_tickers, start, end):
    all_tickers = tickers + list(macro_tickers.values())
    try:
        data = yf.download(all_tickers, start=start, end=end, progress=False)['Close']
        
        # Handle yfinance MultiIndex issue
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel('Ticker')
            
        return data
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

with st.spinner("Fetching Institutional Data Feeds..."):
    # Ensure at least one asset is selected
    if not selected_assets:
        st.warning("Please select at least one asset in the sidebar.")
        st.stop()
        
    df_master = load_data(selected_assets, MACRO_TICKERS, start_date, end_date)
    
    # Calculate Log Returns for everyone
    log_returns = np.log(df_master / df_master.shift(1)).dropna()

# --- TAB STRUCTURE ---
tab_port, tab_backtest, tab_risk, tab_macro = st.tabs([
    "⚖️ Portfolio Optimizer", 
    "🛠 Strategy Backtester", 
    "⚠️ Risk Management (VaR)", 
    "🌍 Macro Regime"
])

# ==========================================
# TAB 1: PORTFOLIO OPTIMIZATION (Markowitz)
# ==========================================
with tab_port:
    st.subheader("Mean-Variance Portfolio Optimization")
    
    if len(selected_assets) < 2:
        st.warning("Select at least 2 assets to optimize a portfolio.")
    else:
        col_p1, col_p2 = st.columns([1, 2])
        
        # 1. Correlation Matrix
        with col_p1:
            st.markdown("**Correlation Matrix**")
            corr_matrix = log_returns[selected_assets].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)
            
        # 2. Efficient Frontier & Optimization
        with col_p2:
            st.markdown("**Efficient Frontier Solver**")
            
            # Helper functions for optimization
            def portfolio_performance(weights, mean_returns, cov_matrix):
                returns = np.sum(mean_returns * weights) * 252
                std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                return returns, std

            def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
                p_ret, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
                return -(p_ret - risk_free_rate) / p_std

            # Data prep
            mu = log_returns[selected_assets].mean()
            sigma = log_returns[selected_assets].cov()
            num_assets = len(selected_assets)
            args = (mu, sigma)
            
            # Constraints: Weights sum to 1, 0 <= weight <= 1
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for asset in range(num_assets))
            init_guess = num_assets * [1. / num_assets,]
            
            # Run Optimization
            result = minimize(neg_sharpe_ratio, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
            
            opt_weights = result.x
            opt_ret, opt_vol = portfolio_performance(opt_weights, mu, sigma)
            opt_sharpe = (opt_ret - 0.02) / opt_vol
            
            # Display Results
            st.success(f"Optimal Sharpe Ratio: {opt_sharpe:.2f}")
            
            # Pie Chart of Weights
            fig_pie = px.pie(values=opt_weights, names=selected_assets, title="Optimal Allocation")
            st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# TAB 2: STRATEGY BACKTESTER
# ==========================================
with tab_backtest:
    st.subheader("Backtesting Engine (Vectorized)")
    
    bt_col1, bt_col2 = st.columns([1, 3])
    
    with bt_col1:
        bt_asset = st.selectbox("Select Asset to Test", selected_assets)
        sma_fast = st.number_input("Fast MA", value=20)
        sma_slow = st.number_input("Slow MA", value=50)
        initial_capital = st.number_input("Initial Capital ($)", value=10000)
        
    with bt_col2:
        # Prepare Data
        df_bt = pd.DataFrame(df_master[bt_asset]).dropna()
        df_bt.columns = ['Close']
        df_bt['Fast_MA'] = df_bt['Close'].rolling(window=sma_fast).mean()
        df_bt['Slow_MA'] = df_bt['Close'].rolling(window=sma_slow).mean()
        
        # Signal: 1 (Long) when Fast > Slow, else 0 (Cash)
        df_bt['Signal'] = np.where(df_bt['Fast_MA'] > df_bt['Slow_MA'], 1, 0)
        df_bt['Position'] = df_bt['Signal'].shift(1) # Enter on next day open
        
        # Calculate Returns
        df_bt['Market_Ret'] = df_bt['Close'].pct_change()
        df_bt['Strategy_Ret'] = df_bt['Market_Ret'] * df_bt['Position']
        
        # Equity Curve
        df_bt['Cumulative_Market'] = (1 + df_bt['Market_Ret']).cumprod() * initial_capital
        df_bt['Cumulative_Strategy'] = (1 + df_bt['Strategy_Ret']).cumprod() * initial_capital
        
        # Drawdown Calculation
        rolling_max = df_bt['Cumulative_Strategy'].cummax()
        drawdown = (df_bt['Cumulative_Strategy'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Metrics
        total_return = (df_bt['Cumulative_Strategy'].iloc[-1] / initial_capital) - 1
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Return", f"{total_return:.2%}")
        m2.metric("Max Drawdown", f"{max_drawdown:.2%}", delta_color="inverse")
        m3.metric("Final Equity", f"${df_bt['Cumulative_Strategy'].iloc[-1]:,.2f}")
        
        # Plot
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Cumulative_Market'], name='Buy & Hold', line=dict(color='gray', dash='dash')))
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Cumulative_Strategy'], name='Strategy (MA Cross)', line=dict(color='green')))
        st.plotly_chart(fig_bt, use_container_width=True)
        
        # Drawdown Plot
        fig_dd = px.area(x=df_bt.index, y=drawdown, title="Drawdown Underwater Plot")
        fig_dd.update_traces(line_color='red')
        st.plotly_chart(fig_dd, use_container_width=True)

# ==========================================
# TAB 3: RISK MANAGEMENT (VaR/CVaR)
# ==========================================
with tab_risk:
    st.subheader("⚠️ Institutional Risk Metrics")
    
    risk_asset = st.selectbox("Analyze Asset Risk", selected_assets, key='risk_select')
    
    # Calculate Metrics
    returns = log_returns[risk_asset]
    
    # VaR (95% and 99%)
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    # CVaR (Expected Shortfall) - Average of losses exceeding VaR
    cvar_95 = returns[returns <= var_95].mean()
    
    # Beta Calculation (vs BTC as benchmark)
    # Covariance(Asset, Benchmark) / Variance(Benchmark)
    if risk_asset != benchmark_asset:
        cov_matrix = np.cov(returns, log_returns[benchmark_asset])
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    else:
        beta = 1.0
        
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Daily VaR (95%)", f"{var_95:.2%}", help="Minimum loss expected 5% of the time")
    r2.metric("CVaR / Expected Shortfall", f"{cvar_95:.2%}", help="Average loss in worst 5% scenarios")
    r3.metric("Beta (vs BTC)", f"{beta:.2f}", help=">1 means more volatile than BTC")
    r4.metric("Annual Volatility", f"{returns.std() * np.sqrt(365):.2%}")
    
    # Visualizing Tail Risk
    fig_hist = px.histogram(returns, nbins=100, title=f"Return Distribution & Tail Risk: {risk_asset}")
    fig_hist.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="VaR 95%")
    st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# TAB 4: MACRO REGIME & AI FILTERS
# ==========================================
with tab_macro:
    st.subheader("🌍 Macro-Economic Regime")
    
    col_m1, col_m2 = st.columns([2, 1])
    
    with col_m1:
        # Plot DXY and Yields
        fig_macro = go.Figure()
        # Normalize data to compare trends
        dxy = df_master[MACRO_TICKERS['DXY (Dollar Index)']]
        yields = df_master[MACRO_TICKERS['US10Y (Treasury Yield)']]
        
        fig_macro.add_trace(go.Scatter(x=dxy.index, y=dxy, name='DXY (Dollar)', yaxis='y1'))
        fig_macro.add_trace(go.Scatter(x=yields.index, y=yields, name='US 10Y Yields', yaxis='y2', line=dict(dash='dot')))
        
        fig_macro.update_layout(
            title="Macro Headwinds: Dollar & Rates",
            yaxis=dict(title="DXY"),
            yaxis2=dict(title="Yields %", overlaying="y", side="right")
        )
        st.plotly_chart(fig_macro, use_container_width=True)
        
    with col_m2:
        st.write("### Regime Filter")
        
        # Simple Regime Logic
        dxy_sma_50 = dxy.rolling(50).mean().iloc[-1]
        dxy_current = dxy.iloc[-1]
        
        regime = "RISK ON ✅"
        if dxy_current > dxy_sma_50:
            regime = "RISK OFF 🛑"
            st.error(f"Macro Regime: {regime}")
            st.write("The Dollar is trending UP. Crypto assets are highly correlated to liquidity. Caution is advised.")
        else:
            st.success(f"Macro Regime: {regime}")
            st.write("The Dollar is weak. Favorable environment for Risk Assets.")

st.markdown("---")
st.caption("Institutional Terminal v2.0 | Not Financial Advice")