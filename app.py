import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import date, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Pro Crypto Analyst", layout="wide", page_icon="📈")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Pro Crypto Analyst & AI Forecaster")
st.markdown("Advanced dashboard for Crypto analysis, Monte Carlo simulation, and AI-driven trend prediction.")

# --- SIDEBAR & CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")

# Extended List (Top 20 + ability to add any)
TOP_CRYPTO = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 
    'ADA-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD',
    'LTC-USD', 'SHIB-USD', 'TRX-USD', 'LINK-USD', 'ATOM-USD',
    'UNI-USD', 'XLM-USD', 'ETC-USD', 'FIL-USD', 'HBAR-USD'
]

ticker_source = st.sidebar.radio("Ticker Source", ["Select from Top List", "Search/Type Custom"])

if ticker_source == "Select from Top List":
    ticker = st.sidebar.selectbox("Select Asset", TOP_CRYPTO)
else:
    ticker = st.sidebar.text_input("Enter Ticker (e.g., PEPE-USD, AAPL)", value="BTC-USD").upper()

# Date Range
st.sidebar.subheader("📅 Timeframe")
years_back = st.sidebar.slider("Years of History", 1, 5, 2)
start_date = date.today() - timedelta(days=years_back*365)
end_date = date.today()

# --- HELPER FUNCTIONS ---

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

@st.cache_data
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        
        # Handle MultiIndex columns (yfinance update fix)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel('Ticker')
            except:
                df.columns = df.columns.get_level_values(-1)
                
        df.reset_index(inplace=True)
        
        # Ensure we have data
        if df.empty:
            return None
            
        # Add Technical Indicators for AI & Charting
        df['RSI'] = calculate_rsi(df)
        df['MACD'], df['Signal'] = calculate_macd(df)
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
        
        # Drop NaN created by indicators
        df.dropna(inplace=True)
        
        return df
    except Exception as e:
        return None

# --- LOAD DATA ---
with st.spinner(f"Loading data for {ticker}..."):
    data = load_data(ticker, start_date, end_date)

if data is None or data.empty:
    st.error(f"❌ Could not load data for {ticker}. Please check the symbol.")
    st.stop()

# --- MAIN DASHBOARD METRICS ---
current_price = float(data['Close'].iloc[-1])
prev_price = float(data['Close'].iloc[-2])
daily_return = (current_price - prev_price) / prev_price

# Risk Metrics Calculation
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
annual_volatility = data['Log_Return'].std() * np.sqrt(365)
sharpe_ratio = (data['Log_Return'].mean() * 365) / (data['Log_Return'].std() * np.sqrt(365)) if data['Log_Return'].std() != 0 else 0

# Max Drawdown
cumulative_returns = (1 + data['Log_Return']).cumprod()
peak = cumulative_returns.expanding(min_periods=1).max()
drawdown = (cumulative_returns / peak) - 1
max_drawdown = drawdown.min()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price", f"${current_price:,.2f}", f"{daily_return:.2%}")
c2.metric("Volatility (Yearly)", f"{annual_volatility:.2%}")
c3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
c4.metric("Max Drawdown", f"{max_drawdown:.2%}")
c5.metric("RSI (14)", f"{data['RSI'].iloc[-1]:.2f}")

st.markdown("---")

# --- TABS LOGIC ---
tab_charts, tab_mc, tab_ai = st.tabs(["📊 Technical Charts", "🎲 Monte Carlo Simulation", "🤖 AI Prediction"])

# ==========================================
# TAB 1: TECHNICAL ANALYSIS
# ==========================================
with tab_charts:
    st.subheader(f"Technical Analysis: {ticker}")
    
    # Chart Options
    chart_opts = st.multiselect("Overlay Indicators", ["SMA 50", "SMA 200", "Bollinger Bands"], default=["SMA 50"])
    
    # Main Candle Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'], name='Price'))
    
    if "SMA 50" in chart_opts:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
    if "SMA 200" in chart_opts:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], line=dict(color='blue', width=1), name='SMA 200'))
    if "Bollinger Bands" in chart_opts:
        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Upper'], line=dict(color='gray', width=0), showlegend=False, name='BB Upper'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['BB_Lower'], line=dict(color='gray', width=0), fill='tonexty', fillcolor='rgba(200,200,200,0.2)', name='BB Range'))

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, title=f"{ticker} Price Action")
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume & MACD
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("**Volume Profile**")
        st.bar_chart(data, x='Date', y='Volume', height=300)
    with col_v2:
        st.markdown("**MACD Oscillator**")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name='MACD'))
        fig_macd.add_trace(go.Scatter(x=data['Date'], y=data['Signal'], name='Signal'))
        fig_macd.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig_macd, use_container_width=True)

# ==========================================
# TAB 2: MONTE CARLO
# ==========================================
with tab_mc:
    st.subheader("🎲 Monte Carlo Stochastic Simulation")
    
    mc_col_param, mc_col_vis = st.columns([1, 3])
    
    with mc_col_param:
        st.info("Configuration")
        mc_sims = st.slider("Simulations", 200, 2000, 500)
        mc_days = st.slider("Forecast Horizon (Days)", 30, 365, 60)
        st.markdown("""
        **Methodology:**
        Uses Geometric Brownian Motion (GBM).
        $dS_t = \mu S_t dt + \sigma S_t dW_t$
        """)
    
    with mc_col_vis:
        # MC Logic
        log_returns = np.log(1 + data['Close'].pct_change()).dropna()
        u = log_returns.mean()
        var = log_returns.var()
        drift = u - (0.5 * var)
        stdev = log_returns.std()
        
        daily_returns_sim = np.exp(drift + stdev * np.random.normal(0, 1, (mc_days, mc_sims)))
        
        price_paths = np.zeros_like(daily_returns_sim)
        price_paths[0] = data['Close'].iloc[-1]
        
        for t in range(1, mc_days):
            price_paths[t] = price_paths[t-1] * daily_returns_sim[t]
            
        fig_mc = go.Figure()
        # Plot only first 100 paths for performance
        for i in range(min(mc_sims, 100)):
            fig_mc.add_trace(go.Scatter(y=price_paths[:, i], mode='lines', opacity=0.1, line=dict(color='cyan', width=1), showlegend=False))
        
        mean_path = np.mean(price_paths, axis=1)
        fig_mc.add_trace(go.Scatter(y=mean_path, mode='lines', name='Average Path', line=dict(color='red', width=3)))
        
        fig_mc.update_layout(title=f"Projected Paths for next {mc_days} days", yaxis_title="Price ($)")
        st.plotly_chart(fig_mc, use_container_width=True)

    # Statistics of Simulation
    final_prices = price_paths[-1]
    q5 = np.percentile(final_prices, 5)
    q50 = np.percentile(final_prices, 50)
    q95 = np.percentile(final_prices, 95)
    
    st.markdown("### Simulation Outcomes")
    m1, m2, m3 = st.columns(3)
    m1.metric("Bear Case (Bot 5%)", f"${q5:.2f}")
    m2.metric("Base Case (Median)", f"${q50:.2f}")
    m3.metric("Bull Case (Top 95%)", f"${q95:.2f}")
    
    # Histogram
    fig_hist = px.histogram(final_prices, nbins=50, title="Distribution of Predicted Prices")
    fig_hist.add_vline(x=current_price, line_dash="dash", line_color="green", annotation_text="Current Price")
    st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================
# TAB 3: AI & MACHINE LEARNING
# ==========================================
with tab_ai:
    st.subheader("🤖 AI Trend Prediction (Random Forest)")
    
    ai_col1, ai_col2 = st.columns([1, 2])
    
    with ai_col1:
        st.write("### Model Settings")
        prediction_days = st.slider("Predict Price in (Days)", 1, 30, 7)
        st.info("The model trains on historical Price, Volume, RSI, MACD, and Moving Averages to predict future movement.")
        
    with ai_col2:
        # Prepare Data for ML
        df_ml = data.copy()
        
        # Target: Future Price
        df_ml['Target'] = df_ml['Close'].shift(-prediction_days)
        
        # Features
        feature_cols = ['Close', 'Volume', 'RSI', 'MACD', 'SMA_50', 'SMA_200', 'BB_Upper', 'BB_Lower']
        df_ml.dropna(inplace=True) # Drop NaNs
        
        X = df_ml[feature_cols]
        y = df_ml['Target']
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = model.score(X_test, y_test)
        
        # Predict Future
        # Get latest data row for features
        latest_features = data[feature_cols].iloc[[-1]] 
        future_pred = model.predict(latest_features)[0]
        
        pct_change = (future_pred - current_price) / current_price
        
        # Display Result
        st.markdown(f"### Prediction for {prediction_days} days from now")
        
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Predicted Price", f"${future_pred:.2f}")
        
        color = "grey"
        signal = "NEUTRAL"
        if pct_change > 0.02: 
            color = "green"
            signal = "BUY"
        elif pct_change < -0.02: 
            color = "red"
            signal = "SELL"
            
        res_col1.markdown(f"**Potential Upside/Downside:** <span style='color:{color}'>{pct_change:.2%}</span>", unsafe_allow_html=True)
        res_col2.markdown(f"<h1 style='color:{color}; text-align: center;'>{signal}</h1>", unsafe_allow_html=True)
        
        st.write(f"Model Error (MAE): ${mae:.2f} | R² Score: {r2:.2f}")

    st.markdown("---")
    
    # Feature Importance Visualization
    st.subheader("🧠 What is driving the AI?")
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    fig_imp = px.bar(feature_df, x='Importance', y='Feature', orientation='h', title="Feature Importance Analysis")
    st.plotly_chart(fig_imp, use_container_width=True)

    # Backtest Chart
    st.subheader("Backtest: Real vs Predicted (Test Set)")
    fig_backtest = go.Figure()
    fig_backtest.add_trace(go.Scatter(y=y_test.values, name='Actual Price', line=dict(color='blue')))
    fig_backtest.add_trace(go.Scatter(y=preds, name='AI Predicted', line=dict(color='orange', dash='dash')))
    st.plotly_chart(fig_backtest, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes only. Cryptocurrency trading involves high risk. This is not financial advice.")
