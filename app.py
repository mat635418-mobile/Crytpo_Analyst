import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Crypto AI Dashboard", layout="wide", page_icon="🪙")

st.title("🪙 Crypto AI & Monte Carlo Dashboard")
st.markdown("""
Questa dashboard analizza le **Top Crypto**, esegue simulazioni **Monte Carlo** e utilizza l'**Intelligenza Artificiale** per suggerire trend di mercato.
""")

# --- SIDEBAR & PARAMETRI ---
st.sidebar.header("⚙️ Configurazione")

# Lista Top Crypto (Espandibile a 500 tramite API esterne, qui usiamo le principali per velocità YFinance)
TOP_CRYPTO = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 
    'ADA-USD', 'AVAX-USD', 'DOGE-USD', 'DOT-USD', 'MATIC-USD',
    'LTC-USD', 'SHIB-USD', 'TRX-USD', 'LINK-USD', 'ATOM-USD'
]

# Selezione Asset
selected_crypto = st.sidebar.selectbox("Seleziona Criptovaluta", TOP_CRYPTO)
input_ticker = st.sidebar.text_input("O inserisci ticker manuale (es. PEPE-USD)", value="")

if input_ticker:
    ticker = input_ticker.upper()
else:
    ticker = selected_crypto

# Periodo Analisi
start_date = st.sidebar.date_input("Data Inizio Analisi", date.today() - timedelta(days=365))
end_date = date.today()

# Parametri Monte Carlo
st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Parametri Monte Carlo")
simulations = st.sidebar.slider("Numero Simulazioni", 200, 1000, 500)
time_horizon = st.sidebar.slider("Giorni di Previsione (MC)", 30, 365, 60)

# Parametri AI
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Parametri AI")
ai_days_ahead = st.sidebar.slider("Giorni da Predire (AI)", 1, 30, 7)

# --- FUNZIONI DI CARICAMENTO DATI ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    data.reset_index(inplace=True)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error(f"Nessun dato trovato per {ticker}. Controlla il simbolo.")
    st.stop()

# Calcolo Metriche Base
current_price = data['Close'].iloc[-1]
prev_price = data['Close'].iloc[-2]
daily_return = (current_price - prev_price) / prev_price
data['Return'] = data['Close'].pct_change()
volatility = data['Return'].std() * np.sqrt(252) # Volatilità annualizzata

# --- HEADER METRICHE ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Prezzo Attuale", f"${current_price:.2f}", f"{daily_return:.2%}")
col2.metric("Volatilità (Annuale)", f"{volatility:.2%}")
col3.metric("Massimo Periodo", f"${data['High'].max():.2f}")
col4.metric("Minimo Periodo", f"${data['Low'].min():.2f}")

st.markdown("---")

# --- TAB SETUP ---
tab1, tab2, tab3 = st.tabs(["📊 Analisi Tecnica", "🎲 Analisi Monte Carlo", "🤖 Previsione AI"])

# --- TAB 1: ANALISI TECNICA ---
with tab1:
    st.subheader(f"Andamento Storico: {ticker}")
    
    # Grafico Candlestick
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data['Date'],
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'], name='market data'))
    
    fig.update_layout(title=f'{ticker} Prezzo', yaxis_title='Prezzo USD', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Grafico Volume
    st.bar_chart(data, x='Date', y='Volume')

# --- TAB 2: MONTE CARLO ---
with tab2:
    st.subheader(f"Simulazione Monte Carlo ({simulations} scenari)")
    st.markdown("La simulazione utilizza il moto browniano geometrico per proiettare possibili percorsi di prezzo futuri basati su volatilità storica e drift.")

    # Preparazione Monte Carlo
    # Formula: St = St-1 * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    last_price = data['Close'].iloc[-1]
    
    # Log returns per calcolo drift
    log_returns = np.log(1 + data['Close'].pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    
    daily_returns_sim = np.exp(drift + stdev * np.random.normal(0, 1, (time_horizon, simulations)))
    
    price_paths = np.zeros_like(daily_returns_sim)
    price_paths[0] = last_price
    
    for t in range(1, time_horizon):
        price_paths[t] = price_paths[t-1] * daily_returns_sim[t]
    
    # Visualizzazione
    fig_mc = go.Figure()
    
    # Aggiungi tracce (limitiamo a 50 per performance rendering se l'utente ne sceglie 1000)
    display_sims = min(simulations, 100)
    for i in range(display_sims):
        fig_mc.add_trace(go.Scatter(y=price_paths[:, i], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
    
    # Aggiungi media
    mean_path = np.mean(price_paths, axis=1)
    fig_mc.add_trace(go.Scatter(y=mean_path, mode='lines', name='Media Simulazioni', line=dict(color='red', width=3)))
    
    fig_mc.update_layout(title=f'Proiezione a {time_horizon} giorni', yaxis_title='Prezzo Proiettato')
    st.plotly_chart(fig_mc, use_container_width=True)

    # Analisi Risultati MC
    final_prices = price_paths[-1]
    expected_price = np.mean(final_prices)
    quantile_5 = np.percentile(final_prices, 5)
    quantile_95 = np.percentile(final_prices, 95)
    
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    mc_col1.metric("Prezzo Atteso (Media)", f"${expected_price:.2f}", delta=f"{(expected_price - last_price)/last_price:.2%}")
    mc_col2.metric("Scenario Pessimista (5%)", f"${quantile_5:.2f}")
    mc_col3.metric("Scenario Ottimista (95%)", f"${quantile_95:.2f}")

# --- TAB 3: AI PREDICTION ---
with tab3:
    st.subheader("🤖 Intelligenza Artificiale Predittiva")
    st.markdown(f"Addestramento modello ML (Random Forest) sui dati storici per predire il prezzo tra **{ai_days_ahead} giorni**.")

    # Feature Engineering Semplice
    df_ai = data[['Date', 'Close']].copy()
    df_ai['Days'] = (df_ai['Date'] - df_ai['Date'].min()).dt.days
    
    # Creiamo finestre temporali (Rolling features)
    df_ai['MA_7'] = df_ai['Close'].rolling(window=7).mean()
    df_ai['MA_30'] = df_ai['Close'].rolling(window=30).mean()
    df_ai['Lag_1'] = df_ai['Close'].shift(1)
    df_ai['Lag_7'] = df_ai['Close'].shift(7)
    
    # Target: Il prezzo tra 'ai_days_ahead' giorni
    df_ai['Target'] = df_ai['Close'].shift(-ai_days_ahead)
    
    # Rimuovi NaN
    df_ai.dropna(inplace=True)
    
    if len(df_ai) > 100:
        # Features e Target
        features = ['Days', 'MA_7', 'MA_30', 'Lag_1', 'Lag_7']
        X = df_ai[features]
        y = df_ai['Target']
        
        # Split Train/Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Modello
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        st.info(f"Accuratezza Modello (R² su Test Set): {score:.2f}")
        
        # Predizione Futura
        last_row = data.iloc[-1]
        # Ricostruiamo le feature per l'ultimo giorno disponibile per predire il futuro
        # Nota: Questo è un calcolo approssimativo per la demo
        last_days_val = (last_row['Date'] - data['Date'].min()).days + ai_days_ahead
        
        # Feature vector per la predizione
        # Dobbiamo calcolare le rolling averages attuali
        ma_7_curr = data['Close'].rolling(7).mean().iloc[-1]
        ma_30_curr = data['Close'].rolling(30).mean().iloc[-1]
        lag_1_curr = data['Close'].iloc[-2]
        lag_7_curr = data['Close'].iloc[-8]
        
        future_features = pd.DataFrame([[last_days_val, ma_7_curr, ma_30_curr, lag_1_curr, lag_7_curr]], 
                                       columns=features)
        
        predicted_price = model.predict(future_features)[0]
        
        # Logica di Suggerimento
        change_pct = (predicted_price - current_price) / current_price
        
        st.markdown("### Risultato Predizione")
        ai_c1, ai_c2 = st.columns(2)
        
        ai_c1.metric(f"Prezzo Previsto ({ai_days_ahead}gg)", f"${predicted_price:.2f}")
        ai_c1.metric("Variazione Prevista", f"{change_pct:.2%}")
        
        recommendation = "NEUTRAL 😐"
        color = "gray"
        if change_pct > 0.05: # > 5% rialzo
            recommendation = "STRONG BUY 🚀"
            color = "green"
        elif change_pct > 0.01:
            recommendation = "BUY 📈"
            color = "lightgreen"
        elif change_pct < -0.05:
            recommendation = "STRONG SELL 🔻"
            color = "red"
        elif change_pct < -0.01:
            recommendation = "SELL 📉"
            color = "orange"
            
        ai_c2.markdown(f"## Suggerimento AI:")
        ai_c2.markdown(f"<h1 style='color: {color};'>{recommendation}</h1>", unsafe_allow_html=True)
        
        # Grafico Real vs Predicted (sul test set)
        st.write("Confronto Reale vs Predetto (Test Set)")
        test_pred = model.predict(X_test)
        
        fig_ai = go.Figure()
        fig_ai.add_trace(go.Scatter(y=y_test.values, name='Reale', mode='markers'))
        fig_ai.add_trace(go.Scatter(y=test_pred, name='Predetto AI', mode='lines', line=dict(color='orange')))
        st.plotly_chart(fig_ai, use_container_width=True)
        
    else:
        st.warning("Dati insufficienti per addestrare il modello AI. Aumenta il periodo storico.")

st.markdown("---")
st.caption("Disclaimer: Questa applicazione è a scopo dimostrativo ed educativo. Non costituisce consulenza finanziaria. Le criptovalute sono asset ad alta volatilità.")
