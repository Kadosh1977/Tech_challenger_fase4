# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv

# ==============================
# Configuração
# ==============================
st.set_page_config(page_title="Dashboard IBOVESPA (CatBoost)", layout="centered")

st.title("📊 Dashboard de Tendência — IBOVESPA")
st.caption("Análise técnica + modelo preditivo (CatBoost)")

CSV_FILE = "Dados Históricos - Ibovespa 20 anos.csv"
THRESHOLD = 0.49
TEST_SIZE = 30
LOG_FILE = "log_previsoes.csv"

# ==============================
# Sidebar — Dashboard Controls
# ==============================
st.sidebar.header("⚙️ Painel de Controle")

janela_grafico = st.sidebar.slider(
    "Janela de análise (pregões)",
    min_value=20,
    max_value=300,
    value=50,
    step=10
)

mostrar_targets = st.sidebar.checkbox(
    "Mostrar últimos targets reais",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.caption("Modelo: CatBoost • Dados: IBOVESPA")

# ==============================
# Artefatos
# ==============================
model = joblib.load("modelo_final_catboost.joblib")
scaler = joblib.load("scaler_dados_ibovespa.joblib")
features_saved = joblib.load("features_modelo_ibov.joblib")

# ==============================
# Funções auxiliares
# ==============================
def tratar_coluna_volume(coluna_volume: pd.Series) -> pd.Series:
    coluna_tratada = coluna_volume.astype(str).copy()
    mult = {"k": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    for suf, m in mult.items():
        mask = coluna_tratada.str.contains(suf, case=False, na=False)
        coluna_tratada.loc[mask] = (
            coluna_tratada.loc[mask]
            .str.replace(suf, "", case=False)
            .str.replace(",", ".")
            .astype(float) * m
        )
    return pd.to_numeric(coluna_tratada, errors="coerce")

def calculate_slope(series: pd.Series, window: int) -> pd.Series:
    slopes = [np.nan] * (window - 1)
    for i in range(window, len(series) + 1):
        y = series.iloc[i - window:i].values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)

def calcular_rsi(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    delta = df["close"].diff()
    ganho = delta.where(delta > 0, 0.0)
    perda = -delta.where(delta < 0, 0.0)
    media_ganho = ganho.rolling(periodo).mean()
    media_perda = perda.rolling(periodo).mean()
    rs = media_ganho / media_perda
    return 100 - (100 / (1 + rs))

def calcular_obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()

def calcular_close_position(df: pd.DataFrame) -> pd.Series:
    faixa = df["high"] - df["low"]
    pos = (df["close"] - df["low"]) / faixa
    pos.loc[faixa == 0] = 0.5
    return pos

def categorizar_periodo(dt):
    if dt.year <= 2009:
        return "crise_2005_2009"
    elif dt.year <= 2019:
        return "pre_pandemia_2010_2019"
    elif dt.year <= 2022:
        return "pandemia_2020_2022"
    return "recente_2023_atual"

# ==============================
# Carregar dados (pipeline original)
# ==============================
dados = pd.read_csv(CSV_FILE)
dados["Data"] = pd.to_datetime(dados["Data"], format="%d.%m.%Y", errors="coerce")
dados = dados.dropna(subset=["Data"]).set_index("Data").sort_index()

dados["Var%"] = dados["Var%"].astype(str).str.replace(",", ".").str.replace("%", "").astype(float)
dados["Vol."] = tratar_coluna_volume(dados["Vol."])

dados = dados.rename(columns={
    "Último": "close",
    "Abertura": "open",
    "Máxima": "high",
    "Mínima": "low",
    "Vol.": "volume",
    "Var%": "var_pct"
})

dados["target"] = (dados["var_pct"].shift(-1) > 0).astype(int)
dados["volume"] = np.log1p(dados["volume"])

mask = ~dados[["volume", "var_pct"]].isnull().any(axis=1)
dados.loc[mask, ["volume", "var_pct"]] = scaler.transform(dados.loc[mask, ["volume", "var_pct"]])

# ==============================
# Dados brutos para gráfico
# ==============================
dados_graf = pd.read_csv(CSV_FILE)
dados_graf["Data"] = pd.to_datetime(dados_graf["Data"], format="%d.%m.%Y", errors="coerce")
dados_graf["Var_pct"] = (
    dados_graf["Var%"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.replace(",", ".", regex=False)
    .astype(float)
)
dados_graf = dados_graf.dropna().sort_values("Data")

# ==============================
# Engenharia de features (inalterada)
# ==============================
dados["open_lag_1"] = dados["open"].shift(1)
dados["high_lag_1"] = dados["high"].shift(1)
dados["low_lag_1"] = dados["low"].shift(1)
dados["volume_lag_1"] = dados["volume"].shift(1)
dados["var_pct_lag_1"] = dados["var_pct"].shift(1)

for lag in [5, 10, 15, 20, 30]:
    dados[f"var_pct_lag_{lag}"] = dados["var_pct"].shift(lag)

dados["return_1w"] = dados["close"].pct_change(5)
dados["return_2m"] = dados["close"].pct_change(60)
dados["volume_pct_change"] = dados["volume"].pct_change()
dados["close_position"] = calcular_close_position(dados)
dados["daily_range"] = dados["high"] - dados["low"]
dados["slope_20d"] = calculate_slope(dados["close"], 20)
dados["slope_50d"] = calculate_slope(dados["close"], 50)
dados["rsi"] = calcular_rsi(dados)
dados["obv"] = calcular_obv(dados)

dados["periodo"] = dados.index.map(categorizar_periodo)
dados = dados.dropna()

# ==============================
# Construção X e y
# ==============================
X_full = dados.drop(columns=["close", "target", "high", "low"])
for col in features_saved:
    if col not in X_full:
        X_full[col] = 0

X = X_full[features_saved]
y = dados["target"].loc[X.index]

ultima_data = X.index.max()

# ==============================
# KPIs
# ==============================
col1, col2, col3 = st.columns(3)
col1.metric("📅 Último Pregão", ultima_data.date())
col2.metric("📈 Close", f"{dados.loc[ultima_data,'close']:.0f}")
col3.metric("📊 RSI", f"{dados.loc[ultima_data,'rsi']:.1f}")

# ==============================
# Gráfico de tendência
# ==============================
dados["MA_20"] = dados["close"].rolling(20).mean()
dados["MA_50"] = dados["close"].rolling(50).mean()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dados.index[-200:], y=dados["close"][-200:], name="Preço"))
fig2.add_trace(go.Scatter(x=dados.index[-200:], y=dados["MA_20"][-200:], name="MA 20"))
fig2.add_trace(go.Scatter(x=dados.index[-200:], y=dados["MA_50"][-200:], name="MA 50"))
fig2.update_layout(title="Tendência do IBOV — Últimos 200 dias")
st.plotly_chart(fig2, use_container_width=True)

# ==============================
# Gráfico variação diária
# ==============================
dados_graf_tail = dados_graf.tail(janela_grafico)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=dados_graf_tail["Data"],
    y=dados_graf_tail["Var_pct"],
    mode="lines+markers",
    name="Variação diária (%)"
))
fig3.update_layout(
    title="Variação diária do IBOV",
    yaxis_title="Variação (%)",
    template="plotly_dark"
)
st.plotly_chart(fig3, use_container_width=True)

# ==============================
# Targets reais
# ==============================
if mostrar_targets:
    st.subheader("📊 Últimos 5 targets reais")
    st.table(y.tail(5).to_frame("Target"))

# ==============================
# Predição
# ==============================
if st.button("📊 Realizar Predição"):
    X_test = X.iloc[-TEST_SIZE:]
    y_test = y.iloc[-TEST_SIZE:]
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= THRESHOLD).astype(int)

    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia", f"{accuracy_score(y_test, pred_test):.3f}")
    c2.metric("Precisão", f"{precision_score(y_test, pred_test):.3f}")
    c3.metric("Recall", f"{recall_score(y_test, pred_test):.3f}")

    st.success("Predição executada com sucesso 🚀")

