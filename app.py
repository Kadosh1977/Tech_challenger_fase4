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
st.set_page_config(page_title="Previsão IBOVESPA (CatBoost)", layout="centered")
st.title("📈 Tendência IBOVESPA - CatBoost")

CSV_FILE = "Dados Históricos - Ibovespa 20 anos.csv"
THRESHOLD = 0.49
TEST_SIZE = 30
LOG_FILE = "log_previsoes.csv"

# ==============================
# Artefatos
# ==============================
model = joblib.load("modelo_final_catboost.joblib")
scaler = joblib.load("scaler_dados_ibovespa.joblib")
features_saved = joblib.load("features_modelo_ibov.joblib")

# ==============================
# Funções auxiliares
# ==============================
def tratar_coluna_volume(coluna):
    coluna = coluna.astype(str)
    mult = {"k": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    for s, m in mult.items():
        mask = coluna.str.contains(s, case=False, na=False)
        coluna.loc[mask] = (
            coluna.loc[mask]
            .str.replace(s, "", case=False)
            .str.replace(",", ".")
            .astype(float) * m
        )
    return pd.to_numeric(coluna, errors="coerce")

def calculate_slope(series, window):
    slopes = [np.nan] * (window - 1)
    for i in range(window, len(series) + 1):
        y = series.iloc[i - window:i].values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)

def calcular_rsi(df, periodo=14):
    delta = df["close"].diff()
    ganho = delta.clip(lower=0)
    perda = -delta.clip(upper=0)
    rs = ganho.rolling(periodo).mean() / perda.rolling(periodo).mean()
    return 100 - (100 / (1 + rs))

def calcular_obv(df):
    return (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()

def calcular_close_position(df):
    faixa = df["high"] - df["low"]
    pos = (df["close"] - df["low"]) / faixa
    pos[faixa == 0] = 0.5
    return pos

def categorizar_periodo(dt):
    if dt.year <= 2009:
        return "crise_2005_2009"
    elif dt.year <= 2019:
        return "pre_pandemia_2010_2019"
    elif dt.year <= 2022:
        return "pandemia_2020_2022"
    else:
        return "recente_2023_atual"

# ==============================
# Carregar e preparar dados
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
# Engenharia de features
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

dados = dados.dropna()

# ==============================
# X e y
# ==============================
X = dados.drop(columns=["close", "high", "low", "target"])
for col in features_saved:
    if col not in X.columns:
        X[col] = 0
X = X[features_saved]
y = dados["target"]

ultima_data = X.index.max()

# ==============================
# DASHBOARD — KPIs
# ==============================
st.subheader("📊 Resumo do Dataset")
k1, k2, k3 = st.columns(3)
k1.metric("📅 Último Pregão", str(ultima_data.date()))
k2.metric("📈 Total de Registros", len(dados))
k3.metric("🎯 Threshold", THRESHOLD)

# ==============================
# GRÁFICOS
# ==============================
st.subheader("📉 Variação Diária — Últimos 50 Pregões")

dados_graf = pd.read_csv(CSV_FILE)
dados_graf["Data"] = pd.to_datetime(dados_graf["Data"], format="%d.%m.%Y", errors="coerce")
dados_graf["Var_pct"] = (
    dados_graf["Var%"].astype(str)
    .str.replace("%", "")
    .str.replace(",", ".")
    .astype(float)
)
dados_graf = dados_graf.dropna().sort_values("Data").tail(50)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dados_graf["Data"],
    y=dados_graf["Var_pct"],
    mode="lines+markers",
    name="Variação (%)"
))
fig.update_layout(title="Variação do IBOV — Últimos 50 pregões")
st.plotly_chart(fig, use_container_width=True)

# ==============================
# BOTÃO DE PREDIÇÃO (COM RESULTADO VISÍVEL)
# ==============================
if st.button("📊 Realizar Predição"):

    X_test = X.iloc[-TEST_SIZE:]
    y_test = y.iloc[-TEST_SIZE:]

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test)

    st.subheader("📊 Métricas do Modelo")
    m1, m2, m3 = st.columns(3)
    m1.metric("Acurácia", f"{acc:.3f}")
    m2.metric("Precisão", f"{precision:.3f}")
    m3.metric("Recall", f"{recall:.3f}")

    st.subheader("🔍 Matriz de Confusão")
    fig_cm, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, pred_test), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig_cm)

    prob_next = model.predict_proba(X.iloc[[-1]])[0, 1]
    pred_next = int(prob_next >= THRESHOLD)

    st.subheader("🔮 Previsão Próximo Pregão")
    if pred_next == 1:
        st.success(f"Alta prevista — Probabilidade: {prob_next*100:.2f}% 📈")
    else:
        st.error(f"Queda / Estável — Probabilidade: {prob_next*100:.2f}% 📉")
