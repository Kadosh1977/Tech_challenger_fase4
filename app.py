# app.py (versão final com features futuras)
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
def tratar_coluna_volume(coluna_volume):
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
# Carregar dados
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
# Engenharia de features (INALTERADA)
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
# X e y
# ==============================
X = dados.drop(columns=["close", "high", "low", "target"])
for col in features_saved:
    if col not in X.columns:
        X[col] = 0
X = X[features_saved]
y = dados["target"]

ultima_data = X.index.max()
ultima_data_fmt = ultima_data.strftime("%d/%m/%Y")

# ==============================
# DASHBOARD (APENAS ADICIONADO)
# ==============================
st.subheader("📌 Visão Geral")

c1, c2, c3 = st.columns(3)
c1.metric("📅 Último Pregão", ultima_data_fmt)
c2.metric("📊 Registros", len(dados))
c3.metric("🎯 Threshold", THRESHOLD)

# ==============================
# Gráfico de médias móveis (ORIGINAL)
# ==============================
st.subheader("📊 Análises Temporais do IBOVESPA")

dados["MA_20"] = dados["close"].rolling(20).mean()
dados["MA_50"] = dados["close"].rolling(50).mean()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dados.index[-200:], y=dados["close"][-200:], name="Preço"))
fig2.add_trace(go.Scatter(x=dados.index[-200:], y=dados["MA_20"][-200:], name="MA 20"))
fig2.add_trace(go.Scatter(x=dados.index[-200:], y=dados["MA_50"][-200:], name="MA 50"))
fig2.update_layout(title="Tendência do IBOV — Últimos 200 dias")
st.plotly_chart(fig2, use_container_width=True)

# ==============================
# Gráfico variação últimos 50 pregões (ORIGINAL)
# ==============================
dados_graf = pd.read_csv(CSV_FILE)
dados_graf["Data"] = pd.to_datetime(dados_graf["Data"], format="%d.%m.%Y", errors="coerce")
dados_graf["Var_pct"] = (
    dados_graf["Var%"]
    .astype(str)
    .str.replace("%", "")
    .str.replace(",", ".")
    .astype(float)
)
dados_graf = dados_graf.dropna().sort_values("Data").tail(50)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=dados_graf["Data"],
    y=dados_graf["Var_pct"],
    mode="lines+markers",
    name="Variação diária (%)"
))
fig3.update_layout(title="Variação diária — Últimos 50 pregões")
st.plotly_chart(fig3, use_container_width=True)

# ==============================
# Botão de predição (INALTERADO)
# ==============================
if st.button("📊 Realizar Predição"):

    X_test = X.iloc[-TEST_SIZE:]
    y_test = y.iloc[-TEST_SIZE:]

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia", f"{acc:.3f}")
    col2.metric("Precisão", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")

    st.subheader("🔮 Tendência para o próximo pregão")
    prob_next = model.predict_proba(X.iloc[[-1]])[0, 1]
    display_prob = float(prob_next)

    if prob_next >= THRESHOLD:
        st.success(f"PREVISÃO: Alta ({prob_next*100:.2f}%) 📈")
    else:
        st.error(f"PREVISÃO: Queda/Estável ({prob_next*100:.2f}%) 📉")
    st.subheader("📈 Probabilidade de Alta estimada pelo Modelo")

    # eixo histórico
    datas_hist = list(X_test.index)
    prob_hist = list(proba_test)

    # próxima data (previsão)
    proxima_data = ultima_data + pd.Timedelta(days=1)

    datas_plot = datas_hist + [proxima_data]
    prob_plot = prob_hist + [display_prob]

    fig_prob = go.Figure()

    # histórico
    fig_prob.add_trace(go.Scatter(
        x=datas_hist,
        y=prob_hist,
        mode="lines+markers",
        name="Probabilidade (Histórico)",
        line=dict(width=2)
    ))

    # ponto de previsão
    fig_prob.add_trace(go.Scatter(
        x=[proxima_data],
        y=[display_prob],
        mode="markers",
        name="Previsão Próximo Pregão",
        marker=dict(size=14, symbol="diamond")
    ))

    # linha de threshold
    fig_prob.add_hline(
        y=THRESHOLD,
        line_dash="dash",
        annotation_text="Threshold",
        annotation_position="top left"
    )

    fig_prob.update_layout(
        title="Probabilidade de Alta — Histórico + Próxima Previsão",
        xaxis_title="Data",
        yaxis_title="Probabilidade",
        yaxis=dict(range=[0, 1]),
        height=450
    )

    st.plotly_chart(fig_prob, use_container_width=True)

