# app.py 
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
import joblib
from catboost import Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import os

# ==============================
# Configuração Streamlit
# ==============================
st.set_page_config(page_title="Previsão IBOVESPA (CatBoost)", layout="centered")
st.title("📈 Tendência IBOVESPA - CatBoost")

CSV_FILE = "base_de_dados.csv"
THRESHOLD = 0.55
TEST_SIZE = 30

# ==============================
# Carregar artefatos
# ==============================
model = joblib.load("modelo_final_catboost.joblib")
scaler = joblib.load("scaler_dados_ibovespa.joblib")
features_saved = joblib.load("colunas_treinamento.joblib").columns.tolist()

# ==============================
# Funções auxiliares 
# ==============================
def tratar_coluna_volume(coluna):
    coluna = coluna.astype(str).copy()
    mult = {'k': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    for s, m in mult.items():
        mask = coluna.str.contains(s, case=False, na=False)
        coluna.loc[mask] = (
            coluna.loc[mask]
            .str.replace(s, '', case=False)
            .str.replace(',', '.')
            .astype(float) * m
        )
    return pd.to_numeric(coluna, errors='coerce')

# ==============================
# Carregar e preparar dados
# ==============================
dados = pd.read_csv(CSV_FILE)
dados['Data'] = pd.to_datetime(dados['Data'], format='%d.%m.%Y', errors='coerce')
dados = dados.dropna(subset=['Data']).set_index('Data').sort_index()

dados['Var%'] = dados['Var%'].astype(str).str.replace(',', '.').str.replace('%', '').astype(float)
dados['Vol.'] = tratar_coluna_volume(dados['Vol.'])

dados = dados.rename(columns={
    'Último': 'close',
    'Abertura': 'open',
    'Máxima': 'high',
    'Mínima': 'low',
    'Vol.': 'volume',
    'Var%': 'var_pct'
})

# Target
dados['target'] = (dados['var_pct'].shift(-1) > 0).astype(int)

# Escalonamento (IGUAL AO JUPYTER)
dados['volume'] = np.log1p(dados['volume'])
dados[['volume', 'var_pct']] = scaler.transform(dados[['volume', 'var_pct']])

# ==============================
# Engenharia de features
# ==============================
#LAGS
for lag in [1]:
    dados[f"open_lag_{lag}"] = dados["open"].shift(lag)
    dados[f"high_lag_{lag}"] = dados["high"].shift(lag)
    dados[f"low_lag_{lag}"] = dados["low"].shift(lag)
    dados[f"volume_lag_{lag}"] = dados["volume"].shift(lag)
    dados[f"var_pct_lag_{lag}"] = dados["var_pct"].shift(lag)

for lag in [5, 10, 15, 20]:
    dados[f"var_pct_lag_{lag}"] = dados["var_pct"].shift(lag)

# Retorno Semanal e Mensal (considerando 5 e 60 dias de pregão)
dados['return_1w'] = dados['close'].pct_change(periods=5)
dados['return_2m'] = dados['close'].pct_change(periods=60)

dados['volume_pct_change'] = dados['volume'].pct_change()
# Criação da feature de Posição do Fechamento
dados['close_position'] = (dados['close'] - dados['low']) / (dados['high'] - dados['low'])
# Trata divisões por zero, caso low == high
dados.loc[dados['high'] == dados['low'], 'close_position'] = 0.5

# Adiciona o range do dia
dados['daily_range'] = dados['high'] - dados['low']

# Calcula o Force Index de 2 dias (exemplo)
dados['force_index'] = (dados['close'].diff()) * dados['volume']
dados['force_index_2d'] = dados['force_index'].rolling(window=2).mean()


#INCLINAÇÃO DA LINHA DE TENDENCIA

from scipy.stats import linregress
import numpy as np

# --- Função para calcular a inclinação de uma janela ---
def calculate_slope(data, window):
    # Cria uma lista de NaN para as primeiras janelas
    slopes = [np.nan] * (window - 1)

    # Itera sobre o DataFrame para calcular a inclinação em janelas móveis
    for i in range(window, len(data) + 1):
        y = data[i-window:i]
        x = np.arange(len(y))

        # Realiza a regressão linear
        slope, _, _, _, _ = linregress(x, y)
        slopes.append(slope)

    return slopes

# --- Adicionando o feature de inclinação ao DataFrame ---
# Calcule a inclinação para janelas de 20 dias
dados['slope_20d'] = calculate_slope(dados['close'], window=20)


#FEATURES DE INDICADORES TÉCNICOS
def calcular_rsi(dados, periodo=14):
    delta = dados['close'].diff()
    ganho = delta.where(delta > 0, 0)
    perda = -delta.where(delta < 0, 0)
    media_ganho = ganho.rolling(window=periodo, min_periods=periodo).mean()
    media_perda = perda.rolling(window=periodo, min_periods=periodo).mean()
    rs = media_ganho / media_perda
    rs.loc[media_perda == 0] = np.inf
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcular_obv(dados):
    direcao = np.sign(dados['close'].diff())
    obv = (direcao * dados['volume']).cumsum()
    return obv

def calcular_close_position(dados):
    faixa_de_preco = dados['high'] - dados['low']
    posicao = (dados['close'] - dados['low']) / faixa_de_preco
    posicao.loc[faixa_de_preco == 0] = 0.5
    return posicao

# Crie as features de volume e preço originais
dados['volume_pct_change'] = dados['volume'].pct_change()
dados['daily_range'] = dados['high'] - dados['low']

# Crie as novas features (RSI, OBV, close_position)
dados['rsi'] = calcular_rsi(dados)
dados['obv'] = calcular_obv(dados)
dados['close_position'] = calcular_close_position(dados)

# Lags para Prevenir Vazamento de Dados

dados['rsi_lag_1'] = dados['rsi'].shift(1)
dados['obv_lag_1'] = dados['obv'].shift(1)
dados['close_position_lag_1'] = dados['close_position'].shift(1)

# Lags para as features de volume e range
dados['volume_lag_1'] = dados['volume'].shift(1)
dados['volume_pct_change_lag_1'] = dados['volume_pct_change'].shift(1)
dados['daily_range_lag_1'] = dados['daily_range'].shift(1)

# Calcular a volatilidade de curto e longo prazo
short_window = 20
long_window = 100

dados.loc[:, 'volatility_short'] = dados['daily_range'].rolling(window=short_window).std()
dados.loc[:, 'volatility_long'] = dados['daily_range'].rolling(window=long_window).std()

# Calcular a proporção de volatilidade
dados.loc[:, 'volatility_ratio'] = (
    dados['volatility_short'] / (dados['volatility_long'] + 1e-6)
)

# Tratar os valores infinitos e nulos
import numpy as np

dados.loc[:, 'volatility_ratio'] = (
    dados['volatility_ratio']
    .replace([np.inf, -np.inf], np.nan)
    .bfill()
)

#SENTIMENTO DO MERCADO
# Aceleração da Força (variação percentual)
dados['force_index_pct_change'] = dados['force_index'].pct_change()

# Aceleração da Força (diferença)
dados['force_index_diff'] = dados['force_index'].diff()

#POSIÇÃO RELATIVA E NORMALIZAÇÃO
# Aceleração da Força (variação percentual)
dados['force_index_pct_change'] = dados['force_index'].pct_change()

# Aceleração da Força (diferença)
dados['force_index_diff'] = dados['force_index'].diff()

#RELEVANCIA TEMPORAL PARA IDENTIFICAR MUDANÇAS DE REGIMES NO MERCADO
# Função para categorizar por períodos históricos
def categorizar_periodo(data):
    if data.year <= 2009:
        return "crise_2005_2009"
    elif data.year <= 2019:
        return "pre_pandemia_2010_2019"
    elif data.year <= 2022:
        return "pandemia_2020_2022"
    else:
        return "recente_2023_atual"

# 1. Criar coluna categórica
dados['periodo'] = dados.index.map(categorizar_periodo)

# 2. Transformar em tipo categórico explícito
dados['periodo'] = dados['periodo'].astype(
    pd.CategoricalDtype(
        categories=["crise_2005_2009", "pre_pandemia_2010_2019", "pandemia_2020_2022", "recente_2023_atual"],
        ordered=True
    )
)

# Limpeza
dados = dados.dropna()

# ==============================
# X e y finais
# ==============================
X = dados.drop(columns=['close', 'high', 'low', 'target'])
y = dados['target']

# Garantir mesmas colunas do treino
for col in features_saved:
    if col not in X.columns:
        X[col] = np.nan
X = X[features_saved]

# ==============================
# Dashboard 
# ==============================
st.subheader("📌 Visão Geral")
st.metric("📅 Último Pregão", X.index.max().strftime('%d/%m/%Y'))
st.metric("📊 Registros", len(X))

# ==============================
# Sidebar
# ==============================
st.sidebar.header("⚙️ Painel de Controle")
janela_grafico = st.sidebar.slider(
    "Janela de análise (pregões)", 20, 300, 50, 10
)
mostrar_targets = st.sidebar.checkbox(
    "Mostrar últimos targets reais", value=True
)

# ==============================
# Preparar dados do gráfico
# ==============================
st.subheader("📊 Tendência do IBOV")

dados['MA_20'] = dados['close'].rolling(20).mean()
dados['MA_50'] = dados['close'].rolling(50).mean()

dados_plot = dados.tail(janela_grafico)
dados_plot['target_plot'] = dados_plot['target'].shift(1)

# ==============================
# Criar gráfico
# ==============================
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dados_plot.index,
    y=dados_plot['close'],
    name='Fechamento'
))

fig.add_trace(go.Scatter(
    x=dados_plot.index,
    y=dados_plot['MA_20'],
    name='MA 20'
))

fig.add_trace(go.Scatter(
    x=dados_plot.index,
    y=dados_plot['MA_50'],
    name='MA 50'
))

# Mostrar targets de alta
if mostrar_targets:
    alvos = dados_plot[dados_plot['target_plot'] == 1]

    fig.add_trace(go.Scatter(
        x=alvos.index,
        y=alvos['close'],
        mode='markers',
        name='Alta real'
    ))

st.plotly_chart(fig, use_container_width=True)


st.subheader("🧠 Probabilidade do Modelo x Preço")

# --- Janela usada no gráfico ---
dados_prob = dados.tail(janela_grafico).copy()

# --- Predição histórica (sem vazamento) ---
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
pool_hist = Pool(X.loc[dados_prob.index], cat_features=cat_features)

dados_prob['proba_modelo'] = model.predict_proba(pool_hist)[:, 1]

# --- Criar figura com eixo duplo ---
fig_prob = go.Figure()

# Preço
fig_prob.add_trace(go.Scatter(
    x=dados_prob.index,
    y=dados_prob['close'],
    name="Fechamento IBOV",
    yaxis="y1",
    line=dict(width=2)
))

# Probabilidade
fig_prob.add_trace(go.Scatter(
    x=dados_prob.index,
    y=dados_prob['proba_modelo'],
    name="Probabilidade de Alta",
    yaxis="y2",
    line=dict(dash="dot")
))

# Threshold
fig_prob.add_trace(go.Scatter(
    x=dados_prob.index,
    y=[THRESHOLD] * len(dados_prob),
    name="Threshold",
    yaxis="y2",
    line=dict(dash="dash")
))

# Targets reais
if mostrar_targets:
    alvos = dados_prob[dados_prob['target'] == 1]
    fig_prob.add_trace(go.Scatter(
        x=alvos.index,
        y=alvos['proba_modelo'],
        mode="markers",
        name="Alta Real",
        yaxis="y2"
    ))

# Layout
fig_prob.update_layout(
    height=500,
    xaxis_title="Data",
    yaxis=dict(title="Preço"),
    yaxis2=dict(
        title="Probabilidade",
        overlaying="y",
        side="right",
        range=[0, 1]
    ),
    legend=dict(orientation="h", y=-0.2)
)

st.plotly_chart(fig_prob, use_container_width=True)


# ==============================
# Predição
# ==============================
if st.button("📊 Realizar Predição"):
    X_test = X.iloc[-TEST_SIZE:]
    y_test = y.iloc[-TEST_SIZE:]

    cat_features = X_test.select_dtypes(include=['object', 'category']).columns.tolist()
    pool_test = Pool(X_test, cat_features=cat_features)

    proba = model.predict_proba(pool_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acurácia", f"{acc:.3f}")
    c2.metric("Precisão", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")
    c4.metric("F1", f"{f1:.3f}")

    st.subheader("🔮 Próximo Pregão")
    next_proba = model.predict_proba(Pool(X.iloc[[-1]], cat_features=cat_features))[0, 1]

    if next_proba >= THRESHOLD:
        st.success(f"ALTA ({next_proba*100:.2f}%) 📈")
    else:
        st.error(f"QUEDA/ESTÁVEL ({next_proba*100:.2f}%) 📉")

