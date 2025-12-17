# app.py 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score
import plotly.graph_objects as go
import os

# ==============================
# Configuração Streamlit
# ==============================
st.set_page_config(page_title="Previsão IBOVESPA (CatBoost)", layout="centered")
st.title("📈 Tendência IBOVESPA - CatBoost")

CSV_FILE = "base_de_dados.csv"
THRESHOLD = 0.49
TEST_SIZE = 30

# ==============================
# Sidebar
# ==============================
st.sidebar.header("⚙️ Painel de Controle")
janela_grafico = st.sidebar.slider("Janela de análise (pregões)", 20, 300, 50, 10)
mostrar_targets = st.sidebar.checkbox("Mostrar últimos targets reais", value=True)

# ==============================
# Carregar artefatos
# ==============================
model = joblib.load("modelo_final_catboost.joblib")
scaler = joblib.load("scaler_dados_ibovespa.joblib")
features_saved = joblib.load("colunas_treinamento.joblib").columns.tolist()

# ==============================
# Funções auxiliares (IGUAIS AO JUPYTER)
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


def calculate_slope(series, window):
    from scipy.stats import linregress
    slopes = [np.nan] * (window - 1)
    for i in range(window, len(series) + 1):
        y = series.iloc[i - window:i].values
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)


def calcular_rsi(df, periodo=14):
    delta = df['close'].diff()
    ganho = delta.where(delta > 0, 0)
    perda = -delta.where(delta < 0, 0)
    media_ganho = ganho.rolling(periodo).mean()
    media_perda = perda.rolling(periodo).mean()
    rs = media_ganho / media_perda
    rs.loc[media_perda == 0] = np.inf
    return 100 - (100 / (1 + rs))


def calcular_obv(df):
    return (np.sign(df['close'].diff()) * df['volume']).cumsum()


def categorizar_periodo(data):
    if data.year <= 2009:
        return "crise_2005_2009"
    elif data.year <= 2019:
        return "pre_pandemia_2010_2019"
    elif data.year <= 2022:
        return "pandemia_2020_2022"
    else:
        return "recente_2023_atual"

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
# Engenharia de features (SUBCONJUNTO DO JUPYTER)
# ==============================

dados['open_lag_1'] = dados['open'].shift(1)
dados['high_lag_1'] = dados['high'].shift(1)
dados['low_lag_1'] = dados['low'].shift(1)
dados['volume_lag_1'] = dados['volume'].shift(1)
dados['var_pct_lag_1'] = dados['var_pct'].shift(1)

for lag in [5, 10, 15, 20]:
    dados[f'var_pct_lag_{lag}'] = dados['var_pct'].shift(lag)

# Retornos
dados['return_1w'] = dados['close'].pct_change(5)
dados['return_2m'] = dados['close'].pct_change(60)

dados['volume_pct_change'] = dados['volume'].pct_change()
dados['daily_range'] = dados['high'] - dados['low']

dados['slope_20d'] = calculate_slope(dados['close'], 20)
dados['rsi'] = calcular_rsi(dados)
dados['obv'] = calcular_obv(dados)

# Período categórico
dados['periodo'] = dados.index.map(categorizar_periodo)
dados['periodo'] = pd.Categorical(
    dados['periodo'],
    categories=["crise_2005_2009", "pre_pandemia_2010_2019", "pandemia_2020_2022", "recente_2023_atual"],
    ordered=True
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
# Dashboard (MANTIDO)
# ==============================
st.subheader("📌 Visão Geral")
st.metric("📅 Último Pregão", X.index.max().strftime('%d/%m/%Y'))
st.metric("📊 Registros", len(X))

# ==============================
# Gráfico de preços
# ==============================
st.subheader("📊 Tendência do IBOV")
dados['MA_20'] = dados['close'].rolling(20).mean()
dados['MA_50'] = dados['close'].rolling(50).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=dados.index[-200:], y=dados['close'][-200:], name='Preço'))
fig.add_trace(go.Scatter(x=dados.index[-200:], y=dados['MA_20'][-200:], name='MA 20'))
fig.add_trace(go.Scatter(x=dados.index[-200:], y=dados['MA_50'][-200:], name='MA 50'))
st.plotly_chart(fig, use_container_width=True)

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

    c1, c2, c3 = st.columns(3)
    c1.metric("Acurácia", f"{acc:.3f}")
    c2.metric("Precisão", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")

    st.subheader("🔮 Próximo Pregão")
    next_proba = model.predict_proba(Pool(X.iloc[[-1]], cat_features=cat_features))[0, 1]

    if next_proba >= THRESHOLD:
        st.success(f"ALTA ({next_proba*100:.2f}%) 📈")
    else:
        st.error(f"QUEDA/ESTÁVEL ({next_proba*100:.2f}%) 📉")
