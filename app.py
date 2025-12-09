# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# ==============================
# Configuração
# ==============================
st.set_page_config(page_title="Previsão IBOVESPA (CatBoost)", layout="centered")
st.title("📈 Tendência IBOVESPA - CatBoost")

CSV_FILE = "Dados Históricos - Ibovespa 20 anos.csv"
THRESHOLD = 0.49  # ajuste conforme teste final
TEST_SIZE = 30

# ==============================
# Artefatos
# ==============================
model = joblib.load("modelo_final_catboost.joblib")
scaler = joblib.load("scaler_dados_ibovespa.joblib")
features_saved = joblib.load("features_modelo_ibov.joblib")

# ==============================
# Funções auxiliares (iguais ao notebook)
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
        x_mean = x.mean()
        y_mean = y.mean()
        num = ((x - x_mean) * (y - y_mean)).sum()
        den = ((x - x_mean) ** 2).sum()
        slope = num / den if den != 0 else np.nan
        slopes.append(slope)
    return pd.Series(slopes, index=series.index)

def calcular_rsi(df: pd.DataFrame, periodo: int = 14) -> pd.Series:
    delta = df["close"].diff()
    ganho = delta.where(delta > 0, 0.0)
    perda = -delta.where(delta < 0, 0.0)
    media_ganho = ganho.rolling(window=periodo, min_periods=periodo).mean()
    media_perda = perda.rolling(window=periodo, min_periods=periodo).mean()
    rs = media_ganho / media_perda
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calcular_obv(df: pd.DataFrame) -> pd.Series:
    direcao = np.sign(df["close"].diff()).fillna(0)
    obv = (direcao * df["volume"]).cumsum()
    return obv

def calcular_close_position(df: pd.DataFrame) -> pd.Series:
    faixa = df["high"] - df["low"]
    pos = (df["close"] - df["low"]) / faixa
    pos.loc[faixa == 0] = 0.5
    return pos

def categorizar_periodo(dt: pd.Timestamp) -> str:
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
dados = dados.dropna(subset=["Data"])
dados = dados.set_index("Data").sort_index()

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

# Target do próximo pregão
dados["target"] = (dados["var_pct"].shift(-1) > 0).astype(int)

# Log1p e escalonamento com scaler salvo
dados["volume"] = np.log1p(dados["volume"])

mask_scale = ~dados[["volume", "var_pct"]].isnull().any(axis=1)
dados.loc[mask_scale, ["volume", "var_pct"]] = scaler.transform(dados.loc[mask_scale, ["volume", "var_pct"]])

# ==============================
# Engenharia de features (igual ao notebook)
# ==============================
# Lags básicos
dados["open_lag_1"] = dados["open"].shift(1)
dados["high_lag_1"] = dados["high"].shift(1)
dados["low_lag_1"] = dados["low"].shift(1)
dados["volume_lag_1"] = dados["volume"].shift(1)
dados["var_pct_lag_1"] = dados["var_pct"].shift(1)

# Lags adicionais de var_pct
for lag in [5, 10, 15, 20, 30]:
    dados[f"var_pct_lag_{lag}"] = dados["var_pct"].shift(lag)

# Retornos e mudanças
dados["return_1w"] = dados["close"].pct_change(periods=5)
dados["return_2m"] = dados["close"].pct_change(periods=60)
dados["volume_pct_change"] = dados["volume"].pct_change()

# Posição do fechamento e faixa diária
dados["close_position"] = calcular_close_position(dados)
dados["daily_range"] = dados["high"] - dados["low"]

# Slopes (tendência)
dados["slope_20d"] = calculate_slope(dados["close"], window=20)
dados["slope_50d"] = calculate_slope(dados["close"], window=50)

# Indicadores técnicos
dados["rsi"] = calcular_rsi(dados)
dados["obv"] = calcular_obv(dados)

# Lags técnicos
dados["rsi_lag_1"] = dados["rsi"].shift(1)
dados["obv_lag_1"] = dados["obv"].shift(1)
dados["close_position_lag_1"] = dados["close_position"].shift(1)
dados["volume_pct_change_lag_1"] = dados["volume_pct_change"].shift(1)
dados["daily_range_lag_1"] = dados["daily_range"].shift(1)

# Volatilidade e proporção
dados["volatility_short"] = dados["daily_range"].rolling(window=5).std()
dados["volatility_long"] = dados["daily_range"].rolling(window=30).std()
dados["volatility_ratio"] = dados["volatility_short"] / (dados["volatility_long"] + 1e-6)
dados["volatility_ratio"] = dados["volatility_ratio"].replace([np.inf, -np.inf], np.nan).bfill()

# Force Index e derivados
dados["force_index"] = (dados["close"].diff()) * dados["volume"]
dados["force_index_2d"] = dados["force_index"].rolling(window=2).mean()
dados["force_index_5d"] = dados["force_index"].rolling(window=5).mean()
dados["force_index_pct_change"] = dados["force_index"].pct_change()
dados["force_index_diff"] = dados["force_index"].diff()

# Período categórico
dados["periodo"] = dados.index.map(categorizar_periodo)
dados["periodo"] = dados["periodo"].astype(
    pd.CategoricalDtype(
        categories=["crise_2005_2009", "pre_pandemia_2010_2019", "pandemia_2020_2022", "recente_2023_atual"],
        ordered=True
    )
)

# Drop de NaNs (alinha com o treino)
dados = dados.dropna()

# ==============================
# Construir X e y com alinhamento às features salvas
# ==============================
X_full = dados.drop(columns=["close", "target", "high", "low"])
# Garantir presença e ordem exata das colunas esperadas
for col in features_saved:
    if col not in X_full.columns:
        X_full[col] = 0

# Ordena exatamente como o treino
X = X_full[features_saved].copy()
y = dados["target"].loc[X.index]

# Checagens de segurança
if X.empty or y.empty:
    st.error("Após o pré-processamento, não há dados suficientes para predição/validação. Verifique o CSV e as features.")
    st.stop()

# ==============================
# Seleções para última data e janela de 30 dias
# ==============================
ultima_data = X.index.max()
X_last = X.loc[[ultima_data]]
y_last = y.loc[ultima_data]

# Garante tamanho mínimo para validação
if len(X) < TEST_SIZE:
    st.warning(f"Poucos dados após engenharia de features. Tamanho disponível: {len(X)}. Ajuste TEST_SIZE={TEST_SIZE}.")
    TEST_SIZE = max(1, len(X))

X_test = X.iloc[-TEST_SIZE:].copy()
y_test = y.iloc[-TEST_SIZE:].copy()

# ==============================
# UI informativa mínima
# ==============================
st.subheader("📅 Última data da base")
st.write(f"Data: **{pd.to_datetime(ultima_data).date()}** (referência visual)")

st.subheader("📊 Últimas 5 targets reais")
ultimas_targets = y.tail(5)
df_targets = pd.DataFrame({
    "Data": ultimas_targets.index,
    "Target": ["Alta (1)" if val == 1 else "Baixa (0)" for val in ultimas_targets.values]
}).set_index("Data")
st.table(df_targets)

# ==============================
# Botão: validação 30 dias + predição do próximo pregão
# ==============================
if st.button("📊 Realizar Predição"):
    # Validação últimos 30 dias (usa predict_proba + THRESHOLD)
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= THRESHOLD).astype(int)
    acc = accuracy_score(y_test, pred_test)

    # Apenas acurácia
    st.subheader("✅ Acurácia")
    st.write(f"Acurácia: **{acc:.3f}**")

    # Predição para o próximo pregão (baseando na última data)
    prob_next = model.predict_proba(X_last)[0, 1]
    pred_next = int(prob_next >= THRESHOLD)

    st.subheader("🔮 Tendência para o próximo pregão")
    if pred_next == 1:
        st.success(f"PREVISÃO: Alta (Probabilidade: {prob_next*100:.2f}%) 📈")
    else:
        st.error(f"PREVISÃO: Queda/Estável (Probabilidade: {(1-prob_next)*100:.2f}%) 📉")
