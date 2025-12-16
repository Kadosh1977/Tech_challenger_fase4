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
THRESHOLD = 0.49  # ajuste conforme teste final
TEST_SIZE = 30
LOG_FILE = "log_previsoes.csv"

# ==============================
# Artefatos (carregar os seus arquivos existentes)
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
# Carregar e preparar dados (mantendo seu fluxo original)
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

# Log1p e escalonamento com scaler salvo (como no seu pipeline)
dados["volume"] = np.log1p(dados["volume"])

mask_scale = ~dados[["volume", "var_pct"]].isnull().any(axis=1)
dados.loc[mask_scale, ["volume", "var_pct"]] = scaler.transform(dados.loc[mask_scale, ["volume", "var_pct"]])

# Dados brutos para uso no gráfico

dados_graf = pd.read_csv("Dados Históricos - Ibovespa 20 anos.csv", decimal=",", thousands=".")
dados_graf['Var_pct'] = (
    dados_graf['Var%']
    .str.replace('%', '', regex=False)
    .str.replace('.', '', regex=False)   
    .str.replace(',', '.', regex=False)  
    .astype(float)                      
)

# ==============================
# Engenharia de features (replicando o que você já tinha)
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
for col in features_saved:
    if col not in X_full.columns:
        X_full[col] = 0

X = X_full[features_saved].copy()
y = dados["target"].loc[X.index]

if X.empty or y.empty:
    st.error("Após o pré-processamento, não há dados suficientes para predição/validação. Verifique o CSV e as features.")
    st.stop()

# ==============================
# Seleções para última data e janela de TEST_SIZE
# ==============================
ultima_data = X.index.max()
X_last = X.loc[[ultima_data]]
y_last = y.loc[ultima_data]

if len(X) < TEST_SIZE:
    st.warning(f"Poucos dados após engenharia de features. Tamanho disponível: {len(X)}. Ajuste TEST_SIZE={TEST_SIZE}.")
    TEST_SIZE = max(1, len(X))

X_test = X.iloc[-TEST_SIZE:].copy()
y_test = y.iloc[-TEST_SIZE:].copy()

# ==============================
# Função para construir features do próximo pregão (recriando exatamente o pipeline)
# ==============================
def construir_features_proximo_pregao(dados, features_saved, scaler):
    # 'dados' já contém todas as colunas e indicadores até a última data real.
    last = dados.iloc[-1:].copy()
    proxima_data = last.index[0] + pd.Timedelta(days=1)

    # criar uma linha future inicialmente copiada da última (valores base)
    future = last.copy()
    future.index = [proxima_data]

    # concat para recalcular lags/rolling para o índice futuro
    temp = pd.concat([dados, future])

    # Recalcular lags e indicadores para o conjunto temp (apenas necessário para a última linha)
    temp["open_lag_1"] = temp["open"].shift(1)
    temp["high_lag_1"] = temp["high"].shift(1)
    temp["low_lag_1"] = temp["low"].shift(1)
    temp["volume_lag_1"] = temp["volume"].shift(1)
    temp["var_pct_lag_1"] = temp["var_pct"].shift(1)
    for lag in [5, 10, 15, 20, 30]:
        temp[f"var_pct_lag_{lag}"] = temp["var_pct"].shift(lag)

    temp["return_1w"] = temp["close"].pct_change(periods=5)
    temp["return_2m"] = temp["close"].pct_change(periods=60)
    temp["volume_pct_change"] = temp["volume"].pct_change()
    temp["close_position"] = calcular_close_position(temp)
    temp["daily_range"] = temp["high"] - temp["low"]
    temp["slope_20d"] = calculate_slope(temp["close"], window=20)
    temp["slope_50d"] = calculate_slope(temp["close"], window=50)
    temp["rsi"] = calcular_rsi(temp)
    temp["obv"] = calcular_obv(temp)
    temp["rsi_lag_1"] = temp["rsi"].shift(1)
    temp["obv_lag_1"] = temp["obv"].shift(1)
    temp["close_position_lag_1"] = temp["close_position"].shift(1)
    temp["volume_pct_change_lag_1"] = temp["volume_pct_change"].shift(1)
    temp["daily_range_lag_1"] = temp["daily_range"].shift(1)
    temp["volatility_short"] = temp["daily_range"].rolling(window=5).std()
    temp["volatility_long"] = temp["daily_range"].rolling(window=30).std()
    temp["volatility_ratio"] = temp["volatility_short"] / (temp["volatility_long"] + 1e-6)
    temp["volatility_ratio"] = temp["volatility_ratio"].replace([np.inf, -np.inf], np.nan).bfill()
    temp["force_index"] = (temp["close"].diff()) * temp["volume"]
    temp["force_index_2d"] = temp["force_index"].rolling(window=2).mean()
    temp["force_index_5d"] = temp["force_index"].rolling(window=5).mean()
    temp["force_index_pct_change"] = temp["force_index"].pct_change()
    temp["force_index_diff"] = temp["force_index"].diff()
    temp["periodo"] = temp.index.map(categorizar_periodo)

    # Selecionar a última linha (o future recalculado)
    future_row = temp.iloc[[-1]].copy()

    # Garantir volume em log1p (já aplicado no dataset original; se necessário, aplica)
    if "volume" in future_row.columns:
        try:
            future_row["volume"] = np.log1p(np.expm1(future_row["volume"]))  # idempotente se já log1p
        except Exception:
            pass

    # Aplicar scaler nos mesmos campos que foram escalados originalmente
    try:
        cols_to_scale = [c for c in ["volume", "var_pct"] if c in future_row.columns]
        if len(cols_to_scale) > 0:
            scaled = scaler.transform(future_row[cols_to_scale])
            future_row[cols_to_scale] = scaled
    except Exception:
        pass

    # Montar X_future alinhado às features salvas
    X_future_full = future_row.drop(columns=[c for c in ["close", "high", "low", "target"] if c in future_row.columns], errors="ignore")
    for col in features_saved:
        if col not in X_future_full.columns:
            X_future_full[col] = 0

    X_future = X_future_full[features_saved].copy()
    return X_future

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

# Análises temporais informativas
st.subheader("📊 Análises Temporais do IBOVESPA")
dados['MA_20'] = dados['close'].rolling(20).mean()
dados['MA_50'] = dados['close'].rolling(50).mean()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dados.index[-200:], y=dados['close'][-200:], name="Preço"))
fig2.add_trace(go.Scatter(x=dados.index[-200:], y=dados['MA_20'][-200:], name="MA 20"))
fig2.add_trace(go.Scatter(x=dados.index[-200:], y=dados['MA_50'][-200:], name="MA 50"))
fig2.update_layout(title="Tendência do IBOV — Últimos 200 dias")
st.plotly_chart(fig2, use_container_width=True)

#st.subheader("📊 Análises Temporais do IBOVESPA")

# Gráfico da variação percentual (últimos 200 dias)
fig3 = go.Figure()
fig3.add_trace(
    go.Scatter(
        x=pd.to_datetime(dados_graf['Data'], format="%d.%m.%Y"),
        y=dados_graf['Var_pct'],
        name="Variação diária (%)"
    )
)
fig3.update_layout(
    title="Variação diária do IBOV — Últimos 200 pregões",
    yaxis_title="Variação (%)",
    yaxis=dict(tickformat=".2f")
)

st.plotly_chart(fig3, use_container_width=True, key="ibov_var_pct")

# ==============================
# Botão: validação TEST_SIZE dias + predição do próximo pregão (com features futuras)
# ==============================
if st.button("📊 Realizar Predição"):
    # Validação últimos TEST_SIZE dias
    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= THRESHOLD).astype(int)
    acc = accuracy_score(y_test, pred_test)
    precision = precision_score(y_test, pred_test)
    recall = recall_score(y_test, pred_test)

    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia", f"{acc:.3f}")
    col2.metric("Precisão", f"{precision:.3f}")
    col3.metric("Recall", f"{recall:.3f}")

    st.subheader("🔍 Matriz de Confusão")
    cm = confusion_matrix(y_test, pred_test)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    st.pyplot(fig_cm)

    # Construir X_future com features futuras (1 dia à frente)
    X_future = construir_features_proximo_pregao(dados, features_saved, scaler)
    prob_next = model.predict_proba(X_future)[0, 1]
    pred_next = int(prob_next >= THRESHOLD)
    display_prob = float(prob_next)  # probabilidade de ALTA (classe 1)

    # GRÁFICO INTERATIVO (histórico + previsão alinhada)
    st.subheader("📈 Evolução Temporal + Previsão do Modelo")

    series_x_hist = list(X_test.index)
    series_y_hist = list(proba_test)
    proxima_data = ultima_data + pd.Timedelta(days=1)

    serie_x = series_x_hist + [proxima_data]
    serie_y = series_y_hist + [display_prob]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=serie_x,
        y=serie_y,
        mode="lines+markers",
        name="Probabilidade (Histórico + Previsão)",
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[proxima_data],
        y=[display_prob],
        mode="markers",
        name="Previsão Próximo Pregão",
        marker=dict(size=14, symbol="diamond", line=dict(width=2))
    ))
    fig.update_layout(
        title="Probabilidade de Alta (Últimos pregões + Próxima previsão)",
        xaxis_title="Data",
        yaxis_title="Probabilidade",
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # Texto com previsão (exibe probabilidade de alta e a classe prevista)
    st.subheader("🔮 Tendência para o próximo pregão")
    if pred_next == 1:
        st.success(f"PREVISÃO: Alta (Probabilidade de Alta: {display_prob*100:.2f}%) 📈")
    else:
        st.error(f"PREVISÃO: Queda/Estável (Probabilidade de Alta: {display_prob*100:.2f}%) 📉")

    # Log de uso
    log_entry = {
        "data_execucao": str(pd.Timestamp.now()),
        "data_ultimo_pregao": str(ultima_data),
        "data_previsao": str(proxima_data),
        "probabilidade_alta": float(display_prob),
        "classe_prevista": int(pred_next)
    }
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

    st.info(f"📄 Log salvo em {LOG_FILE}")
