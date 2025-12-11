# app.py completo revisado

import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import plotly.graph_objects as go

# ----------------------------------------------
# Carregamento dos dados e modelo
# ----------------------------------------------
dados = pd.read_csv("dados_tratados.csv", index_col=0, parse_dates=True)
model = CatBoostClassifier().load_model("modelo_catboost.cbm")
scaler = StandardScaler()
features_saved = list(pd.read_csv("features_usadas.csv")["features"])

# ----------------------------------------------
# Função para categorizar período
# ----------------------------------------------
def categorizar_periodo(dt):
    h = dt.hour
    if h < 10:
        return "pré-abertura"
    elif h < 16:
        return "pregão"
    else:
        return "after-market"

# ----------------------------------------------
# Função que constrói features do próximo pregão
# ----------------------------------------------
def construir_features_proximo_pregao(dados, features_saved, scaler):
    df = dados.copy()
    proxima_data = df.index[-1] + pd.Timedelta(days=1)

    future = pd.DataFrame(index=[proxima_data], columns=df.columns)

    for col in df.columns:
        if "lag" in col:
            lag_n = int(col.split("_")[-1])
            future[col] = df[col].iloc[-lag_n] if len(df) > lag_n else df[col].iloc[-1]

    future["open"] = df["open"].iloc[-1]
    future["high"] = df["high"].iloc[-1]
    future["low"] = df["low"].iloc[-1]
    future["close"] = df["close"].iloc[-1]

    window_df = df.iloc[-60:].append(future)
    future["close_position"] = (future["close"] - df["low"].iloc[-1]) / (df["high"].iloc[-1] - df["low"].iloc[-1] + 1e-6)
    future["daily_range"] = df["high"].iloc[-1] - df["low"].iloc[-1]
    future["volume"] = np.log1p(df["volume"].iloc[-1])

    try:
        scaled = scaler.transform(future[["volume", "var_pct"]])
        future[["volume", "var_pct"]] = scaled
    except:
        pass

    for col in features_saved:
        if col not in future.columns:
            future[col] = df[col].iloc[-1] if col in df.columns else 0

    return future[features_saved]

# ----------------------------------------------
# Separação treino/teste
# ----------------------------------------------
X = dados[features_saved]
y = dados["target"]
X_test = X.tail(30)
y_test = y.tail(30)
X_last = X.tail(1)
ultima_data = X.index[-1]

# ----------------------------------------------
# Interface Streamlit
# ----------------------------------------------
st.title("Previsão Mercado – CatBoost")

THRESHOLD = 0.5

# ----------------------------------------------
# Botão principal
# ----------------------------------------------
if st.button("📊 Realizar Predição"):

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= THRESHOLD).astype(int)
    acc = accuracy_score(y_test, pred_test)
    prec = precision_score(y_test, pred_test)
    rec = recall_score(y_test, pred_test)

    st.subheader("📌 Métricas de Validação (últimos 30 dias)")
    st.write(f"Acurácia: **{acc:.3f}**")
    st.write(f"Precisão: **{prec:.3f}**")
    st.write(f"Recall: **{rec:.3f}**")

    X_future = construir_features_proximo_pregao(dados, features_saved, scaler)
    prob_next = model.predict_proba(X_future)[0, 1]
    pred_next = int(prob_next >= THRESHOLD)
    display_prob = float(prob_next)

    st.subheader("📈 Evolução Temporal + Previsão do Modelo")

    series_x_hist = list(X_test.index)
    series_y_hist = list(proba_test)
    proxima_data = ultima_data + pd.Timedelta(days=1)

    serie_x = series_x_hist + [proxima_data]
    serie_y = series_y_hist + [display_prob]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie_x, y=serie_y, mode="lines+markers", line=dict(width=2), name="Probabilidade (Histórico + Previsão)"))
    fig.add_trace(go.Scatter(x=[proxima_data], y=[display_prob], mode="markers", marker=dict(size=14, symbol="diamond"), name="Previsão Próximo Pregão"))

    fig.update_layout(title="Probabilidade de Alta (Últimos pregões + Próxima previsão)", xaxis_title="Data", yaxis_title="Probabilidade", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔮 Tendência para o próximo pregão")
    if pred_next == 1:
        st.success(f"PREVISÃO: Alta (Probabilidade: {display_prob*100:.2f}%) 📈")
    else:
        st.error(f"PREVISÃO: Queda/Estável (Probabilidade: {(1-display_prob)*100:.2f}%) 📉")

    log_df = pd.DataFrame({
        "data": [proxima_data],
        "probabilidade_alta": [display_prob],
        "previsao": [pred_next]
    })
    log_df.to_csv("log_previsoes.csv", mode="a", header=not pd.io.common.file_exists("log_previsoes.csv"), index=False)

    st.info("Log salvo em log_previsoes.csv")
