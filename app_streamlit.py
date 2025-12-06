import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import Pool
from scipy.stats import linregress
from datetime import datetime, timedelta

# *******************************************************************
# ⚠️ CAMINHO DA BASE DE DADOS LOCAL
DATA_FILE = 'Dados Históricos - Ibovespa 20 anos.csv'
# *******************************************************************

# --- FUNÇÕES DE PRÉ-PROCESSAMENTO E FEATURE ENGINEERING (IDÊNTICAS AO TREINO) ---

def tratar_coluna_volume(coluna_volume):
    """Converte o volume de string (com k, M, B) para float."""
    coluna_tratada = coluna_volume.astype(str).copy()
    multiplicadores = {'k': 1_000, 'M': 1_000_000, 'B': 1_000_000_000}
    for sufixo, multiplicador in multiplicadores.items():
        mask = coluna_tratada.str.contains(sufixo, case=False, na=False)
        coluna_tratada.loc[mask] = (
            coluna_tratada.loc[mask]
            .str.replace(sufixo, '', case=False)
            .str.replace(',', '.')
            .astype(float)
            * multiplicador
        )
    coluna_tratada = pd.to_numeric(coluna_tratada, errors='coerce')
    return coluna_tratada

def calculate_slope(data, window):
    """Calcula a inclinação da regressão linear em janelas móveis."""
    slopes = [np.nan] * (window - 1)
    for i in range(window, len(data) + 1):
        y = data[i-window:i]
        x = np.arange(len(y))
        slope, _, _, _, _ = linregress(x, y)
        slopes.append(slope)
    return slopes

def calcular_rsi(dados, periodo=14):
    """Calcula o RSI."""
    delta = dados['close'].diff(); ganho = delta.where(delta > 0, 0)
    perda = -delta.where(delta < 0, 0); media_ganho = ganho.rolling(window=periodo, min_periods=periodo).mean()
    media_perda = perda.rolling(window=periodo, min_periods=periodo).mean(); rs = media_ganho / media_perda
    rs.loc[media_perda == 0] = np.inf; rsi = 100 - (100 / (1 + rs)); return rsi

def calcular_obv(dados):
    """Calcula o On-Balance Volume (OBV)."""
    direcao = np.sign(dados['close'].diff()); obv = (direcao * dados['volume']).cumsum(); return obv

def calcular_close_position(dados):
    """Calcula a Posição do Fechamento."""
    faixa_de_preco = dados['high'] - dados['low']; posicao = (dados['close'] - dados['low']) / faixa_de_preco
    posicao.loc[faixa_de_preco == 0] = 0.5; return posicao

def categorizar_periodo(data):
    """Cria a feature categórica de regime de mercado."""
    if data.year <= 2009: return "crise_2005_2009"
    elif data.year <= 2019: return "pre_pandemia_2010_2019"
    elif data.year <= 2022: return "pandemia_2020_2022"
    else: return "recente_2023_atual"

def criar_features_para_predicao(dados_com_historico, scaler):
    """Cria todas as features e retorna a linha final pronta para predição."""

    # 1. Renomear, Indexar e Limpar
    dados = dados_com_historico.rename(columns=({'Último': 'close', 'Abertura': 'open', 'Máxima': 'high', 'Mínima': 'low', 'Vol.': 'volume', 'Var%': 'var_pct'}))
    dados['Data'] = pd.to_datetime(dados['Data'])
    dados = dados.set_index('Data').sort_index()

    dados['var_pct'] = pd.to_numeric(dados['var_pct'], errors='coerce')
    dados['volume'] = tratar_coluna_volume(dados['volume'])

    # 2. Log e Scaling (Usando o scaler pré-treinado)
    dados['volume'] = np.log1p(dados['volume'])
    features_to_scale = ['volume', 'var_pct']
    scaled_data = dados[features_to_scale].copy()
    dados.loc[:, features_to_scale] = scaler.transform(scaled_data)

    # 3. Criação de features (Lags, Retornos, Indicadores...) - IDÊNTICO AO TREINO
    for lag in [1]:
        for col in ['open', 'high', 'low', 'volume', 'var_pct']:
            dados[f"{col}_lag_{lag}"] = dados[col].shift(lag)
    for lag in [5, 10, 15, 20]:
        dados[f"var_pct_lag_{lag}"] = dados["var_pct"].shift(lag)

    dados['return_1w'] = dados['close'].pct_change(periods=5)
    dados['return_2m'] = dados['close'].pct_change(periods=60)
    dados['volume_pct_change'] = dados['volume'].pct_change()
    dados['daily_range'] = dados['high'] - dados['low']
    dados['force_index'] = (dados['close'].diff()) * dados['volume']
    dados['force_index_2d'] = dados['force_index'].rolling(window=2).mean()

    dados['rsi'] = calcular_rsi(dados)
    dados['obv'] = calcular_obv(dados)
    dados['close_position'] = calcular_close_position(dados)

    for col in ['rsi', 'obv', 'close_position', 'volume', 'volume_pct_change', 'daily_range']:
        dados[f"{col}_lag_1"] = dados[col].shift(1)

    dados['volatility_short'] = dados['daily_range'].rolling(window=20).std()
    dados['volatility_long'] = dados['daily_range'].rolling(window=100).std()
    dados['volatility_ratio'] = dados['volatility_short'] / (dados['volatility_long'] + 1e-6)
    dados['volatility_ratio'] = dados['volatility_ratio'].replace([np.inf, -np.inf], np.nan).bfill()

    dados['force_index_pct_change'] = dados['force_index'].pct_change()
    dados['force_index_diff'] = dados['force_index'].diff()

    periodo_dtype = pd.CategoricalDtype(categories=["crise_2005_2009", "pre_pandemia_2010_2019", "pandemia_2020_2022", "recente_2023_atual"], ordered=True)
    dados['periodo'] = dados.index.map(categorizar_periodo).astype(periodo_dtype)

    # 4. Seleção e Limpeza da Linha Final
    final_features = [
        'open', 'volume', 'var_pct', 'return_1w', 'return_2m', 'volume_pct_change', 'close_position',
        'daily_range', 'force_index', 'force_index_2d', 'slope_20d', 'rsi', 'obv',
        'volatility_short', 'volatility_long', 'volatility_ratio', 'force_index_pct_change',
        'force_index_diff', 'periodo', 'open_lag_1', 'high_lag_1', 'low_lag_1', 'volume_lag_1',
        'var_pct_lag_1', 'var_pct_lag_5', 'var_pct_lag_10', 'var_pct_lag_15', 'var_pct_lag_20',
        'rsi_lag_1', 'obv_lag_1', 'close_position_lag_1', 'volume_pct_change_lag_1',
        'daily_range_lag_1'
    ]
    X_pred = dados.iloc[[-1]][final_features].copy()
    X_pred.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # --- BLOCO DE IMPUTAÇÃO CORRIGIDO E SEGURO ---

    # Imputa colunas NUMÉRICAS com 0
    numeric_cols = X_pred.select_dtypes(include=np.number).columns
    X_pred.loc[:, numeric_cols] = X_pred[numeric_cols].fillna(0) 

    # Imputa colunas CATEGÓRICAS com a moda (categoria mais frequente no histórico de predição)
    categorical_cols = X_pred.select_dtypes(include='category').columns
    for col in categorical_cols:
        # Pega a moda (valor mais frequente)
        imputer_value = X_pred[col].mode()[0]
        # Atribuição segura com .loc para evitar SettingWithCopy/TypeError
        X_pred.loc[:, col] = X_pred[col].fillna(imputer_value)

    # --- FIM DO BLOCO CORRIGIDO ---

    # Ajustar a coluna categórica (mantido, mas não estritamente necessário se a imputação foi correta)
    for col in X_pred.select_dtypes(include=['category']).columns.tolist():
        X_pred[col] = X_pred[col].astype(periodo_dtype)

    return X_pred

# --- 2. CARGA DE DADOS HISTÓRICOS E MODELOS ---

@st.cache_data
def load_historical_data():
    """Carrega dados históricos, calcula o slope (que depende de todo o histórico) e retorna as últimas linhas."""
    try:
        data = pd.read_csv(DATA_FILE)
        # Renomeia para colunas originais
        data = data.rename(columns={'Vol.': 'volume', 'Var%': 'var_pct', 'Último': 'close', 'Abertura': 'open', 'Máxima': 'high', 'Mínima': 'low'})
        data['Data'] = pd.to_datetime(data['Data'], format='%d.%m.%Y')
        data.dropna(subset=['close', 'open', 'high', 'low'], inplace=True)

        # ⚠️ CORREÇÃO DA ENTRADA: Limpa e converte 'var_pct' para float
        data['var_pct'] = data['var_pct'].astype(str).str.replace(',', '.').str.replace('%', '').astype(float)
        
        # Calcula Slope (depende da coluna 'Último' original)
        data['slope_20d'] = calculate_slope(data['close'], window=20)
        
        # Seleciona o histórico necessário (ex: 100 dias, suficiente para return_2m e slope_20d)
        return data[['Data', 'close', 'open', 'high', 'low', 'volume', 'var_pct', 'slope_20d']].tail(100)
    except FileNotFoundError:
        st.error(f"Erro: Arquivo de dados '{DATA_FILE}' não encontrado. Verifique o caminho.")
        return None
        
@st.cache_resource
def load_models():
    """Carrega o modelo e o scaler salvos em .joblib."""
    try:
        # Assumindo que os arquivos .joblib estão no mesmo diretório
        model = joblib.load('catboost_model_final.joblib')
        scaler = joblib.load('scaler_volume_varpct.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("Erro: Arquivos 'catboost_model_final.joblib' e/ou 'scaler_volume_varpct.joblib' não encontrados.")
        st.warning("Certifique-se de que os arquivos do seu projeto anterior estão no mesmo diretório deste script.")
        return None, None

model, scaler = load_models()
historical_data = load_historical_data()


# --- 3. INTERFACE E LÓGICA DE PREDIÇÃO DO STREAMLIT ---

st.title("📈 Previsão de Movimento do IBOVESPA (CatBoost)")

if model and scaler and historical_data is not None:
    st.sidebar.success("Modelos e dados históricos carregados.")

    with st.form("dados_ibovespa"):
        st.subheader("Dados de Fechamento do Último Pregão")

        # Sugere a data do próximo dia útil
        last_date = historical_data['Data'].iloc[-1].date()
        data_fechamento = st.date_input("Data do Pregão (dia anterior à previsão)", value=last_date)

        # Valores iniciais baseados na última linha do histórico
        last_day = historical_data.iloc[-1]
        close = st.number_input("Último (Fechamento)", min_value=1.0, value=float(last_day['close']), step=100.0)
        open_val = st.number_input("Abertura", min_value=1.0, value=float(last_day['open']), step=100.0)
        high = st.number_input("Máxima", min_value=1.0, value=float(last_day['high']), step=100.0)
        low = st.number_input("Mínima", min_value=1.0, value=float(last_day['low']), step=100.0)

        volume = st.text_input("Vol. (ex: 5.5M, 120k)", value="5.5M") # Entrada original do notebook é string
        var_pct = st.number_input("Var% (ex: 0.82 para +0.82%)", value=float(last_day['var_pct']), format="%.2f")

        submit_button = st.form_submit_button("Realizar Predição")

    if submit_button:
        # Cria o DataFrame com o novo ponto de dado no formato 'raw'
        new_data = pd.DataFrame({
            'Data': [data_fechamento],
            'close': [close], 'open': [open_val], 'high': [high], 'low': [low],
            'volume': [volume], 'var_pct': [var_pct],
            'slope_20d': [np.nan] # Slope da última linha será calculado no 'df_full'
        })

        # Prepara o histórico para concatenação (renomeando para o formato original do CSV)
        historical_data_raw_format = historical_data.rename(columns={'close': 'Último', 'open': 'Abertura', 'high': 'Máxima', 'low': 'Mínima', 'volume': 'Vol.', 'var_pct': 'Var%', 'slope_20d': 'Slope_20d_Historico'})
        new_data_raw_format = new_data.rename(columns={'close': 'Último', 'open': 'Abertura', 'high': 'Máxima', 'low': 'Mínima', 'volume': 'Vol.', 'var_pct': 'Var%'})

        # Concatena o histórico e o novo dado
        df_full = pd.concat([historical_data_raw_format.drop(columns=['Slope_20d_Historico']), new_data_raw_format.drop(columns=['slope_20d'])], ignore_index=True)
        df_full['Data'] = pd.to_datetime(df_full['Data'])
        df_full = df_full.sort_values('Data')

        # Recalcula a feature 'slope_20d' no conjunto completo
        df_full['slope_20d'] = calculate_slope(df_full['Último'], window=20)

        # 5. Cria as Features e Prepara para a Predição (usando o scaler e as funções de FE)
        X_pred = criar_features_para_predicao(df_full, scaler)

        # 6. Realiza a Predição
        prediction_pool = Pool(data=X_pred, cat_features=X_pred.select_dtypes(include=['category']).columns.tolist())
        pred_proba = model.predict_proba(prediction_pool)[0, 1]
        
        # ⚠️ AJUSTE O THRESHOLD AQUI PARA O VALOR OTIMIZADO (Use o valor exato do seu projeto)
        OPTIMIZED_THRESHOLD = 0.55  # Exemplo: Se o seu valor era 0.55, mantenha 0.55
        prediction = 1 if pred_proba > OPTIMIZED_THRESHOLD else 0

        # 7. Exibe os Resultados
        st.subheader("Resultado da Previsão")
        if prediction == 1:
            st.success(f"**PREVISÃO:** O IBOVESPA deve **SUBIR**! (Probabilidade: {pred_proba*100:.2f}%) ⬆️")
        else:
            st.error(f"**PREVISÃO:** O IBOVESPA deve **CAIR/FICAR ESTÁVEL**! (Probabilidade: {(1-pred_proba)*100:.2f}%) ⬇️")

        st.info(f"Probabilidade de Subida (Target=1): {pred_proba*100:.2f}%")
        st.caption(f"A decisão foi tomada usando o threshold otimizado de {OPTIMIZED_THRESHOLD:.2f}.") # Adiciona informação do threshold
        st.caption("A previsão do dia D+1 exige dados completos do dia D para calcular os lags e indicadores técnicos.")

else:
    st.error("A aplicação não pode rodar devido a um erro na carga dos modelos ou dos dados históricos.")
