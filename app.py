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
    return pd.to_numeric(coluna_

