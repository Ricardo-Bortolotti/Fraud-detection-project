"""Streamlit app para predições interativas de detecção de fraude."""

import json
import os
from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_DIR = PROJECT_ROOT / "models" / "champion"
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Detecção de Fraude",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Detecção de Fraude em Cartões de Crédito")
st.markdown(
    "Esta interface usa o modelo campeão (selecionado por PR-AUC) para prever se uma transação é fraudulenta."
)

# Tabs para diferentes modos de entrada
tab1, tab2 = st.tabs(["Entrada Manual", "Upload CSV"])

with tab1:
    st.header("Entrada Manual")
    st.markdown("Insira os valores das features para fazer uma predição.")

    # Criar colunas para as features
    col1, col2, col3 = st.columns(3)

    with col1:
        V1 = st.number_input("V1", value=0.0, format="%.4f")
        V2 = st.number_input("V2", value=0.0, format="%.4f")
        V3 = st.number_input("V3", value=0.0, format="%.4f")
        V4 = st.number_input("V4", value=0.0, format="%.4f")
        V5 = st.number_input("V5", value=0.0, format="%.4f")
        V6 = st.number_input("V6", value=0.0, format="%.4f")
        V7 = st.number_input("V7", value=0.0, format="%.4f")
        V8 = st.number_input("V8", value=0.0, format="%.4f")
        V9 = st.number_input("V9", value=0.0, format="%.4f")
        V10 = st.number_input("V10", value=0.0, format="%.4f")

    with col2:
        V11 = st.number_input("V11", value=0.0, format="%.4f")
        V12 = st.number_input("V12", value=0.0, format="%.4f")
        V13 = st.number_input("V13", value=0.0, format="%.4f")
        V14 = st.number_input("V14", value=0.0, format="%.4f")
        V15 = st.number_input("V15", value=0.0, format="%.4f")
        V16 = st.number_input("V16", value=0.0, format="%.4f")
        V17 = st.number_input("V17", value=0.0, format="%.4f")
        V18 = st.number_input("V18", value=0.0, format="%.4f")
        V19 = st.number_input("V19", value=0.0, format="%.4f")
        V20 = st.number_input("V20", value=0.0, format="%.4f")

    with col3:
        V21 = st.number_input("V21", value=0.0, format="%.4f")
        V22 = st.number_input("V22", value=0.0, format="%.4f")
        V23 = st.number_input("V23", value=0.0, format="%.4f")
        V24 = st.number_input("V24", value=0.0, format="%.4f")
        V25 = st.number_input("V25", value=0.0, format="%.4f")
        V26 = st.number_input("V26", value=0.0, format="%.4f")
        V27 = st.number_input("V27", value=0.0, format="%.4f")
        V28 = st.number_input("V28", value=0.0, format="%.4f")
        Amount = st.number_input("Amount", value=0.0, format="%.2f")

    if st.button("Fazer Predição", type="primary"):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "V1": V1,
                    "V2": V2,
                    "V3": V3,
                    "V4": V4,
                    "V5": V5,
                    "V6": V6,
                    "V7": V7,
                    "V8": V8,
                    "V9": V9,
                    "V10": V10,
                    "V11": V11,
                    "V12": V12,
                    "V13": V13,
                    "V14": V14,
                    "V15": V15,
                    "V16": V16,
                    "V17": V17,
                    "V18": V18,
                    "V19": V19,
                    "V20": V20,
                    "V21": V21,
                    "V22": V22,
                    "V23": V23,
                    "V24": V24,
                    "V25": V25,
                    "V26": V26,
                    "V27": V27,
                    "V28": V28,
                    "Amount": Amount,
                },
                timeout=5,
            )
            response.raise_for_status()
            result = response.json()

            st.divider()
            if result["is_fraud"]:
                st.error(f"🚨 **FRAUDE DETECTADA**")
                st.metric("Probabilidade de Fraude", f"{result['probability']:.2%}")
            else:
                st.success(f"✅ **Transação Legítima**")
                st.metric("Probabilidade de Fraude", f"{result['probability']:.2%}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Não foi possível conectar à API. Certifique-se de que a API está rodando em "
                f"{API_URL}. Execute: `uvicorn api.main:app --reload`"
            )
        except Exception as e:
            st.error(f"Erro ao fazer predição: {str(e)}")

with tab2:
    st.header("Upload CSV")
    st.markdown("Faça upload de um arquivo CSV com as features para fazer predições em lote.")

    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Prévia dos dados:")
            st.dataframe(df.head())

            # Verificar se as colunas necessárias estão presentes
            required_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                st.error(f"Colunas faltando: {missing_cols}")
            else:
                if st.button("Fazer Predições em Lote", type="primary"):
                    try:
                        results = []
                        for _, row in df.iterrows():
                            response = requests.post(
                                f"{API_URL}/predict",
                                json={col: row[col] for col in required_cols},
                                timeout=5,
                            )
                            response.raise_for_status()
                            result = response.json()
                            results.append(result)

                        # Adicionar resultados ao dataframe
                        results_df = df.copy()
                        results_df["prediction"] = [r["prediction"] for r in results]
                        results_df["probability"] = [r["probability"] for r in results]
                        results_df["is_fraud"] = [r["is_fraud"] for r in results]

                        st.divider()
                        st.success("Predições concluídas!")
                        st.write("Resultados:")
                        st.dataframe(results_df)

                        # Estatísticas
                        fraud_count = sum(results_df["is_fraud"])
                        total_count = len(results_df)
                        st.metric("Total de Transações", total_count)
                        st.metric("Transações Fraudulentas", fraud_count)
                        st.metric("Taxa de Fraude", f"{fraud_count/total_count:.2%}")

                        # Botão para download
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Resultados",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )

                    except requests.exceptions.ConnectionError:
                        st.error(
                            "Não foi possível conectar à API. Certifique-se de que a API está rodando em "
                            f"{API_URL}. Execute: `uvicorn api.main:app --reload`"
                        )
                    except Exception as e:
                        st.error(f"Erro ao fazer predições: {str(e)}")

        except Exception as e:
            st.error(f"Erro ao ler arquivo CSV: {str(e)}")

# Informações do modelo
st.divider()
st.header("Informações do Modelo")

try:
    response = requests.get(f"{API_URL}/model-info", timeout=5)
    response.raise_for_status()
    model_info = response.json()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metadados")
        st.write(f"**Run ID:** {model_info['run_id']}")
        st.write(f"**Tipo de Modelo:** {model_info['model_type']}")
        st.write(f"**PR-AUC:** {model_info['pr_auc']:.4f}")

    with col2:
        st.subheader("Métricas")
        metrics = model_info["metrics"]
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                st.write(f"**{key}:** {value:.4f}")

except requests.exceptions.ConnectionError:
    st.warning(
        "Não foi possível conectar à API para obter informações do modelo. "
        "Certifique-se de que a API está rodando."
    )
except Exception as e:
    st.warning(f"Erro ao obter informações do modelo: {str(e)}")
