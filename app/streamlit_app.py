"""Streamlit app para predições interativas de detecção de fraude."""

import json
import os
from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sqlalchemy import create_engine, text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAMPION_DIR = PROJECT_ROOT / "models" / "champion"
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL")

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
tab1, tab2, tab3 = st.tabs(["Entrada Manual", "Upload JSON", "Monitoramento"])

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
    st.header("Upload JSON")
    st.markdown("Faça upload de um arquivo JSON com as features para fazer predições em lote.")

    uploaded_file = st.file_uploader("Escolha um arquivo JSON", type=["json"])

    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            
            # Verificar se é uma lista de objetos
            if not isinstance(data, list):
                st.error("O JSON deve ser uma lista de objetos com as features.")
            else:
                df = pd.DataFrame(data)
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
                            json_result = results_df.to_json(orient="records", indent=2)
                            st.download_button(
                                label="Download Resultados",
                                data=json_result,
                                file_name="predictions.json",
                                mime="application/json",
                            )

                        except requests.exceptions.ConnectionError:
                            st.error(
                                "Não foi possível conectar à API. Certifique-se de que a API está rodando em "
                                f"{API_URL}. Execute: `uvicorn api.main:app --reload`"
                            )
                        except Exception as e:
                            st.error(f"Erro ao fazer predições: {str(e)}")

        except Exception as e:
            st.error(f"Erro ao ler arquivo JSON: {str(e)}")

with tab3:
    st.header("Monitoramento de Predições")
    st.markdown("Visualize estatísticas e histórico das predições salvas no banco de dados.")
    
    # Conectar ao banco de dados
    try:
        engine = create_engine(DATABASE_URL)
        
        # Buscar dados do banco
        with engine.connect() as conn:
            # Estatísticas gerais
            stats_query = text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE is_fraud = true) as fraud_count,
                    COUNT(*) FILTER (WHERE is_fraud = false) as legitimate_count,
                    AVG(probability) as avg_probability,
                    MIN(timestamp) as first_prediction,
                    MAX(timestamp) as last_prediction
                FROM predictions
            """)
            stats = conn.execute(stats_query).fetchone()
            
            if stats.total == 0:
                st.info("Nenhuma predição registrada no banco de dados ainda.")
            else:
                # Métricas principais
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total de Predições", stats.total)
                with col2:
                    st.metric("Fraudes Detectadas", stats.fraud_count)
                with col3:
                    st.metric("Transações Legítimas", stats.legitimate_count)
                with col4:
                    fraud_rate = (stats.fraud_count / stats.total * 100) if stats.total > 0 else 0
                    st.metric("Taxa de Fraude", f"{fraud_rate:.2f}%")
                
                st.divider()
                
                # Gráfico de pizza - Distribuição de fraudes
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Distribuição de Predições")
                    pie_data = pd.DataFrame({
                        "Tipo": ["Legítima", "Fraude"],
                        "Quantidade": [stats.legitimate_count, stats.fraud_count]
                    })
                    fig_pie = px.pie(pie_data, values="Quantidade", names="Tipo", 
                                     color="Tipo", color_discrete_map={"Legítima": "green", "Fraude": "red"})
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.subheader("Probabilidade Média")
                    st.metric("Probabilidade Média de Fraude", f"{stats.avg_probability:.2%}")
                    
                    # Buscar distribuição de probabilidades
                    prob_query = text("""
                        SELECT 
                            CASE 
                                WHEN probability < 0.2 THEN '0-20%'
                                WHEN probability < 0.4 THEN '20-40%'
                                WHEN probability < 0.6 THEN '40-60%'
                                WHEN probability < 0.8 THEN '60-80%'
                                ELSE '80-100%'
                            END as prob_range,
                            COUNT(*) as count
                        FROM predictions
                        GROUP BY prob_range
                        ORDER BY prob_range
                    """)
                    prob_dist = conn.execute(prob_query).fetchall()
                    prob_df = pd.DataFrame(prob_dist, columns=["Faixa", "Quantidade"])
                    fig_bar = px.bar(prob_df, x="Faixa", y="Quantidade", 
                                    title="Distribuição de Probabilidades",
                                    color="Quantidade", color_continuous_scale="Viridis")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.divider()
                
                # Timeline de predições
                st.subheader("Timeline de Predições")
                timeline_query = text("""
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE is_fraud = true) as frauds
                    FROM predictions
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                    LIMIT 30
                """)
                timeline_data = conn.execute(timeline_query).fetchall()
                timeline_df = pd.DataFrame(timeline_data, columns=["Data", "Total", "Fraudes"])
                timeline_df = timeline_df.sort_values("Data")
                
                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Scatter(x=timeline_df["Data"], y=timeline_df["Total"], 
                                               mode="lines+markers", name="Total", line=dict(color="blue")))
                fig_timeline.add_trace(go.Scatter(x=timeline_df["Data"], y=timeline_df["Fraudes"], 
                                               mode="lines+markers", name="Fraudes", line=dict(color="red")))
                fig_timeline.update_layout(title="Predições por Data", xaxis_title="Data", yaxis_title="Quantidade")
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                st.divider()
                
                # Tabela de predições recentes
                st.subheader("Predições Recentes")
                recent_query = text("""
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT 50
                """)
                recent_data = conn.execute(recent_query).fetchall()
                recent_df = pd.DataFrame(recent_data)
                
                if not recent_df.empty:
                    # Formatar colunas para exibição
                    recent_df["timestamp"] = pd.to_datetime(recent_df["timestamp"])
                    recent_df = recent_df[["id", "timestamp", "amount", "prediction", "probability", "is_fraud"]]
                    recent_df.columns = ["ID", "Timestamp", "Amount", "Predição", "Probabilidade", "É Fraude?"]
                    st.dataframe(recent_df, use_container_width=True)
                
                # Informações de período
                st.caption(f"Período: {stats.first_prediction} até {stats.last_prediction}")
    
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {str(e)}")
        st.info("Certifique-se de que o PostgreSQL está rodando e as credenciais estão corretas.")

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
