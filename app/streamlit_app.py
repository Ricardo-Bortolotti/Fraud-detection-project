"""Streamlit app for interactive credit card fraud predictions."""

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
    page_title="Fraud Detection",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Credit Card Fraud Detection")
st.markdown(
    "This interface uses the champion model (selected by PR-AUC) to predict if a transaction is fraudulent."
)

# Tabs for different input modes
tab1, tab2, tab3 = st.tabs(["Manual Input", "JSON Upload", "Monitoring"])

with tab1:
    st.header("Manual Input")
    st.markdown("Enter feature values to make a single prediction.")

    # Feature input columns
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

    if st.button("Make Prediction", type="primary"):
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
                st.error(f"🚨 **FRAUD DETECTED**")
                st.metric("Fraud Probability", f"{result['probability']:.2%}")
            else:
                st.success(f"✅ **Legitimate Transaction**")
                st.metric("Fraud Probability", f"{result['probability']:.2%}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API. Ensure the API is running at "
                f"{API_URL}. Run: `uvicorn api.main:app --reload`"
            )
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

with tab2:
    st.header("JSON Upload")
    st.markdown("Upload a JSON file with features for batch predictions.")

    uploaded_file = st.file_uploader("Choose a JSON file", type=["json"])

    if uploaded_file is not None:
        try:
            data = json.load(uploaded_file)
            
            # Expect a list of feature objects
            if not isinstance(data, list):
                st.error("The JSON must be a list of objects with features.")
            else:
                df = pd.DataFrame(data)
                st.write("Data preview:")
                st.dataframe(df.head())

                # Required feature columns
                required_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
                missing_cols = set(required_cols) - set(df.columns)
                if missing_cols:
                    st.error(f"Missing columns: {missing_cols}")
                else:
                    if st.button("Make Batch Predictions", type="primary"):
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

                            # Merge API results into the dataframe
                            results_df = df.copy()
                            results_df["prediction"] = [r["prediction"] for r in results]
                            results_df["probability"] = [r["probability"] for r in results]
                            results_df["is_fraud"] = [r["is_fraud"] for r in results]

                            st.divider()
                            st.success("Predictions completed!")
                            st.write("Results:")
                            st.dataframe(results_df)

                            # Batch statistics
                            fraud_count = sum(results_df["is_fraud"])
                            total_count = len(results_df)
                            st.metric("Total Transactions", total_count)
                            st.metric("Fraudulent Transactions", fraud_count)
                            st.metric("Fraud Rate", f"{fraud_count/total_count:.2%}")

                            # Download results
                            json_result = results_df.to_json(orient="records", indent=2)
                            st.download_button(
                                label="Download Results",
                                data=json_result,
                                file_name="predictions.json",
                                mime="application/json",
                            )

                        except requests.exceptions.ConnectionError:
                            st.error(
                                "Could not connect to the API. Ensure the API is running at "
                                f"{API_URL}. Run: `uvicorn api.main:app --reload`"
                            )
                        except Exception as e:
                            st.error(f"Error making predictions: {str(e)}")

        except Exception as e:
            st.error(f"Error reading JSON file: {str(e)}")

with tab3:
    st.header("Prediction Monitoring")
    st.markdown("Visualize statistics and history of predictions saved in the database.")
    
    # Database connection
    try:
        engine = create_engine(DATABASE_URL)
        
        # Query prediction history
        with engine.connect() as conn:
            # Aggregate statistics
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
                st.info("No predictions recorded in the database yet.")
            else:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Predictions", stats.total)
                with col2:
                    st.metric("Frauds Detected", stats.fraud_count)
                with col3:
                    st.metric("Legitimate Transactions", stats.legitimate_count)
                with col4:
                    fraud_rate = (stats.fraud_count / stats.total * 100) if stats.total > 0 else 0
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                
                st.divider()
                
                # Pie chart: fraud vs legitimate
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Prediction Distribution")
                    pie_data = pd.DataFrame({
                        "Type": ["Legitimate", "Fraud"],
                        "Count": [stats.legitimate_count, stats.fraud_count]
                    })
                    fig_pie = px.pie(pie_data, values="Count", names="Type", 
                                     color="Type", color_discrete_map={"Legitimate": "green", "Fraud": "red"})
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.subheader("Average Probability")
                    st.metric("Average Fraud Probability", f"{stats.avg_probability:.2%}")
                    
                    # Probability bucket distribution
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
                    prob_df = pd.DataFrame(prob_dist, columns=["Range", "Count"])
                    fig_bar = px.bar(prob_df, x="Range", y="Count", 
                                    title="Probability Distribution",
                                    color="Count", color_continuous_scale="Viridis")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                st.divider()
                
                # Prediction timeline
                st.subheader("Prediction Timeline")
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
                timeline_df = pd.DataFrame(timeline_data, columns=["Date", "Total", "Frauds"])
                timeline_df = timeline_df.sort_values("Date")
                
                fig_timeline = go.Figure()
                fig_timeline.add_trace(go.Scatter(x=timeline_df["Date"], y=timeline_df["Total"], 
                                               mode="lines+markers", name="Total", line=dict(color="blue")))
                fig_timeline.add_trace(go.Scatter(x=timeline_df["Date"], y=timeline_df["Frauds"], 
                                               mode="lines+markers", name="Frauds", line=dict(color="red")))
                fig_timeline.update_layout(title="Predictions by Date", xaxis_title="Date", yaxis_title="Count")
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                st.divider()
                
                # Recent predictions table
                st.subheader("Recent Predictions")
                recent_query = text("""
                    SELECT * FROM predictions 
                    ORDER BY timestamp DESC 
                    LIMIT 50
                """)
                recent_data = conn.execute(recent_query).fetchall()
                recent_df = pd.DataFrame(recent_data)
                
                if not recent_df.empty:
                    # Display column labels
                    recent_df["timestamp"] = pd.to_datetime(recent_df["timestamp"])
                    recent_df = recent_df[["id", "timestamp", "amount", "prediction", "probability", "is_fraud"]]
                    recent_df.columns = ["ID", "Timestamp", "Amount", "Prediction", "Probability", "Is Fraud?"]
                    st.dataframe(recent_df, use_container_width=True)
                
                # Time range caption
                st.caption(f"Period: {stats.first_prediction} to {stats.last_prediction}")
    
    except Exception as e:
        st.error(f"Error connecting to the database: {str(e)}")
        st.info("Ensure PostgreSQL is running and credentials are correct.")

# Champion model metadata
st.divider()
st.header("Model Information")

try:
    response = requests.get(f"{API_URL}/model-info", timeout=5)
    response.raise_for_status()
    model_info = response.json()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Metadata")
        st.write(f"**Run ID:** {model_info['run_id']}")
        st.write(f"**Model Type:** {model_info['model_type']}")
        st.write(f"**PR-AUC:** {model_info['pr_auc']:.4f}")

    with col2:
        st.subheader("Metrics")
        metrics = model_info["metrics"]
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                st.write(f"**{key}:** {value:.4f}")

except requests.exceptions.ConnectionError:
    st.warning(
        "Could not connect to the API to get model information. "
        "Ensure the API is running."
    )
except Exception as e:
    st.warning(f"Error getting model information: {str(e)}")
