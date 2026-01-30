import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="GridPulse Dashboard", layout="wide")

st.title("GridPulse — Operator Dashboard")
api_base = st.text_input("API Base URL", "http://localhost:8000")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Fetch Forecast"):
        try:
            r = requests.get(f"{api_base}/forecast")
            payload = r.json()
            st.json(payload)

            forecasts = payload.get("forecasts", {})
            if "load_mw" in forecasts:
                f = forecasts["load_mw"]
                df = pd.DataFrame({
                    "timestamp": f["timestamp"],
                    "p50": f["forecast"],
                })
                for q, vals in f.get("quantiles", {}).items():
                    df[f"q{q}"] = vals
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                fig = px.line(df, x="timestamp", y=[c for c in df.columns if c != "timestamp"], title="Load Forecast (24h)")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))

with col2:
    if st.button("Fetch Anomalies"):
        try:
            r = requests.get(f"{api_base}/anomaly")
            payload = r.json()
            st.json(payload)
            combined = payload.get("combined", [])
            if combined:
                st.metric("Anomaly Count", sum(combined))
        except Exception as e:
            st.error(str(e))

with col3:
    st.write("Optimization demo:")
    horizon = st.number_input("Horizon (hours)", value=1, min_value=1, max_value=168)
    load = st.number_input("Forecast Load (MW)", value=8000.0)
    ren = st.number_input("Forecast Renewables (MW)", value=3200.0)
    if st.button("Optimize Dispatch"):
        try:
            payload = {
                "forecast_load_mw": [load] * int(horizon),
                "forecast_renewables_mw": [ren] * int(horizon),
            }
            r = requests.post(f"{api_base}/optimize", json=payload)
            st.json(r.json())
        except Exception as e:
            st.error(str(e))

st.divider()

if st.button("Run Monitoring"):
    try:
        r = requests.get(f"{api_base}/monitor")
        st.json(r.json())
    except Exception as e:
        st.error(str(e))

st.caption("This dashboard summarizes forecasting, anomaly detection, optimization, and monitoring.")
