import streamlit as st
import requests

st.set_page_config(page_title="GridPulse Dashboard", layout="wide")

st.title("GridPulse — Operator Dashboard (Prototype)")

api_base = st.text_input("API Base URL", "http://localhost:8000")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Fetch Forecast"):
        try:
            r = requests.get(f"{api_base}/forecast")
            st.json(r.json())
        except Exception as e:
            st.error(str(e))
with col2:
    if st.button("Fetch Anomalies"):
        try:
            r = requests.get(f"{api_base}/anomaly")
            st.json(r.json())
        except Exception as e:
            st.error(str(e))
with col3:
    st.write("Optimization demo:")
    load = st.number_input("Forecast Load (MW)", value=8000.0)
    ren = st.number_input("Forecast Renewables (MW)", value=3200.0)
    if st.button("Optimize Dispatch"):
        try:
            r = requests.post(f"{api_base}/optimize", json={"forecast_load_mw": load, "forecast_renewables_mw": ren})
            st.json(r.json())
        except Exception as e:
            st.error(str(e))

st.divider()
st.caption("This dashboard is intentionally minimal for Week 1. Expand visuals in Week 4.")
