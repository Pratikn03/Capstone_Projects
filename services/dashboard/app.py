"""Streamlit dashboard UI."""
import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="GridPulse Dashboard", layout="wide")

def fetch_json(method: str, url: str, **kwargs):
    try:
        r = requests.request(method, url, timeout=60, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def schedule_refresh(minutes: int) -> None:
    if minutes <= 0:
        return
    interval_ms = int(minutes * 60 * 1000)
    components.html(
        f"<script>setTimeout(function(){{window.location.reload();}}, {interval_ms});</script>",
        height=0,
    )


def kpi_row(total_cost=None, per_hour_cost=None, carbon_cost=None, carbon_kg=None, anomalies=None):
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Expected Cost (USD)",
        f"{total_cost:,.2f}" if total_cost is not None else "—",
        delta=(f"{per_hour_cost:,.2f}/hr" if per_hour_cost is not None else None),
        help="Total dispatch cost over the forecast horizon (energy + penalties).",
    )
    c2.metric(
        "Carbon Cost (USD)",
        f"{carbon_cost:,.2f}" if carbon_cost is not None else "—",
        delta=(f"{carbon_kg:,.0f} kg" if carbon_kg is not None else None),
        help="Carbon price × grid energy; delta shows emissions in kg if available.",
    )
    c3.metric(
        "Anomaly Count",
        int(anomalies) if anomalies is not None else "—",
        help="Count of anomaly flags in the most recent monitoring window.",
    )


def build_kpis(api_base: str, horizon: int = 24):
    forecasts, err = fetch_json("GET", f"{api_base}/forecast", params={"horizon": horizon})
    if err or not forecasts:
        return None, None, None, None, None, err or "No forecast response"

    fcasts = forecasts.get("forecasts", {})
    load = fcasts.get("load_mw", {}).get("forecast")
    wind = fcasts.get("wind_mw", {}).get("forecast")
    solar = fcasts.get("solar_mw", {}).get("forecast")
    if not (load and wind and solar):
        return None, None, None, None, None, "Forecasts missing: train models or update configs/forecast.yaml"

    renew = [w + s for w, s in zip(wind, solar)]
    optimize, err = fetch_json(
        "POST",
        f"{api_base}/optimize",
        json={"forecast_load_mw": load, "forecast_renewables_mw": renew},
    )
    if err or not optimize:
        return None, None, None, None, None, err or "Optimization failed"

    anomalies, err = fetch_json("GET", f"{api_base}/anomaly")
    if err or not anomalies:
        total_cost = optimize.get("expected_cost_usd")
        per_hour = (total_cost / len(load)) if (total_cost is not None and len(load)) else None
        return (
            total_cost,
            per_hour,
            optimize.get("carbon_cost_usd"),
            optimize.get("carbon_kg"),
            None,
            err,
        )

    combined = anomalies.get("combined", [])
    horizon = len(load)
    total_cost = optimize.get("expected_cost_usd")
    per_hour = (total_cost / horizon) if (total_cost is not None and horizon) else None
    return (
        total_cost,
        per_hour,
        optimize.get("carbon_cost_usd"),
        optimize.get("carbon_kg"),
        sum(combined),
        None,
    )


st.title("GridPulse — Operator Dashboard")
api_base = st.text_input("API Base URL", "http://localhost:8000")

refresh_minutes = st.sidebar.number_input("Auto-refresh (minutes)", min_value=0, max_value=60, value=0)
if refresh_minutes > 0:
    st.sidebar.caption(f"Auto-refresh enabled: every {refresh_minutes} min")
    schedule_refresh(refresh_minutes)

st.subheader("Live KPIs")
if st.button("Refresh KPIs"):
    total_cost, per_hour, carbon_cost, carbon_kg, anomalies, err = build_kpis(api_base)
    if err:
        st.warning(err)
    kpi_row(
        total_cost=total_cost,
        per_hour_cost=per_hour,
        carbon_cost=carbon_cost,
        carbon_kg=carbon_kg,
        anomalies=anomalies,
    )
else:
    kpi_row()

tabs = st.tabs(["Forecast", "Optimization", "Monitoring"])

with tabs[0]:
    st.subheader("Forecast")
    if st.button("Fetch Forecast"):
        payload, err = fetch_json("GET", f"{api_base}/forecast")
        if err:
            st.error(err)
        else:
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

with tabs[1]:
    st.subheader("Optimization")
    horizon = st.number_input("Horizon (hours)", value=1, min_value=1, max_value=168)
    load = st.number_input("Forecast Load (MW)", value=8000.0)
    ren = st.number_input("Forecast Renewables (MW)", value=3200.0)
    if st.button("Optimize Dispatch"):
        payload, err = fetch_json(
            "POST",
            f"{api_base}/optimize",
            json={"forecast_load_mw": [load] * int(horizon), "forecast_renewables_mw": [ren] * int(horizon)},
        )
        if err:
            st.error(err)
        else:
            st.json(payload)
            horizon = int(horizon)
            total_cost = payload.get("expected_cost_usd")
            per_hour = (total_cost / horizon) if (total_cost is not None and horizon) else None
            kpi_row(
                total_cost=total_cost,
                per_hour_cost=per_hour,
                carbon_cost=payload.get("carbon_cost_usd"),
                carbon_kg=payload.get("carbon_kg"),
            )

with tabs[2]:
    st.subheader("Monitoring")
    if st.button("Run Monitoring"):
        payload, err = fetch_json("GET", f"{api_base}/monitor")
        if err:
            st.error(err)
        else:
            st.json(payload)

    if st.button("Fetch Anomalies"):
        payload, err = fetch_json("GET", f"{api_base}/anomaly")
        if err:
            st.error(err)
        else:
            st.json(payload)
            combined = payload.get("combined", [])
            kpi_row(anomalies=sum(combined) if combined else 0)

st.caption("Live operator view for forecasting, optimization, and monitoring.")
