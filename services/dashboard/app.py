"""Streamlit dashboard UI."""
import streamlit as st
import streamlit.components.v1 as components
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="GridPulse Dashboard", layout="wide")

def fetch_json(method: str, url: str, **kwargs):
    # Key: Streamlit UI layout and API calls
    try:
        r = requests.request(method, url, timeout=60, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)

def fetch_intervals(api_base: str, target: str, horizon: int):
    payload, err = fetch_json(
        "GET",
        f"{api_base}/forecast/with-intervals",
        params={"target": target, "horizon": horizon},
    )
    if err or not payload:
        return None, None, None, err or "Interval forecast unavailable"
    yhat = payload.get("yhat")
    lower = payload.get("pi90_lower")
    upper = payload.get("pi90_upper")
    if not yhat:
        return None, None, None, "Interval forecast missing yhat"
    return yhat, lower, upper, None


def interval_dict(lower, upper):
    if lower is None or upper is None:
        return None
    if len(lower) != len(upper):
        return None
    return {"lower": lower, "upper": upper}


def quantile_interval(forecast_entry, lower_q="0.1", upper_q="0.9"):
    quantiles = forecast_entry.get("quantiles", {}) if forecast_entry else {}
    return interval_dict(quantiles.get(lower_q), quantiles.get(upper_q))


def sum_intervals(lower_a, upper_a, lower_b, upper_b):
    if lower_a is None or upper_a is None or lower_b is None or upper_b is None:
        return None, None
    if len(lower_a) != len(lower_b) or len(upper_a) != len(upper_b):
        return None, None
    return [a + b for a, b in zip(lower_a, lower_b)], [a + b for a, b in zip(upper_a, upper_b)]


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

    load_int = quantile_interval(fcasts.get("load_mw", {}))
    wind_int = quantile_interval(fcasts.get("wind_mw", {}))
    solar_int = quantile_interval(fcasts.get("solar_mw", {}))
    load_yhat, load_lower, load_upper, err = fetch_intervals(api_base, "load_mw", horizon)
    if not err:
        load = load_yhat
        load_int = interval_dict(load_lower, load_upper) or load_int
    wind_yhat, wind_lower, wind_upper, err = fetch_intervals(api_base, "wind_mw", horizon)
    if not err:
        wind = wind_yhat
        wind_int = interval_dict(wind_lower, wind_upper) or wind_int
    solar_yhat, solar_lower, solar_upper, err = fetch_intervals(api_base, "solar_mw", horizon)
    if not err:
        solar = solar_yhat
        solar_int = interval_dict(solar_lower, solar_upper) or solar_int

    renew = [w + s for w, s in zip(wind, solar)]
    renew_lower, renew_upper = sum_intervals(
        wind_int["lower"] if wind_int else None,
        wind_int["upper"] if wind_int else None,
        solar_int["lower"] if solar_int else None,
        solar_int["upper"] if solar_int else None,
    )
    renew_int = interval_dict(renew_lower, renew_upper)
    opt_payload = {"forecast_load_mw": load, "forecast_renewables_mw": renew}
    if load_int:
        opt_payload["load_interval"] = load_int
    if renew_int:
        opt_payload["renewables_interval"] = renew_int
    optimize, err = fetch_json(
        "POST",
        f"{api_base}/optimize",
        json=opt_payload,
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
        payload, err = fetch_json("GET", f"{api_base}/forecast", params={"horizon": 24})
        if err:
            st.error(err)
        else:
            st.json(payload)
            forecasts = payload.get("forecasts", {})
            if "load_mw" in forecasts:
                f = forecasts["load_mw"]
                horizon = len(f.get("forecast", [])) or 24
                df = pd.DataFrame({
                    "timestamp": f["timestamp"],
                    "p50": f["forecast"],
                })
                for q, vals in f.get("quantiles", {}).items():
                    df[f"q{q}"] = vals
                yhat, lower, upper, int_err = fetch_intervals(api_base, "load_mw", horizon)
                if not int_err:
                    if yhat:
                        df["p50"] = yhat
                    if lower and upper and len(lower) == len(df) and len(upper) == len(df):
                        df["pi90_lower"] = lower
                        df["pi90_upper"] = upper
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                fig = px.line(df, x="timestamp", y=[c for c in df.columns if c != "timestamp"], title="Load Forecast (24h)")
                st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Optimization")
    horizon = st.number_input("Horizon (hours)", value=1, min_value=1, max_value=168)
    load = st.number_input("Forecast Load (MW)", value=8000.0)
    ren = st.number_input("Forecast Renewables (MW)", value=3200.0)
    use_api_intervals = st.checkbox("Use API forecast intervals", value=True)
    if st.button("Optimize Dispatch"):
        horizon = int(horizon)
        load_series = [load] * horizon
        ren_series = [ren] * horizon
        load_interval = None
        renew_interval = None
        if use_api_intervals:
            wind_series = None
            solar_series = None
            wind_interval = None
            solar_interval = None
            load_yhat, load_lower, load_upper, load_err = fetch_intervals(api_base, "load_mw", horizon)
            if not load_err:
                load_series = load_yhat
                load_interval = interval_dict(load_lower, load_upper)
            wind_yhat, wind_lower, wind_upper, wind_err = fetch_intervals(api_base, "wind_mw", horizon)
            if not wind_err:
                wind_series = wind_yhat
                wind_interval = interval_dict(wind_lower, wind_upper)
            solar_yhat, solar_lower, solar_upper, solar_err = fetch_intervals(api_base, "solar_mw", horizon)
            if not solar_err:
                solar_series = solar_yhat
                solar_interval = interval_dict(solar_lower, solar_upper)

            if wind_series and solar_series:
                ren_series = [w + s for w, s in zip(wind_series, solar_series)]

            needs_fallback = load_err or wind_err or solar_err or load_interval is None or wind_interval is None or solar_interval is None
            if needs_fallback:
                fallback, fb_err = fetch_json("GET", f"{api_base}/forecast", params={"horizon": horizon})
                if fb_err or not fallback:
                    st.warning("Interval forecasts unavailable; using manual inputs.")
                else:
                    fcasts = fallback.get("forecasts", {})
                    if load_err:
                        load_series = fcasts.get("load_mw", {}).get("forecast", load_series)
                    if load_interval is None:
                        load_interval = quantile_interval(fcasts.get("load_mw", {})) or load_interval
                    if wind_err:
                        wind_series = fcasts.get("wind_mw", {}).get("forecast", wind_series)
                    if wind_interval is None:
                        wind_interval = quantile_interval(fcasts.get("wind_mw", {})) or wind_interval
                    if solar_err:
                        solar_series = fcasts.get("solar_mw", {}).get("forecast", solar_series)
                    if solar_interval is None:
                        solar_interval = quantile_interval(fcasts.get("solar_mw", {})) or solar_interval
                    if wind_series and solar_series:
                        ren_series = [w + s for w, s in zip(wind_series, solar_series)]

            if wind_interval and solar_interval:
                renew_lower, renew_upper = sum_intervals(
                    wind_interval["lower"],
                    wind_interval["upper"],
                    solar_interval["lower"],
                    solar_interval["upper"],
                )
                renew_interval = interval_dict(renew_lower, renew_upper)

        opt_payload = {"forecast_load_mw": load_series, "forecast_renewables_mw": ren_series}
        if load_interval:
            opt_payload["load_interval"] = load_interval
        if renew_interval:
            opt_payload["renewables_interval"] = renew_interval
        payload, err = fetch_json("POST", f"{api_base}/optimize", json=opt_payload)
        if err:
            st.error(err)
        else:
            st.json(payload)
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
