from __future__ import annotations

def predict_next_24h(features_df, model_bundle):
    """Return a structured output dict ready for API/dashboard."""
    # TODO: implement quantile forecasts + intervals
    return {
        "timestamp": None,
        "forecast_load_mw": None,
        "forecast_wind_mw": None,
        "forecast_solar_mw": None,
        "confidence": None,
    }
