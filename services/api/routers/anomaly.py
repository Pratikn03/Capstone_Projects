"""API router: anomaly."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from gridpulse.anomaly.detect import detect_anomalies
from gridpulse.forecasting.baselines import persistence_24h

router = APIRouter()


class AnomalyRequest(BaseModel):
    actual: List[float]
    forecast: List[float]
    features: Optional[List[List[float]]] = None


class AnomalyResponse(BaseModel):
    residual_z: List[bool]
    iforest: List[bool]
    combined: List[bool]
    z_scores: List[float]


@router.post("", response_model=AnomalyResponse)
def post_anomalies(req: AnomalyRequest):
    # Key: API endpoint handler
    out = detect_anomalies(req.actual, req.forecast, req.features)
    return AnomalyResponse(
        residual_z=list(out["residual_z"]),
        iforest=list(out["iforest"]),
        combined=list(out["combined"]),
        z_scores=list(out["z_scores"]),
    )


@router.get("", response_model=AnomalyResponse)
def get_anomalies():
    # Use last 7 days of data with a persistence baseline to compute residuals
    features_path = "data/processed/features.parquet"
    df = pd.read_parquet(features_path)
    df = df.sort_values("timestamp")
    window = df.tail(7 * 24)

    actual = window["load_mw"].to_numpy()
    forecast = persistence_24h(window, "load_mw")

    # Trim NaNs from persistence
    mask = np.isfinite(forecast)
    actual = actual[mask]
    forecast = forecast[mask]

    out = detect_anomalies(actual, forecast, window[mask][["wind_mw", "solar_mw", "hour", "dayofweek"]])
    return AnomalyResponse(
        residual_z=list(out["residual_z"]),
        iforest=list(out["iforest"]),
        combined=list(out["combined"]),
        z_scores=list(out["z_scores"]),
    )
