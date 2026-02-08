"""API router: forecast intervals using conformal calibration."""
from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from gridpulse.forecasting.predict import predict_next_24h
from gridpulse.forecasting.uncertainty.conformal import load_conformal
from services.api.config import load_uncertainty_config, get_conformal_path
from services.api.routers.forecast import _load_cfg, _resolve_model_path, _cached_bundle

router = APIRouter()


class ForecastResponse(BaseModel):
    yhat: List[float]
    pi90_lower: Optional[List[float]] = None
    pi90_upper: Optional[List[float]] = None


@router.get("/with-intervals", response_model=ForecastResponse)
def forecast_with_intervals(
    target: str = Query(default="load_mw"),
    horizon: int = Query(default=24, ge=1, le=168),
):
    cfg = _load_cfg()
    features_path = Path(cfg.get("data", {}).get("features_path", "data/processed/features.parquet"))
    if not features_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing features file: {features_path}")

    model_path = _resolve_model_path(target, cfg)
    if not model_path:
        raise HTTPException(status_code=404, detail=f"No model bundle found for target: {target}")

    df = pd.read_parquet(features_path)
    bundle = _cached_bundle(str(model_path))
    pred = predict_next_24h(df, bundle, horizon=horizon)
    yhat = pred.get("forecast", [])
    if not yhat:
        raise HTTPException(status_code=500, detail="Forecast generation failed.")

    unc_cfg = load_uncertainty_config()
    if not unc_cfg.get("enabled", True):
        raise HTTPException(status_code=503, detail="Uncertainty config disabled.")

    conformal_path = get_conformal_path(target, unc_cfg)
    if not conformal_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing conformal artifact: {conformal_path}")

    ci = load_conformal(conformal_path)
    if ci.q_h is not None and len(ci.q_h) != horizon and ci.q_global is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Conformal intervals calibrated for horizon {len(ci.q_h)} and no global fallback is available. "
                f"Use horizon={len(ci.q_h)} or recalibrate to the requested horizon."
            ),
        )
    lower, upper = ci.predict_interval(np.asarray(yhat, dtype=float))
    return ForecastResponse(
        yhat=list(yhat),
        pi90_lower=lower.tolist(),
        pi90_upper=upper.tolist(),
    )
