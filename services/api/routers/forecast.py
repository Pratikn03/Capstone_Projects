from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter()

class ForecastResponse(BaseModel):
    timestamp: Optional[str] = None
    forecast_load_mw: Optional[float] = None
    forecast_wind_mw: Optional[float] = None
    forecast_solar_mw: Optional[float] = None
    confidence: Optional[float] = None
    meta: Dict[str, Any] = {}

@router.get("", response_model=ForecastResponse)
def get_forecast():
    # TODO: wire real model inference
    return ForecastResponse(meta={"note": "placeholder"})
