"""
Optional router that exposes forecast + conformal intervals.

Integrate by including this router in services/api/main.py:
  from services.api.routers.forecast_intervals import router as intervals_router
  app.include_router(intervals_router, prefix="/forecast")

This is kept separate so you don't break your existing /forecast route.
"""
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ForecastResponse(BaseModel):
    yhat: List[float]
    pi90_lower: Optional[List[float]] = None
    pi90_upper: Optional[List[float]] = None


@router.get("/with-intervals", response_model=ForecastResponse)
def forecast_with_intervals():
    # TODO: load your model bundle + compute yhat
    # TODO: load conformal calibration object and compute intervals
    return ForecastResponse(
        yhat=[0.0, 0.0],
        pi90_lower=[-1.0, -1.0],
        pi90_upper=[1.0, 1.0],
    )
