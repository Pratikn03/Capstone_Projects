"""API router: optimize."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from fastapi import APIRouter
from pydantic import BaseModel

from gridpulse.optimizer import optimize_dispatch

router = APIRouter()


class OptimizeRequest(BaseModel):
    forecast_load_mw: Union[float, List[float]]
    forecast_renewables_mw: Union[float, List[float]]
    forecast_price_eur_mwh: Optional[Union[float, List[float]]] = None
    forecast_carbon_kg_per_mwh: Optional[Union[float, List[float]]] = None
    config: Optional[Dict[str, Any]] = None


class OptimizeResponse(BaseModel):
    dispatch_plan: Dict[str, Any]
    expected_cost_usd: Optional[float] = None
    carbon_kg: Optional[float] = None
    carbon_cost_usd: Optional[float] = None


def _load_cfg() -> dict:
    # Key: API endpoint handler
    # Prefer optimization.yaml, fallback to optimize.yaml
    for path in [Path("configs/optimization.yaml"), Path("configs/optimize.yaml")]:
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8"))
    return {}


@router.post("", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    cfg = req.config or _load_cfg()
    result = optimize_dispatch(
        req.forecast_load_mw, 
        req.forecast_renewables_mw, 
        cfg, 
        forecast_price=req.forecast_price_eur_mwh,
        forecast_carbon_kg=req.forecast_carbon_kg_per_mwh,
    )
    return OptimizeResponse(
        dispatch_plan=result,
        expected_cost_usd=result.get("expected_cost_usd"),
        carbon_kg=result.get("carbon_kg"),
        carbon_cost_usd=result.get("carbon_cost_usd"),
    )
