from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, Optional

router = APIRouter()

class OptimizeRequest(BaseModel):
    forecast_load_mw: float
    forecast_renewables_mw: float

class OptimizeResponse(BaseModel):
    dispatch_plan: Dict[str, Any]
    expected_cost_usd: Optional[float] = None
    carbon_kg: Optional[float] = None

@router.post("", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    # TODO: call optimizer engine
    plan = {"renewables": req.forecast_renewables_mw, "grid": max(0.0, req.forecast_load_mw - req.forecast_renewables_mw), "battery": 0.0}
    return OptimizeResponse(dispatch_plan=plan)
