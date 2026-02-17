"""API router: optimize."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gridpulse.optimizer import optimize_dispatch
from gridpulse.optimizer.baselines import grid_only_dispatch
from gridpulse.optimizer.robust_dispatch import RobustDispatchConfig, optimize_robust_dispatch

router = APIRouter()


class IntervalBounds(BaseModel):
    lower: Optional[Union[float, List[float]]] = None
    upper: Optional[Union[float, List[float]]] = None


class OptimizeRequest(BaseModel):
    forecast_load_mw: Union[float, List[float]]
    forecast_renewables_mw: Union[float, List[float]]
    forecast_price_eur_mwh: Optional[Union[float, List[float]]] = None
    forecast_carbon_kg_per_mwh: Optional[Union[float, List[float]]] = None
    load_interval: Optional[IntervalBounds] = None
    renewables_interval: Optional[IntervalBounds] = None
    optimization_mode: Literal["robust", "deterministic"] = "robust"
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


def _resolve_interval(
    forecast: Union[float, List[float]],
    interval: Optional[IntervalBounds],
    label: str,
) -> tuple[Union[float, List[float]], Union[float, List[float]]]:
    if interval is None:
        return forecast, forecast
    if interval.lower is None or interval.upper is None:
        raise ValueError(f"{label}.lower and {label}.upper must both be provided")
    return interval.lower, interval.upper


def _build_robust_config(cfg: dict[str, Any]) -> RobustDispatchConfig:
    battery = cfg.get("battery", {}) if isinstance(cfg, dict) else {}
    grid = cfg.get("grid", {}) if isinstance(cfg, dict) else {}

    capacity = float(battery.get("capacity_mwh", 100.0))
    max_power = float(battery.get("max_power_mw", 50.0))
    max_charge = float(battery.get("max_charge_mw", max_power))
    max_discharge = float(battery.get("max_discharge_mw", max_power))
    efficiency = float(battery.get("efficiency", battery.get("efficiency_regime_a", 0.95)))

    return RobustDispatchConfig(
        battery_capacity_mwh=capacity,
        battery_max_charge_mw=max_charge,
        battery_max_discharge_mw=max_discharge,
        battery_charge_efficiency=efficiency,
        battery_discharge_efficiency=efficiency,
        battery_initial_soc_mwh=float(battery.get("initial_soc_mwh", capacity / 2.0)),
        battery_min_soc_mwh=float(battery.get("min_soc_mwh", 0.0)),
        battery_max_soc_mwh=float(battery.get("max_soc_mwh", capacity)),
        max_grid_import_mw=float(grid.get("max_import_mw", grid.get("max_draw_mw", 500.0))),
        default_price_per_mwh=float(grid.get("price_per_mwh", grid.get("price_usd_per_mwh", 60.0))),
        degradation_cost_per_mwh=float(battery.get("degradation_cost_per_mwh", 10.0)),
        time_step_hours=1.0,
        solver_name=str(cfg.get("solver_name", "appsi_highs")),
    )


@router.post("", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    cfg = req.config or _load_cfg()
    try:
        if req.optimization_mode == "deterministic":
            load_interval = req.load_interval.model_dump() if req.load_interval else None
            renewables_interval = req.renewables_interval.model_dump() if req.renewables_interval else None
            result = optimize_dispatch(
                req.forecast_load_mw,
                req.forecast_renewables_mw,
                cfg,
                forecast_price=req.forecast_price_eur_mwh,
                forecast_carbon_kg=req.forecast_carbon_kg_per_mwh,
                load_interval=load_interval,
                renewables_interval=renewables_interval,
            )
            return OptimizeResponse(
                dispatch_plan=result,
                expected_cost_usd=result.get("expected_cost_usd"),
                carbon_kg=result.get("carbon_kg"),
                carbon_cost_usd=result.get("carbon_cost_usd"),
            )

        load_lower, load_upper = _resolve_interval(req.forecast_load_mw, req.load_interval, "load_interval")
        robust_cfg = _build_robust_config(cfg)
        result = optimize_robust_dispatch(
            load_lower_bound=load_lower,
            load_upper_bound=load_upper,
            renewables_forecast=req.forecast_renewables_mw,
            price=req.forecast_price_eur_mwh,
            config=robust_cfg,
            verbose=False,
        )
        return OptimizeResponse(
            dispatch_plan=result,
            expected_cost_usd=result.get("total_cost"),
            carbon_kg=None,
            carbon_cost_usd=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/baseline", response_model=OptimizeResponse)
def optimize_baseline(req: OptimizeRequest):
    cfg = req.config or _load_cfg()
    result = grid_only_dispatch(
        req.forecast_load_mw,
        req.forecast_renewables_mw,
        cfg,
        price_series=req.forecast_price_eur_mwh,
        carbon_series=req.forecast_carbon_kg_per_mwh,
    )
    return OptimizeResponse(
        dispatch_plan=result,
        expected_cost_usd=result.get("expected_cost_usd"),
        carbon_kg=result.get("carbon_kg"),
        carbon_cost_usd=result.get("carbon_cost_usd"),
    )
