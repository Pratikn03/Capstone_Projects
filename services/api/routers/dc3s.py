"""API router: DC3S (Drift-Calibrated Conformal Safety Shield)."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gridpulse.dc3s.calibration import build_uncertainty_set
from gridpulse.dc3s.certificate import (
    compute_config_hash,
    compute_model_hash,
    get_certificate,
    make_certificate,
    store_certificate,
)
from gridpulse.dc3s.drift import PageHinkleyDetector
from gridpulse.dc3s.quality import compute_reliability
from gridpulse.dc3s.shield import repair_action
from gridpulse.dc3s.state import DC3SStateStore
from gridpulse.forecasting.predict import predict_next_24h
from gridpulse.forecasting.uncertainty.conformal import load_conformal
from gridpulse.optimizer import optimize_dispatch
from gridpulse.optimizer.robust_dispatch import optimize_robust_dispatch
from gridpulse.safety.bms import SafetyLayer, SafetyViolation
from services.api.config import get_bms_config, get_conformal_path, load_uncertainty_config
from services.api.routers.forecast import _cached_bundle, _load_cfg as _load_forecast_cfg, _resolve_model_path
from services.api.routers.optimize import _build_robust_config, _load_cfg as _load_optimize_cfg

router = APIRouter()


class DC3SStepRequest(BaseModel):
    device_id: str
    zone_id: Literal["DE", "US"] = "DE"
    current_soc_mwh: float = Field(..., ge=0.0)
    telemetry_event: Dict[str, Any] = Field(default_factory=dict)
    last_actual_load_mw: Optional[float] = None
    last_pred_load_mw: Optional[float] = None
    horizon: int = Field(default=24, ge=1, le=168)
    controller: Literal["deterministic", "robust", "heuristic"] = "deterministic"
    include_certificate: bool = True


class DispatchAction(BaseModel):
    charge_mw: float
    discharge_mw: float


class UncertaintyPayload(BaseModel):
    lower: List[float]
    upper: List[float]
    meta: Dict[str, Any] = Field(default_factory=dict)


class DC3SStepResponse(BaseModel):
    proposed_action: DispatchAction
    safe_action: DispatchAction
    dispatch_plan: Optional[Dict[str, Any]] = None
    uncertainty: UncertaintyPayload
    certificate_id: str
    command_id: str
    certificate: Optional[Dict[str, Any]] = None


def _load_dc3s_cfg(path: str | Path = "configs/dc3s.yaml") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise HTTPException(status_code=500, detail=f"Missing DC3S config: {cfg_path}")
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if "dc3s" not in payload or not isinstance(payload["dc3s"], dict):
        raise HTTPException(status_code=500, detail="configs/dc3s.yaml must contain top-level 'dc3s' object.")
    return payload


def _extract_ts(event: Dict[str, Any]) -> str:
    for key in ("ts_utc", "utc_timestamp", "timestamp", "ts"):
        val = event.get(key)
        if isinstance(val, str) and val:
            return val
    return datetime.now(timezone.utc).isoformat()


def _predict_target(
    *,
    target: str,
    horizon: int,
    features_df: pd.DataFrame,
    forecast_cfg: Dict[str, Any],
    required: bool,
) -> tuple[np.ndarray, Optional[Path]]:
    model_path = _resolve_model_path(target, forecast_cfg)
    if model_path is None:
        if required:
            raise HTTPException(status_code=404, detail=f"No model bundle found for target: {target}")
        return np.zeros(horizon, dtype=float), None

    bundle = _cached_bundle(str(model_path))
    pred = predict_next_24h(features_df, bundle, horizon=horizon)
    series = np.asarray(pred.get("forecast", []), dtype=float).reshape(-1)
    if series.size != horizon:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast length mismatch for {target}: expected {horizon}, got {series.size}",
        )
    return series, model_path


def _resolve_conformal_q(target: str, horizon: int) -> np.ndarray:
    unc_cfg = load_uncertainty_config()
    conformal_path = get_conformal_path(target, unc_cfg)
    if not conformal_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing conformal artifact: {conformal_path}")

    ci = load_conformal(conformal_path)
    if ci.q_h is not None and len(ci.q_h) == horizon:
        q = np.asarray(ci.q_h, dtype=float).reshape(-1)
    elif ci.q_global is not None:
        q = np.full(horizon, float(ci.q_global), dtype=float)
    elif ci.q_h is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Conformal horizon mismatch for {target}: calibrated={len(ci.q_h)} requested={horizon} "
                "and no global fallback available."
            ),
        )
    else:
        raise HTTPException(status_code=500, detail=f"Conformal artifact has no usable quantile for {target}.")

    if np.any(q < 0):
        raise HTTPException(status_code=500, detail=f"Conformal quantiles for {target} must be non-negative.")
    return q


def _build_default_price_and_carbon(features_df: pd.DataFrame, horizon: int, opt_cfg: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    grid_cfg = opt_cfg.get("grid", {}) if isinstance(opt_cfg, dict) else {}
    default_price = float(grid_cfg.get("price_per_mwh", grid_cfg.get("price_usd_per_mwh", 70.0)))
    default_carbon = float(grid_cfg.get("carbon_kg_per_mwh", 400.0))

    price_col = "price_eur_mwh" if "price_eur_mwh" in features_df.columns else "price_usd_mwh"
    if price_col in features_df.columns:
        price_series = pd.to_numeric(features_df[price_col], errors="coerce").dropna()
        latest_price = float(price_series.iloc[-1]) if not price_series.empty else default_price
    else:
        latest_price = default_price

    carbon_col = "carbon_kg_per_mwh" if "carbon_kg_per_mwh" in features_df.columns else "moer_kg_per_mwh"
    if carbon_col in features_df.columns:
        carbon_series = pd.to_numeric(features_df[carbon_col], errors="coerce").dropna()
        latest_carbon = float(carbon_series.iloc[-1]) if not carbon_series.empty else default_carbon
    else:
        latest_carbon = default_carbon

    return (
        np.full(horizon, latest_price, dtype=float),
        np.full(horizon, latest_carbon, dtype=float),
    )


def _first_action(dispatch_plan: Dict[str, Any]) -> Dict[str, float]:
    charge = dispatch_plan.get("battery_charge_mw", [0.0])
    discharge = dispatch_plan.get("battery_discharge_mw", [0.0])
    return {
        "charge_mw": float(charge[0]) if charge else 0.0,
        "discharge_mw": float(discharge[0]) if discharge else 0.0,
    }


def _load_features_df(forecast_cfg: Dict[str, Any]) -> pd.DataFrame:
    features_path = Path(forecast_cfg.get("data", {}).get("features_path", "data/processed/features.parquet"))
    if not features_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing features file: {features_path}")
    return pd.read_parquet(features_path)


@router.post("/step", response_model=DC3SStepResponse)
def dc3s_step(req: DC3SStepRequest) -> DC3SStepResponse:
    if req.controller == "heuristic":
        raise HTTPException(status_code=422, detail="controller='heuristic' is not implemented in Section 3.")

    cfg_all = _load_dc3s_cfg()
    dc3s_cfg = cfg_all["dc3s"]
    audit_cfg = dc3s_cfg.get("audit", {})
    audit_path = str(audit_cfg.get("duckdb_path", "data/audit/dc3s_audit.duckdb"))
    audit_table = str(audit_cfg.get("table_name", "dispatch_certificates"))
    state_table = str(audit_cfg.get("state_table_name", "dc3s_online_state"))

    state_store = DC3SStateStore(duckdb_path=audit_path, table_name=state_table)
    try:
        state_key_target = "load_mw"
        state_row = state_store.get(zone_id=req.zone_id, device_id=req.device_id, target=state_key_target) or {}

        w_t, quality_flags = compute_reliability(
            req.telemetry_event or {},
            state_row.get("last_event"),
            expected_cadence_s=float(dc3s_cfg.get("expected_cadence_s", 3600)),
            reliability_cfg=dc3s_cfg.get("reliability", {}),
        )

        detector = PageHinkleyDetector.from_state(state_row.get("drift_state"), cfg=dc3s_cfg.get("drift", {}))
        drift_info: Dict[str, Any] = {
            "drift": False,
            "score": 0.0,
            "count": detector.count,
            "cooldown_remaining": detector.cooldown_remaining,
            "mean_residual": detector.mean,
        }
        if req.last_actual_load_mw is not None and req.last_pred_load_mw is not None:
            residual = abs(float(req.last_actual_load_mw) - float(req.last_pred_load_mw))
            drift_info = detector.update(residual)
            drift_info["residual_magnitude"] = float(residual)

        forecast_cfg = _load_forecast_cfg()
        features_df = _load_features_df(forecast_cfg)
        opt_cfg = _load_optimize_cfg()

        load_yhat, load_model_path = _predict_target(
            target="load_mw",
            horizon=req.horizon,
            features_df=features_df,
            forecast_cfg=forecast_cfg,
            required=True,
        )
        wind_yhat, wind_model_path = _predict_target(
            target="wind_mw",
            horizon=req.horizon,
            features_df=features_df,
            forecast_cfg=forecast_cfg,
            required=False,
        )
        solar_yhat, solar_model_path = _predict_target(
            target="solar_mw",
            horizon=req.horizon,
            features_df=features_df,
            forecast_cfg=forecast_cfg,
            required=False,
        )
        renewables = wind_yhat + solar_yhat

        q = _resolve_conformal_q(target="load_mw", horizon=req.horizon)
        lower, upper, uncertainty_meta = build_uncertainty_set(
            yhat=load_yhat,
            q=q,
            w_t=w_t,
            drift_flag=bool(drift_info.get("drift", False)),
            cfg=dc3s_cfg,
            prev_inflation=state_row.get("last_inflation"),
        )

        price_series, carbon_series = _build_default_price_and_carbon(features_df, req.horizon, opt_cfg)

        robust_cfg = _build_robust_config(opt_cfg)
        if req.controller == "deterministic":
            dispatch_plan = optimize_dispatch(
                load_yhat.tolist(),
                renewables.tolist(),
                opt_cfg,
                forecast_price=price_series.tolist(),
                forecast_carbon_kg=carbon_series.tolist(),
            )
        else:
            dispatch_plan = optimize_robust_dispatch(
                load_lower_bound=lower.tolist(),
                load_upper_bound=upper.tolist(),
                renewables_forecast=renewables.tolist(),
                price=price_series.tolist(),
                config=robust_cfg,
                verbose=False,
            )

        proposed_action = _first_action(dispatch_plan)

        bms_cfg = get_bms_config()
        bms = SafetyLayer(
            capacity_mwh=bms_cfg["capacity_mwh"],
            max_power_mw=bms_cfg["max_power_mw"],
            min_soc_pct=bms_cfg["min_soc_pct"],
            max_soc_pct=bms_cfg["max_soc_pct"],
        )

        battery_cfg = opt_cfg.get("battery", {}) if isinstance(opt_cfg, dict) else {}
        capacity_opt = float(battery_cfg.get("capacity_mwh", 10.0))
        max_power_opt = float(battery_cfg.get("max_power_mw", battery_cfg.get("max_charge_mw", 5.0)))
        max_charge_opt = float(battery_cfg.get("max_charge_mw", max_power_opt))
        max_discharge_opt = float(battery_cfg.get("max_discharge_mw", max_power_opt))
        eff = float(battery_cfg.get("efficiency", battery_cfg.get("efficiency_regime_a", 0.95)))
        min_soc_opt = float(battery_cfg.get("min_soc_mwh", 0.0))
        max_soc_opt = float(battery_cfg.get("max_soc_mwh", capacity_opt))

        bms_capacity = float(bms_cfg["capacity_mwh"])
        bms_min_soc = float(bms_cfg["min_soc_pct"]) * bms_capacity
        bms_max_soc = float(bms_cfg["max_soc_pct"]) * bms_capacity
        bms_max_power = float(bms_cfg["max_power_mw"])

        capacity = min(capacity_opt, bms_capacity)
        max_power = min(max_power_opt, bms_max_power)
        max_charge = min(max_charge_opt, bms_max_power)
        max_discharge = min(max_discharge_opt, bms_max_power)
        min_soc = max(min_soc_opt, bms_min_soc)
        max_soc = min(max_soc_opt, bms_max_soc)
        last_action = state_row.get("last_action") or {}
        last_net = float(last_action.get("discharge_mw", 0.0)) - float(last_action.get("charge_mw", 0.0))

        uncertainty_set = {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "meta": uncertainty_meta,
            "renewables_forecast": renewables.tolist(),
            "price": price_series.tolist(),
        }
        constraints = {
            "capacity_mwh": capacity,
            "min_soc_mwh": min_soc,
            "max_soc_mwh": max_soc,
            "max_power_mw": max_power,
            "max_charge_mw": max_charge,
            "max_discharge_mw": max_discharge,
            "ramp_mw": float(dc3s_cfg.get("shield", {}).get("max_ramp_mw", 0.0) or 0.0),
            "last_net_mw": last_net,
            "charge_efficiency": eff,
            "discharge_efficiency": eff,
            "current_soc_mwh": float(req.current_soc_mwh),
            "robust_config": robust_cfg,
            "degradation_cost_per_mwh": float(battery_cfg.get("degradation_cost_per_mwh", 10.0)),
            "max_grid_import_mw": float(opt_cfg.get("grid", {}).get("max_import_mw", 500.0)),
            "default_price_per_mwh": float(opt_cfg.get("grid", {}).get("price_per_mwh", 60.0)),
            "risk_weight_worst_case": float(opt_cfg.get("robust", {}).get("risk_weight_worst_case", 1.0)),
        }
        safe_action, repair_meta = repair_action(
            a_star=proposed_action,
            state={"current_soc_mwh": float(req.current_soc_mwh)},
            uncertainty_set=uncertainty_set,
            constraints=constraints,
            cfg=dc3s_cfg,
        )

        try:
            bms.validate_dispatch(
                current_soc=float(req.current_soc_mwh),
                charge_mw=float(safe_action["charge_mw"]),
                discharge_mw=float(safe_action["discharge_mw"]),
            )
        except SafetyViolation as exc:
            raise HTTPException(status_code=400, detail=f"DC3S safe action violates BMS: {exc}") from exc

        command_id = str(uuid4())
        model_hash = compute_model_hash([p for p in [load_model_path, wind_model_path, solar_model_path] if p is not None])
        dc3s_cfg_path = Path("configs/dc3s.yaml")
        config_bytes = dc3s_cfg_path.read_bytes() if dc3s_cfg_path.exists() else json.dumps(cfg_all, sort_keys=True).encode("utf-8")
        config_hash = compute_config_hash(config_bytes)

        certificate = make_certificate(
            command_id=command_id,
            device_id=req.device_id,
            zone_id=req.zone_id,
            controller=req.controller,
            proposed_action=proposed_action,
            safe_action=safe_action,
            uncertainty={
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "meta": uncertainty_meta,
                "shield_repair": repair_meta,
            },
            reliability={"w_t": float(w_t), "flags": quality_flags},
            drift=drift_info,
            model_hash=model_hash,
            config_hash=config_hash,
            prev_hash=state_row.get("last_prev_hash"),
            dispatch_plan=dispatch_plan,
        )
        store_certificate(certificate, duckdb_path=audit_path, table_name=audit_table)

        state_store.upsert(
            zone_id=req.zone_id,
            device_id=req.device_id,
            target=state_key_target,
            last_timestamp=_extract_ts(req.telemetry_event),
            last_yhat=float(load_yhat[0]),
            last_y_true=float(req.last_actual_load_mw) if req.last_actual_load_mw is not None else state_row.get("last_y_true"),
            drift_state=detector.to_state(),
            adaptive_state={},
            last_prev_hash=str(certificate.get("certificate_hash")),
            last_inflation=float(uncertainty_meta.get("inflation", 1.0)),
            last_event=req.telemetry_event,
            last_action=safe_action,
        )

        return DC3SStepResponse(
            proposed_action=DispatchAction(**proposed_action),
            safe_action=DispatchAction(**safe_action),
            dispatch_plan=dispatch_plan,
            uncertainty=UncertaintyPayload(
                lower=lower.astype(float).tolist(),
                upper=upper.astype(float).tolist(),
                meta=uncertainty_meta,
            ),
            certificate_id=command_id,
            command_id=command_id,
            certificate=certificate if req.include_certificate else None,
        )
    finally:
        state_store.close()


@router.get("/audit/{command_id}")
def dc3s_audit(command_id: str) -> Dict[str, Any]:
    cfg_all = _load_dc3s_cfg()
    dc3s_cfg = cfg_all["dc3s"]
    audit_cfg = dc3s_cfg.get("audit", {})
    cert = get_certificate(
        command_id=command_id,
        duckdb_path=str(audit_cfg.get("duckdb_path", "data/audit/dc3s_audit.duckdb")),
        table_name=str(audit_cfg.get("table_name", "dispatch_certificates")),
    )
    if cert is None:
        raise HTTPException(status_code=404, detail=f"No certificate found for command_id={command_id}")
    return cert
