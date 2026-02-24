"""Controller adapters used by CPSBench-IoT evaluation runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from gridpulse.dc3s.calibration import build_uncertainty_set
from gridpulse.dc3s.certificate import compute_config_hash, compute_model_hash, make_certificate
from gridpulse.dc3s.drift import PageHinkleyDetector
from gridpulse.dc3s.quality import compute_reliability
from gridpulse.dc3s.shield import repair_action
from gridpulse.optimizer import optimize_dispatch
from gridpulse.optimizer.robust_dispatch import RobustDispatchConfig, optimize_robust_dispatch


def _as_array(values: Any, horizon: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Input array must be non-empty")
    if horizon is not None and arr.size == 1 and horizon > 1:
        arr = np.full(horizon, float(arr[0]), dtype=float)
    return arr


def _load_yaml(path: str) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _load_optimization_cfg() -> dict[str, Any]:
    for p in ("configs/optimization.yaml", "configs/optimize.yaml"):
        payload = _load_yaml(p)
        if payload:
            return payload
    return {}


def _load_dc3s_cfg() -> dict[str, Any]:
    payload = _load_yaml("configs/dc3s.yaml")
    dc3s = payload.get("dc3s", {}) if isinstance(payload, dict) else {}
    return dc3s if isinstance(dc3s, dict) else {}


def _battery_constraints(cfg: Mapping[str, Any]) -> dict[str, float]:
    battery = dict(cfg.get("battery", {}))
    capacity = float(battery.get("capacity_mwh", 100.0))
    max_power = float(battery.get("max_power_mw", 50.0))
    max_charge = float(battery.get("max_charge_mw", max_power))
    max_discharge = float(battery.get("max_discharge_mw", max_power))
    min_soc = float(battery.get("min_soc_mwh", 0.0))
    max_soc = float(battery.get("max_soc_mwh", capacity))
    initial_soc = float(battery.get("initial_soc_mwh", capacity * 0.5))
    efficiency = float(battery.get("efficiency", battery.get("efficiency_regime_a", 0.95)))
    return {
        "capacity_mwh": capacity,
        "max_power_mw": max_power,
        "max_charge_mw": max_charge,
        "max_discharge_mw": max_discharge,
        "min_soc_mwh": min_soc,
        "max_soc_mwh": max_soc,
        "initial_soc_mwh": initial_soc,
        "efficiency": efficiency,
    }


def _simulate_soc(
    *,
    charge: np.ndarray,
    discharge: np.ndarray,
    initial_soc: float,
    min_soc: float,
    max_soc: float,
    efficiency: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eff = max(float(efficiency), 1e-6)
    soc = np.zeros(len(charge), dtype=float)
    charge_safe = np.zeros(len(charge), dtype=float)
    discharge_safe = np.zeros(len(charge), dtype=float)
    current_soc = float(initial_soc)
    for i in range(len(charge)):
        c = max(0.0, float(charge[i]))
        d = max(0.0, float(discharge[i]))
        if c > 0.0 and d > 0.0:
            if d >= c:
                c = 0.0
            else:
                d = 0.0

        c = min(c, max(0.0, (max_soc - current_soc) / eff))
        d = min(d, max(0.0, (current_soc - min_soc) * eff))
        current_soc = current_soc + eff * c - d / eff
        current_soc = min(max_soc, max(min_soc, current_soc))

        charge_safe[i] = c
        discharge_safe[i] = d
        soc[i] = current_soc
    return charge_safe, discharge_safe, soc


def _default_interval(load_forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = np.maximum(75.0, 0.08 * np.abs(load_forecast))
    return np.maximum(0.0, load_forecast - q), load_forecast + q


def deterministic_lp_dispatch(
    *,
    load_forecast: np.ndarray,
    renewables_forecast: np.ndarray,
    price: np.ndarray,
    carbon: np.ndarray,
    optimization_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic baseline adapter using the existing LP optimizer."""
    cfg = dict(optimization_cfg or _load_optimization_cfg())
    constraints = _battery_constraints(cfg)
    try:
        dispatch = optimize_dispatch(
            forecast_load=load_forecast.tolist(),
            forecast_renewables=renewables_forecast.tolist(),
            config=cfg,
            forecast_price=price.tolist(),
            forecast_carbon_kg=carbon.tolist(),
        )
    except Exception:
        zeros = np.zeros_like(load_forecast)
        dispatch = {"battery_charge_mw": zeros.tolist(), "battery_discharge_mw": zeros.tolist(), "soc_mwh": []}

    proposed_charge = _as_array(dispatch.get("battery_charge_mw", np.zeros_like(load_forecast)), horizon=len(load_forecast))
    proposed_discharge = _as_array(dispatch.get("battery_discharge_mw", np.zeros_like(load_forecast)), horizon=len(load_forecast))
    if dispatch.get("soc_mwh"):
        soc = _as_array(dispatch.get("soc_mwh"), horizon=len(load_forecast))
        safe_charge = proposed_charge.copy()
        safe_discharge = proposed_discharge.copy()
    else:
        safe_charge, safe_discharge, soc = _simulate_soc(
            charge=proposed_charge,
            discharge=proposed_discharge,
            initial_soc=constraints["initial_soc_mwh"],
            min_soc=constraints["min_soc_mwh"],
            max_soc=constraints["max_soc_mwh"],
            efficiency=constraints["efficiency"],
        )

    lower, upper = _default_interval(load_forecast)
    return {
        "policy": "deterministic_lp",
        "dispatch_plan": dispatch,
        "proposed_charge_mw": proposed_charge,
        "proposed_discharge_mw": proposed_discharge,
        "safe_charge_mw": safe_charge,
        "safe_discharge_mw": safe_discharge,
        "soc_mwh": soc,
        "interval_lower": lower,
        "interval_upper": upper,
        "certificates": [None] * len(load_forecast),
        "constraints": constraints,
        "expected_cost_usd": float(dispatch.get("expected_cost_usd")) if dispatch.get("expected_cost_usd") is not None else None,
        "carbon_kg": float(dispatch.get("carbon_kg")) if dispatch.get("carbon_kg") is not None else None,
    }


def robust_fixed_interval_dispatch(
    *,
    load_forecast: np.ndarray,
    renewables_forecast: np.ndarray,
    price: np.ndarray,
    optimization_cfg: Mapping[str, Any] | None = None,
    interval_width_fraction: float = 0.12,
) -> dict[str, Any]:
    """Robust baseline adapter using fixed uncertainty bands around load forecast."""
    cfg = dict(optimization_cfg or _load_optimization_cfg())
    constraints = _battery_constraints(cfg)
    width = np.maximum(50.0, interval_width_fraction * np.abs(load_forecast))
    lower = np.maximum(0.0, load_forecast - width)
    upper = load_forecast + width
    robust_cfg = RobustDispatchConfig(
        battery_capacity_mwh=constraints["capacity_mwh"],
        battery_max_charge_mw=constraints["max_charge_mw"],
        battery_max_discharge_mw=constraints["max_discharge_mw"],
        battery_charge_efficiency=constraints["efficiency"],
        battery_discharge_efficiency=constraints["efficiency"],
        battery_initial_soc_mwh=constraints["initial_soc_mwh"],
        battery_min_soc_mwh=constraints["min_soc_mwh"],
        battery_max_soc_mwh=constraints["max_soc_mwh"],
        max_grid_import_mw=float(cfg.get("grid", {}).get("max_import_mw", 500.0)),
        default_price_per_mwh=float(np.mean(price)),
        degradation_cost_per_mwh=float(cfg.get("battery", {}).get("degradation_cost_per_mwh", 10.0)),
        risk_weight_worst_case=float(cfg.get("robust", {}).get("risk_weight_worst_case", 1.0)),
        solver_name=str(cfg.get("solver_name", "appsi_highs")),
    )
    try:
        robust = optimize_robust_dispatch(
            load_lower_bound=lower.tolist(),
            load_upper_bound=upper.tolist(),
            renewables_forecast=renewables_forecast.tolist(),
            price=price.tolist(),
            config=robust_cfg,
            verbose=False,
        )
    except Exception:
        robust = {"battery_charge_mw": np.zeros_like(load_forecast).tolist(), "battery_discharge_mw": np.zeros_like(load_forecast).tolist(), "feasible": False}

    proposed_charge = _as_array(robust.get("battery_charge_mw", np.zeros_like(load_forecast)), horizon=len(load_forecast))
    proposed_discharge = _as_array(robust.get("battery_discharge_mw", np.zeros_like(load_forecast)), horizon=len(load_forecast))
    safe_charge, safe_discharge, soc = _simulate_soc(
        charge=proposed_charge,
        discharge=proposed_discharge,
        initial_soc=constraints["initial_soc_mwh"],
        min_soc=constraints["min_soc_mwh"],
        max_soc=constraints["max_soc_mwh"],
        efficiency=constraints["efficiency"],
    )
    return {
        "policy": "robust_fixed_interval",
        "dispatch_plan": robust,
        "proposed_charge_mw": proposed_charge,
        "proposed_discharge_mw": proposed_discharge,
        "safe_charge_mw": safe_charge,
        "safe_discharge_mw": safe_discharge,
        "soc_mwh": soc,
        "interval_lower": lower,
        "interval_upper": upper,
        "certificates": [None] * len(load_forecast),
        "constraints": constraints,
        "expected_cost_usd": float(robust.get("total_cost")) if robust.get("total_cost") is not None else None,
        "carbon_kg": None,
    }


def naive_safe_clip_dispatch(
    *,
    load_forecast: np.ndarray,
    renewables_forecast: np.ndarray,
    price: np.ndarray,
    carbon: np.ndarray,
    timestamps: pd.Series,
    optimization_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Naive battery baseline with hour-rule actions and strict safety clipping."""
    cfg = dict(optimization_cfg or _load_optimization_cfg())
    constraints = _battery_constraints(cfg)
    max_charge = constraints["max_charge_mw"]
    max_discharge = constraints["max_discharge_mw"]
    deg_cost = float(cfg.get("battery", {}).get("degradation_cost_per_mwh", 10.0))

    proposed_charge = np.zeros(len(load_forecast), dtype=float)
    proposed_discharge = np.zeros(len(load_forecast), dtype=float)
    hours = pd.to_datetime(timestamps, utc=True).dt.hour.to_numpy(dtype=int)
    for i, hr in enumerate(hours):
        if 0 <= hr <= 5:
            proposed_charge[i] = 0.60 * max_charge
        elif 17 <= hr <= 21:
            proposed_discharge[i] = 0.60 * max_discharge

    safe_charge, safe_discharge, soc = _simulate_soc(
        charge=proposed_charge,
        discharge=proposed_discharge,
        initial_soc=constraints["initial_soc_mwh"],
        min_soc=constraints["min_soc_mwh"],
        max_soc=constraints["max_soc_mwh"],
        efficiency=constraints["efficiency"],
    )
    grid_import = np.maximum(0.0, load_forecast - renewables_forecast - safe_discharge + safe_charge)
    expected_cost = float(np.sum(price * grid_import + deg_cost * (safe_charge + safe_discharge)))
    carbon_kg = float(np.sum(carbon * grid_import))
    lower, upper = _default_interval(load_forecast)
    return {
        "policy": "naive_safe_clip",
        "dispatch_plan": {
            "battery_charge_mw": safe_charge.tolist(),
            "battery_discharge_mw": safe_discharge.tolist(),
            "soc_mwh": soc.tolist(),
            "grid_mw": grid_import.tolist(),
            "expected_cost_usd": expected_cost,
            "carbon_kg": carbon_kg,
        },
        "proposed_charge_mw": proposed_charge,
        "proposed_discharge_mw": proposed_discharge,
        "safe_charge_mw": safe_charge,
        "safe_discharge_mw": safe_discharge,
        "soc_mwh": soc,
        "interval_lower": lower,
        "interval_upper": upper,
        "certificates": [None] * len(load_forecast),
        "constraints": constraints,
        "expected_cost_usd": expected_cost,
        "carbon_kg": carbon_kg,
    }


def dc3s_wrapped_dispatch(
    *,
    load_forecast: np.ndarray,
    renewables_forecast: np.ndarray,
    load_true: np.ndarray,
    telemetry_events: list[Mapping[str, Any]],
    price: np.ndarray,
    optimization_cfg: Mapping[str, Any] | None = None,
    command_prefix: str = "cpsbench",
) -> dict[str, Any]:
    """Wrap deterministic controller actions with DC3S uncertainty and shield repair."""
    opt_cfg = dict(optimization_cfg or _load_optimization_cfg())
    dc3s_cfg = _load_dc3s_cfg()
    base = deterministic_lp_dispatch(
        load_forecast=load_forecast,
        renewables_forecast=renewables_forecast,
        price=price,
        carbon=np.zeros_like(price),
        optimization_cfg=opt_cfg,
    )
    constraints = dict(base["constraints"])
    cadence = float(dc3s_cfg.get("expected_cadence_s", 3600.0))

    detector = PageHinkleyDetector.from_state(None, cfg=dc3s_cfg.get("drift", {}))
    config_hash = compute_config_hash(json.dumps(dc3s_cfg, sort_keys=True).encode("utf-8"))
    model_hash = compute_model_hash([])

    proposed_charge = _as_array(base["proposed_charge_mw"], horizon=len(load_forecast))
    proposed_discharge = _as_array(base["proposed_discharge_mw"], horizon=len(load_forecast))
    safe_charge = np.zeros(len(load_forecast), dtype=float)
    safe_discharge = np.zeros(len(load_forecast), dtype=float)
    soc = np.zeros(len(load_forecast), dtype=float)
    lower = np.zeros(len(load_forecast), dtype=float)
    upper = np.zeros(len(load_forecast), dtype=float)
    interventions = np.zeros(len(load_forecast), dtype=bool)
    certificates: list[dict[str, Any]] = []

    q = max(50.0, float(np.quantile(np.abs(load_true - load_forecast), 0.90)))
    current_soc = float(constraints["initial_soc_mwh"])
    prev_event: Mapping[str, Any] | None = None
    prev_inflation: float | None = None

    for t in range(len(load_forecast)):
        event = dict(telemetry_events[t])
        if "ts_utc" not in event:
            event["ts_utc"] = str(event.get("timestamp", t))

        w_t, flags = compute_reliability(
            event,
            prev_event,
            expected_cadence_s=cadence,
            reliability_cfg=dc3s_cfg.get("reliability", {}),
        )
        residual = abs(float(load_true[t]) - float(load_forecast[t]))
        drift = detector.update(residual)

        lo_t, hi_t, meta = build_uncertainty_set(
            yhat=np.asarray([load_forecast[t]], dtype=float),
            q=np.asarray([q], dtype=float),
            w_t=w_t,
            drift_flag=bool(drift.get("drift", False)),
            cfg=dc3s_cfg,
            prev_inflation=prev_inflation,
        )
        prev_inflation = float(meta.get("inflation", 1.0))
        lower[t] = float(lo_t[0])
        upper[t] = float(hi_t[0])

        a_star = {"charge_mw": float(proposed_charge[t]), "discharge_mw": float(proposed_discharge[t])}
        repair_constraints = {
            **constraints,
            "current_soc_mwh": current_soc,
            "last_net_mw": float(safe_discharge[t - 1] - safe_charge[t - 1]) if t > 0 else 0.0,
            "ramp_mw": float(dc3s_cfg.get("shield", {}).get("max_ramp_mw", 0.0) or 0.0),
            "degradation_cost_per_mwh": float(opt_cfg.get("battery", {}).get("degradation_cost_per_mwh", 10.0)),
            "max_grid_import_mw": float(opt_cfg.get("grid", {}).get("max_import_mw", 500.0)),
            "default_price_per_mwh": float(np.mean(price)),
            "risk_weight_worst_case": float(opt_cfg.get("robust", {}).get("risk_weight_worst_case", 1.0)),
        }
        uncertainty_set = {
            "lower": [float(lo_t[0])],
            "upper": [float(hi_t[0])],
            "meta": meta,
            "renewables_forecast": [float(renewables_forecast[t])],
            "price": [float(price[t])],
        }
        safe_action, repair_meta = repair_action(
            a_star=a_star,
            state={"current_soc_mwh": current_soc},
            uncertainty_set=uncertainty_set,
            constraints=repair_constraints,
            cfg=dc3s_cfg,
        )

        safe_charge[t] = float(safe_action["charge_mw"])
        safe_discharge[t] = float(safe_action["discharge_mw"])
        interventions[t] = (
            abs(safe_charge[t] - proposed_charge[t]) > 1e-6 or abs(safe_discharge[t] - proposed_discharge[t]) > 1e-6
        )
        eff = max(float(constraints["efficiency"]), 1e-6)
        current_soc = current_soc + eff * safe_charge[t] - safe_discharge[t] / eff
        current_soc = min(float(constraints["max_soc_mwh"]), max(float(constraints["min_soc_mwh"]), current_soc))
        soc[t] = current_soc

        cert = make_certificate(
            command_id=f"{command_prefix}-{t:04d}",
            device_id=str(event.get("device_id", "bench-device")),
            zone_id=str(event.get("zone_id", "DE")),
            controller="deterministic",
            proposed_action=a_star,
            safe_action=safe_action,
            uncertainty={
                "lower": [float(lo_t[0])],
                "upper": [float(hi_t[0])],
                "meta": meta,
                "shield_repair": repair_meta,
            },
            reliability={"w_t": float(w_t), "flags": flags},
            drift=drift,
            model_hash=model_hash,
            config_hash=config_hash,
            prev_hash=certificates[-1]["certificate_hash"] if certificates else None,
            dispatch_plan=None,
        )
        certificates.append(cert)
        prev_event = event

    grid_import = np.maximum(0.0, load_forecast - renewables_forecast - safe_discharge + safe_charge)
    expected_cost = float(np.sum(price * grid_import))
    return {
        "policy": "dc3s_wrapped",
        "dispatch_plan": {
            "battery_charge_mw": safe_charge.tolist(),
            "battery_discharge_mw": safe_discharge.tolist(),
            "soc_mwh": soc.tolist(),
            "grid_mw": grid_import.tolist(),
            "expected_cost_usd": expected_cost,
            "interventions": interventions.astype(int).tolist(),
        },
        "proposed_charge_mw": proposed_charge,
        "proposed_discharge_mw": proposed_discharge,
        "safe_charge_mw": safe_charge,
        "safe_discharge_mw": safe_discharge,
        "soc_mwh": soc,
        "interval_lower": lower,
        "interval_upper": upper,
        "certificates": certificates,
        "constraints": constraints,
        "expected_cost_usd": expected_cost,
        "carbon_kg": None,
    }
