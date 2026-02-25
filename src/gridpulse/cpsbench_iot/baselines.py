"""Controller adapters used by CPSBench-IoT evaluation runs."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from gridpulse.dc3s.calibration import build_uncertainty_set
from gridpulse.dc3s.certificate import compute_config_hash, compute_model_hash, make_certificate
from gridpulse.dc3s.drift import PageHinkleyDetector
from gridpulse.dc3s.guarantee_checks import evaluate_guarantee_checks
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


def _f(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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
    charge_eff = float(battery.get("charge_efficiency", efficiency))
    discharge_eff = float(battery.get("discharge_efficiency", efficiency))
    dt_hours = float(cfg.get("time_step_hours", 1.0))
    return {
        "capacity_mwh": capacity,
        "max_power_mw": max_power,
        "max_charge_mw": max_charge,
        "max_discharge_mw": max_discharge,
        "min_soc_mwh": min_soc,
        "max_soc_mwh": max_soc,
        "initial_soc_mwh": initial_soc,
        "efficiency": efficiency,
        "charge_efficiency": charge_eff,
        "discharge_efficiency": discharge_eff,
        "time_step_hours": dt_hours,
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


def soc_step_unclamped(
    *,
    current_soc_mwh: float,
    charge_mw: float,
    discharge_mw: float,
    constraints: Mapping[str, Any],
) -> float:
    """Unclamped one-step plant SOC update."""
    dt = max(_f(constraints.get("time_step_hours"), 1.0), 1e-9)
    eta_c = max(_f(constraints.get("charge_efficiency"), _f(constraints.get("efficiency"), 1.0)), 1e-6)
    eta_d = max(_f(constraints.get("discharge_efficiency"), _f(constraints.get("efficiency"), 1.0)), 1e-6)
    return float(current_soc_mwh + dt * (eta_c * max(charge_mw, 0.0) - (max(discharge_mw, 0.0) / eta_d)))


def project_action_observed_soc(
    *,
    action: Mapping[str, Any],
    observed_soc_mwh: float,
    constraints: Mapping[str, Any],
) -> dict[str, float]:
    """Clip an action against observed SOC and power bounds."""
    max_power = _f(constraints.get("max_power_mw"), 0.0)
    max_charge = _f(constraints.get("max_charge_mw"), max_power)
    max_discharge = _f(constraints.get("max_discharge_mw"), max_power)
    min_soc = _f(constraints.get("min_soc_mwh"), 0.0)
    max_soc = _f(constraints.get("max_soc_mwh"), _f(constraints.get("capacity_mwh"), observed_soc_mwh))
    dt = max(_f(constraints.get("time_step_hours"), 1.0), 1e-9)
    eta_c = max(_f(constraints.get("charge_efficiency"), _f(constraints.get("efficiency"), 1.0)), 1e-6)
    eta_d = max(_f(constraints.get("discharge_efficiency"), _f(constraints.get("efficiency"), 1.0)), 1e-6)

    charge = max(0.0, _f(action.get("charge_mw"), 0.0))
    discharge = max(0.0, _f(action.get("discharge_mw"), 0.0))
    if charge > 0.0 and discharge > 0.0:
        if discharge >= charge:
            charge = 0.0
        else:
            discharge = 0.0

    charge = min(charge, max_charge, max_power)
    discharge = min(discharge, max_discharge, max_power)

    max_feasible_charge = max(0.0, (max_soc - observed_soc_mwh) / (dt * eta_c))
    max_feasible_discharge = max(0.0, (observed_soc_mwh - min_soc) * eta_d / dt)
    charge = min(charge, max_feasible_charge)
    discharge = min(discharge, max_feasible_discharge)

    return {"charge_mw": float(charge), "discharge_mw": float(discharge)}


@dataclass
class DC3SLoopState:
    detector: PageHinkleyDetector
    prev_event: Mapping[str, Any] | None = None
    prev_inflation: float | None = None
    prev_hash: str | None = None


def init_dc3s_loop_state(dc3s_cfg: Mapping[str, Any]) -> DC3SLoopState:
    return DC3SLoopState(detector=PageHinkleyDetector.from_state(None, cfg=dc3s_cfg.get("drift", {})))


def deterministic_lp_step(
    *,
    load_obs_window: np.ndarray,
    renew_obs_window: np.ndarray,
    price_window: np.ndarray,
    carbon_window: np.ndarray,
    observed_soc_mwh: float,
    optimization_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Single-step deterministic policy from receding-horizon LP solve."""
    cfg = dict(optimization_cfg or _load_optimization_cfg())
    battery = dict(cfg.get("battery", {}))
    battery["initial_soc_mwh"] = float(observed_soc_mwh)
    cfg["battery"] = battery
    constraints = _battery_constraints(cfg)

    try:
        plan = optimize_dispatch(
            forecast_load=load_obs_window.tolist(),
            forecast_renewables=renew_obs_window.tolist(),
            config=cfg,
            forecast_price=price_window.tolist(),
            forecast_carbon_kg=carbon_window.tolist(),
        )
    except Exception:
        plan = {
            "battery_charge_mw": [0.0] * len(load_obs_window),
            "battery_discharge_mw": [0.0] * len(load_obs_window),
            "expected_cost_usd": None,
            "carbon_kg": None,
        }

    proposed = {
        "charge_mw": float((plan.get("battery_charge_mw") or [0.0])[0]),
        "discharge_mw": float((plan.get("battery_discharge_mw") or [0.0])[0]),
    }
    safe = dict(proposed)
    lower, upper = _default_interval(load_obs_window)
    return {
        "policy": "deterministic_lp",
        "proposed_action": proposed,
        "safe_action": safe,
        "dispatch_plan": plan,
        "interval_lower_t": float(lower[0]),
        "interval_upper_t": float(upper[0]),
        "constraints": constraints,
        "expected_cost_usd_step": float(price_window[0] * max(0.0, load_obs_window[0] - renew_obs_window[0] - safe["discharge_mw"] + safe["charge_mw"])),
        "certificate": None,
    }


def robust_fixed_interval_step(
    *,
    load_obs_window: np.ndarray,
    renew_obs_window: np.ndarray,
    price_window: np.ndarray,
    observed_soc_mwh: float,
    optimization_cfg: Mapping[str, Any] | None = None,
    interval_width_fraction: float = 0.12,
) -> dict[str, Any]:
    """Single-step robust fixed-width policy."""
    cfg = dict(optimization_cfg or _load_optimization_cfg())
    constraints = _battery_constraints(cfg)
    width = np.maximum(50.0, interval_width_fraction * np.abs(load_obs_window))
    lower = np.maximum(0.0, load_obs_window - width)
    upper = load_obs_window + width
    robust_cfg = RobustDispatchConfig(
        battery_capacity_mwh=constraints["capacity_mwh"],
        battery_max_charge_mw=constraints["max_charge_mw"],
        battery_max_discharge_mw=constraints["max_discharge_mw"],
        battery_charge_efficiency=constraints["charge_efficiency"],
        battery_discharge_efficiency=constraints["discharge_efficiency"],
        battery_initial_soc_mwh=float(observed_soc_mwh),
        battery_min_soc_mwh=constraints["min_soc_mwh"],
        battery_max_soc_mwh=constraints["max_soc_mwh"],
        max_grid_import_mw=float(cfg.get("grid", {}).get("max_import_mw", 500.0)),
        default_price_per_mwh=float(np.mean(price_window)),
        degradation_cost_per_mwh=float(cfg.get("battery", {}).get("degradation_cost_per_mwh", 10.0)),
        risk_weight_worst_case=float(cfg.get("robust", {}).get("risk_weight_worst_case", 1.0)),
        time_step_hours=float(cfg.get("time_step_hours", 1.0)),
        solver_name=str(cfg.get("solver_name", "appsi_highs")),
    )
    try:
        robust = optimize_robust_dispatch(
            load_lower_bound=lower.tolist(),
            load_upper_bound=upper.tolist(),
            renewables_forecast=renew_obs_window.tolist(),
            price=price_window.tolist(),
            config=robust_cfg,
            verbose=False,
        )
    except Exception:
        robust = {"battery_charge_mw": [0.0], "battery_discharge_mw": [0.0], "total_cost": None}

    proposed = {
        "charge_mw": float((robust.get("battery_charge_mw") or [0.0])[0]),
        "discharge_mw": float((robust.get("battery_discharge_mw") or [0.0])[0]),
    }
    safe = dict(proposed)
    return {
        "policy": "robust_fixed_interval",
        "proposed_action": proposed,
        "safe_action": safe,
        "dispatch_plan": robust,
        "interval_lower_t": float(lower[0]),
        "interval_upper_t": float(upper[0]),
        "constraints": constraints,
        "expected_cost_usd_step": float(price_window[0] * max(0.0, load_obs_window[0] - renew_obs_window[0] - safe["discharge_mw"] + safe["charge_mw"])),
        "certificate": None,
    }


def naive_safe_clip_step(
    *,
    timestamp: pd.Timestamp,
    load_obs_t: float,
    renew_obs_t: float,
    price_t: float,
    carbon_t: float,
    observed_soc_mwh: float,
    optimization_cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rule-based baseline with local clipping against observed SOC only."""
    cfg = dict(optimization_cfg or _load_optimization_cfg())
    constraints = _battery_constraints(cfg)
    hour = int(pd.to_datetime(timestamp, utc=True).hour)
    proposed = {"charge_mw": 0.0, "discharge_mw": 0.0}
    if 0 <= hour <= 5:
        proposed["charge_mw"] = 0.60 * constraints["max_charge_mw"]
    elif 17 <= hour <= 21:
        proposed["discharge_mw"] = 0.60 * constraints["max_discharge_mw"]
    safe = project_action_observed_soc(action=proposed, observed_soc_mwh=observed_soc_mwh, constraints=constraints)
    lower, upper = _default_interval(np.asarray([float(load_obs_t)]))
    grid_import_t = max(0.0, float(load_obs_t - renew_obs_t - safe["discharge_mw"] + safe["charge_mw"]))
    deg_cost = float(cfg.get("battery", {}).get("degradation_cost_per_mwh", 10.0))
    dt = float(cfg.get("time_step_hours", 1.0))
    expected_cost_t = float(price_t * grid_import_t * dt + deg_cost * (safe["charge_mw"] + safe["discharge_mw"]) * dt)
    carbon_kg_t = float(carbon_t * grid_import_t * dt)
    return {
        "policy": "naive_safe_clip",
        "proposed_action": proposed,
        "safe_action": safe,
        "dispatch_plan": None,
        "interval_lower_t": float(lower[0]),
        "interval_upper_t": float(upper[0]),
        "constraints": constraints,
        "expected_cost_usd_step": expected_cost_t,
        "carbon_kg_step": carbon_kg_t,
        "certificate": None,
    }


def dc3s_wrapped_step(
    *,
    load_obs_window: np.ndarray,
    renew_obs_window: np.ndarray,
    load_true_t: float,
    telemetry_event: Mapping[str, Any],
    price_window: np.ndarray,
    observed_soc_mwh: float,
    dc3s_state: DC3SLoopState,
    command_id: str,
    optimization_cfg: Mapping[str, Any] | None = None,
    dc3s_cfg: Mapping[str, Any] | None = None,
    q_base: float | None = None,
) -> dict[str, Any]:
    """Single-step DC3S policy with uncertainty inflation and action repair."""
    opt_cfg = dict(optimization_cfg or _load_optimization_cfg())
    cfg = dict(dc3s_cfg or _load_dc3s_cfg())
    base = deterministic_lp_step(
        load_obs_window=load_obs_window,
        renew_obs_window=renew_obs_window,
        price_window=price_window,
        carbon_window=np.zeros_like(price_window),
        observed_soc_mwh=observed_soc_mwh,
        optimization_cfg=opt_cfg,
    )
    constraints = dict(base["constraints"])
    cadence = float(cfg.get("expected_cadence_s", 3600.0))
    event = dict(telemetry_event)
    if "ts_utc" not in event:
        event["ts_utc"] = str(event.get("timestamp", command_id))

    w_t, flags = compute_reliability(
        event,
        dc3s_state.prev_event,
        expected_cadence_s=cadence,
        reliability_cfg=cfg.get("reliability", {}),
    )
    residual = abs(float(load_true_t) - float(load_obs_window[0]))
    drift = dc3s_state.detector.update(residual)
    q_t = float(q_base if q_base is not None else max(50.0, residual))
    lo, hi, meta = build_uncertainty_set(
        yhat=np.asarray([load_obs_window[0]], dtype=float),
        q=np.asarray([q_t], dtype=float),
        w_t=float(w_t),
        drift_flag=bool(drift.get("drift", False)),
        cfg=cfg,
        prev_inflation=dc3s_state.prev_inflation,
    )
    dc3s_state.prev_inflation = float(meta.get("inflation", 1.0))

    proposed = dict(base["proposed_action"])
    repair_constraints = {
        **constraints,
        "current_soc_mwh": float(observed_soc_mwh),
        "last_net_mw": 0.0,
        "ramp_mw": float(cfg.get("shield", {}).get("max_ramp_mw", 0.0) or 0.0),
        "degradation_cost_per_mwh": float(opt_cfg.get("battery", {}).get("degradation_cost_per_mwh", 10.0)),
        "max_grid_import_mw": float(opt_cfg.get("grid", {}).get("max_import_mw", 500.0)),
        "default_price_per_mwh": float(np.mean(price_window)),
        "risk_weight_worst_case": float(opt_cfg.get("robust", {}).get("risk_weight_worst_case", 1.0)),
    }
    uncertainty_set = {
        "lower": [float(lo[0])],
        "upper": [float(hi[0])],
        "meta": meta,
        "renewables_forecast": [float(renew_obs_window[0])],
        "price": [float(price_window[0])],
    }
    safe, repair_meta = repair_action(
        a_star=proposed,
        state={"current_soc_mwh": float(observed_soc_mwh)},
        uncertainty_set=uncertainty_set,
        constraints=repair_constraints,
        cfg=cfg,
    )
    guarantee_ok, guarantee_reasons, _ = evaluate_guarantee_checks(
        current_soc=float(observed_soc_mwh),
        action=safe,
        constraints=repair_constraints,
    )

    config_hash = compute_config_hash(json.dumps(cfg, sort_keys=True).encode("utf-8"))
    model_hash = compute_model_hash([])
    certificate = make_certificate(
        command_id=command_id,
        device_id=str(event.get("device_id", "bench-device")),
        zone_id=str(event.get("zone_id", "DE")),
        controller="deterministic",
        proposed_action=proposed,
        safe_action=safe,
        uncertainty={"lower": [float(lo[0])], "upper": [float(hi[0])], "meta": meta, "shield_repair": repair_meta},
        reliability={"w_t": float(w_t), "flags": flags},
        drift=drift,
        model_hash=model_hash,
        config_hash=config_hash,
        prev_hash=dc3s_state.prev_hash,
        dispatch_plan=base.get("dispatch_plan"),
        intervened=bool(repair_meta.get("repaired", False)),
        intervention_reason=(repair_meta.get("robust_meta") or {}).get("reason") if isinstance(repair_meta.get("robust_meta"), dict) else "projection_clip",
        reliability_w=float(w_t),
        drift_flag=bool(drift.get("drift", False)),
        inflation=float(meta.get("inflation", 1.0)),
        guarantee_checks_passed=bool(guarantee_ok),
        guarantee_fail_reasons=guarantee_reasons,
        assumptions_version=str(cfg.get("assumptions_version", "dc3s-assumptions-v1")),
    )
    dc3s_state.prev_event = event
    dc3s_state.prev_hash = str(certificate.get("certificate_hash"))

    return {
        "policy": "dc3s_wrapped",
        "proposed_action": proposed,
        "safe_action": safe,
        "dispatch_plan": base.get("dispatch_plan"),
        "interval_lower_t": float(lo[0]),
        "interval_upper_t": float(hi[0]),
        "constraints": constraints,
        "expected_cost_usd_step": float(price_window[0] * max(0.0, load_obs_window[0] - renew_obs_window[0] - safe["discharge_mw"] + safe["charge_mw"])),
        "certificate": certificate,
        "reliability_w": float(w_t),
        "drift_flag": bool(drift.get("drift", False)),
        "inflation": float(meta.get("inflation", 1.0)),
        "guarantee_checks_passed": bool(guarantee_ok),
        "guarantee_fail_reasons": guarantee_reasons,
    }
