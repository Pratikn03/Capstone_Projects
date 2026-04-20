"""Controller adapters used by CPSBench-IoT evaluation runs."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from orius.dc3s.calibration import build_uncertainty_set, build_uncertainty_set_kappa
from orius.dc3s.certificate import compute_config_hash, compute_model_hash, make_certificate
from orius.dc3s.drift import PageHinkleyDetector
from orius.dc3s.ftit import update as update_ftit_state
from orius.dc3s.guarantee_checks import evaluate_guarantee_checks
from orius.dc3s.quality import compute_reliability
from orius.dc3s.shield import repair_action
from orius.optimizer import optimize_dispatch
from orius.optimizer.scenario_robust_dispatch import optimize_scenario_robust_dispatch
from orius.optimizer.robust_dispatch import RobustDispatchConfig, optimize_robust_dispatch


def _as_array(values: Any, horizon: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Input array must be non-empty")
    if horizon is not None and arr.size == 1 and horizon > 1:
        arr = np.full(horizon, float(arr[0]), dtype=float)
    return arr


def _f(value: Any, default: float) -> float:
    try:
        v = float(value)
        if not math.isfinite(v):
            return float(default)
        return v
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


def _resolve_dc3s_cfg_path() -> Path:
    env_path = os.getenv("ORIUS_DC3S_CONFIG", "").strip()
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
    return Path("configs/dc3s.yaml")


def _load_dc3s_cfg() -> dict[str, Any]:
    payload = _load_yaml(str(_resolve_dc3s_cfg_path()))
    dc3s = payload.get("dc3s", {}) if isinstance(payload, dict) else {}
    return dc3s if isinstance(dc3s, dict) else {}


_DC3S_DISPATCH_VARIANT_TO_POLICY = {
    "no_wt": "dc3s_no_wt",
    "no_drift": "dc3s_no_drift",
    "linear": "dc3s_linear",
    "kappa": "dc3s_kappa",
}


def _normalize_dc3s_dispatch_variant(variant: str) -> str:
    variant_norm = str(variant).strip().lower()
    if variant_norm not in _DC3S_DISPATCH_VARIANT_TO_POLICY:
        raise ValueError(f"Unsupported DC3S dispatch variant: {variant}")
    return variant_norm


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


def scenario_robust_dispatch(
    *,
    load_forecast: np.ndarray,
    renewables_forecast: np.ndarray,
    load_true: np.ndarray,
    price: np.ndarray,
    optimization_cfg: Mapping[str, Any] | None = None,
    seed: int = 0,
    n_scenarios: int = 30,
    scenario_scale: float = 1.0,
    risk_weight_worst_case: float | None = None,
) -> dict[str, Any]:
    """Robust baseline adapter using sampled load scenarios."""
    load_forecast_arr = _as_array(load_forecast)
    horizon = len(load_forecast_arr)
    renewables_forecast_arr = _as_array(renewables_forecast, horizon=horizon)
    load_true_arr = _as_array(load_true, horizon=horizon)
    price_arr = _as_array(price, horizon=horizon)
    if horizon == 0:
        raise ValueError("scenario_robust_dispatch requires a non-empty horizon")
    if int(n_scenarios) < 1:
        raise ValueError("n_scenarios must be >= 1")
    if float(scenario_scale) < 0.0:
        raise ValueError("scenario_scale must be >= 0")

    cfg = dict(optimization_cfg or _load_optimization_cfg())
    constraints = _battery_constraints(cfg)
    risk_weight = (
        float(risk_weight_worst_case)
        if risk_weight_worst_case is not None
        else float(cfg.get("robust", {}).get("risk_weight_worst_case", 1.0))
    )

    sigma = float(np.std(load_true_arr - load_forecast_arr))
    rng = np.random.default_rng(int(seed))
    eps = rng.normal(
        loc=0.0,
        scale=float(scenario_scale) * sigma,
        size=(int(n_scenarios), horizon),
    )
    scenarios = np.clip(load_forecast_arr[None, :] + eps, a_min=0.0, a_max=None)
    interval_lower = scenarios.min(axis=0)
    interval_upper = scenarios.max(axis=0)

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
        default_price_per_mwh=float(np.mean(price_arr)),
        degradation_cost_per_mwh=float(cfg.get("battery", {}).get("degradation_cost_per_mwh", 10.0)),
        risk_weight_worst_case=risk_weight,
        time_step_hours=float(cfg.get("time_step_hours", 1.0)),
        solver_name=str(cfg.get("solver_name", "appsi_highs")),
    )
    try:
        robust = optimize_scenario_robust_dispatch(
            load_scenarios=scenarios,
            renewables_forecast=renewables_forecast_arr.tolist(),
            price=price_arr.tolist(),
            config=robust_cfg,
            verbose=False,
        )
    except Exception:
        robust = {
            "battery_charge_mw": np.zeros_like(load_forecast_arr).tolist(),
            "battery_discharge_mw": np.zeros_like(load_forecast_arr).tolist(),
            "total_cost": None,
            "feasible": False,
            "solver_status": "exception",
            "scenario_costs": [],
            "binding_scenario": None,
        }

    proposed_charge = _as_array(robust.get("battery_charge_mw", np.zeros_like(load_forecast_arr)), horizon=horizon)
    proposed_discharge = _as_array(robust.get("battery_discharge_mw", np.zeros_like(load_forecast_arr)), horizon=horizon)
    safe_charge, safe_discharge, soc = _simulate_soc(
        charge=proposed_charge,
        discharge=proposed_discharge,
        initial_soc=constraints["initial_soc_mwh"],
        min_soc=constraints["min_soc_mwh"],
        max_soc=constraints["max_soc_mwh"],
        efficiency=constraints["efficiency"],
    )
    grid_import = np.maximum(0.0, load_forecast_arr - renewables_forecast_arr - safe_discharge + safe_charge)
    expected_cost = float(np.sum(price_arr * grid_import))
    return {
        "policy": "scenario_robust",
        "dispatch_plan": robust,
        "proposed_charge_mw": proposed_charge,
        "proposed_discharge_mw": proposed_discharge,
        "safe_charge_mw": safe_charge,
        "safe_discharge_mw": safe_discharge,
        "soc_mwh": soc,
        "interval_lower": interval_lower,
        "interval_upper": interval_upper,
        "certificates": [None] * horizon,
        "constraints": constraints,
        "expected_cost_usd": expected_cost,
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


def _dc3s_dispatch_impl(
    *,
    load_forecast: np.ndarray,
    renewables_forecast: np.ndarray,
    load_true: np.ndarray,
    telemetry_events: list[Mapping[str, Any]],
    price: np.ndarray,
    optimization_cfg: Mapping[str, Any] | None = None,
    command_prefix: str = "cpsbench",
    policy_name: str,
    law_override: str | None,
    variant: str = "linear",
) -> dict[str, Any]:
    """Wrap deterministic controller actions with DC3S uncertainty and shield repair."""
    variant = _normalize_dc3s_dispatch_variant(variant)
    opt_cfg = dict(optimization_cfg or _load_optimization_cfg())
    dc3s_cfg = _load_dc3s_cfg()
    if law_override is not None:
        dc3s_cfg["law"] = str(law_override)
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
    sigma_sq = float(np.var(np.abs(load_true - load_forecast)))
    current_soc = float(constraints["initial_soc_mwh"])
    prev_event: Mapping[str, Any] | None = None
    prev_inflation: float | None = None
    adaptive_state: dict[str, Any] = {}

    for t in range(len(load_forecast)):
        event = dict(telemetry_events[t])
        if "ts_utc" not in event:
            event["ts_utc"] = str(event.get("timestamp", t))
        ftit_cfg = {**dict(dc3s_cfg.get("ftit", {})), "law": str(dc3s_cfg.get("law", "linear")).strip().lower()}

        w_t, flags = compute_reliability(
            event,
            prev_event,
            expected_cadence_s=cadence,
            reliability_cfg=dc3s_cfg.get("reliability", {}),
            adaptive_state=adaptive_state,
            ftit_cfg=ftit_cfg,
        )
        residual = abs(float(load_true[t]) - float(load_forecast[t]))
        if variant == "no_drift":
            drift = {"drift": False}
        else:
            drift = detector.update(residual)
        ftit_state = update_ftit_state(
            adaptive_state=adaptive_state,
            fault_flags=flags.get("fault_flags", {}),
            constraints={
                **constraints,
                "current_soc_mwh": current_soc,
            },
            cfg=ftit_cfg,
            stale_tracker=flags.get("stale_tracker"),
            sigma2_observation=float(residual) ** 2,
        )
        adaptive_state = ftit_state["adaptive_state"]
        cfg_runtime = dict(dc3s_cfg)
        if str(cfg_runtime.get("law", "linear")).strip().lower() == "ftit_ro":
            cfg_runtime["ftit_runtime"] = {"sigma2": float(ftit_state["sigma2"])}
        w_used_for_inflation = 1.0 if variant == "no_wt" else float(w_t)

        if variant == "kappa":
            lo_t, hi_t, meta = build_uncertainty_set_kappa(
                yhat=np.asarray([load_forecast[t]], dtype=float),
                q=np.asarray([q], dtype=float),
                w_t=w_used_for_inflation,
                drift_flag=bool(drift.get("drift", False)),
                cfg=cfg_runtime,
                sigma_sq=sigma_sq,
                prev_inflation=prev_inflation,
            )
        else:
            lo_t, hi_t, meta = build_uncertainty_set(
                yhat=np.asarray([load_forecast[t]], dtype=float),
                q=np.asarray([q], dtype=float),
                w_t=w_used_for_inflation,
                drift_flag=bool(drift.get("drift", False)),
                cfg=cfg_runtime,
                prev_inflation=prev_inflation,
            )
        prev_inflation = float(meta.get("inflation", 1.0))
        meta["gamma_mw"] = float(ftit_state["gamma_mw"])
        meta["e_t_mwh"] = float(ftit_state["e_t_mwh"])
        meta["soc_tube_lower_mwh"] = float(ftit_state["soc_tube_lower_mwh"])
        meta["soc_tube_upper_mwh"] = float(ftit_state["soc_tube_upper_mwh"])
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
        if str(cfg_runtime.get("law", "linear")).strip().lower() == "ftit_ro":
            repair_constraints["ftit_soc_min_mwh"] = float(ftit_state["soc_tube_lower_mwh"])
            repair_constraints["ftit_soc_max_mwh"] = float(ftit_state["soc_tube_upper_mwh"])
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
            cfg=cfg_runtime,
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
            controller=policy_name,
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
            gamma_mw=float(ftit_state["gamma_mw"]),
            e_t_mwh=float(ftit_state["e_t_mwh"]),
            soc_tube_lower_mwh=float(ftit_state["soc_tube_lower_mwh"]),
            soc_tube_upper_mwh=float(ftit_state["soc_tube_upper_mwh"]),
        )
        certificates.append(cert)
        prev_event = event

    grid_import = np.maximum(0.0, load_forecast - renewables_forecast - safe_discharge + safe_charge)
    expected_cost = float(np.sum(price * grid_import))
    return {
        "policy": policy_name,
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


def dc3s_wrapped_dispatch(
    *,
    load_forecast: np.ndarray,
    renewables_forecast: np.ndarray,
    load_true: np.ndarray,
    telemetry_events: list[Mapping[str, Any]],
    price: np.ndarray,
    optimization_cfg: Mapping[str, Any] | None = None,
    command_prefix: str = "cpsbench",
    variant: str = "linear",
) -> dict[str, Any]:
    variant_norm = _normalize_dc3s_dispatch_variant(variant)
    return _dc3s_dispatch_impl(
        load_forecast=load_forecast,
        renewables_forecast=renewables_forecast,
        load_true=load_true,
        telemetry_events=telemetry_events,
        price=price,
        optimization_cfg=optimization_cfg,
        command_prefix=command_prefix,
        policy_name=_DC3S_DISPATCH_VARIANT_TO_POLICY[variant_norm],
        law_override="linear",
        variant=variant_norm,
    )


def dc3s_ftit_dispatch(
    *,
    load_forecast: np.ndarray,
    renewables_forecast: np.ndarray,
    load_true: np.ndarray,
    telemetry_events: list[Mapping[str, Any]],
    price: np.ndarray,
    optimization_cfg: Mapping[str, Any] | None = None,
    command_prefix: str = "cpsbench",
) -> dict[str, Any]:
    return _dc3s_dispatch_impl(
        load_forecast=load_forecast,
        renewables_forecast=renewables_forecast,
        load_true=load_true,
        telemetry_events=telemetry_events,
        price=price,
        optimization_cfg=optimization_cfg,
        command_prefix=command_prefix,
        policy_name="dc3s_ftit",
        law_override="ftit_ro",
        variant="linear",
    )


def aci_conformal_dispatch(
    *,
    load_forecast: np.ndarray,
    renewables_forecast: np.ndarray,
    load_true: np.ndarray,
    telemetry_events: list[Mapping[str, Any]],
    price: np.ndarray,
    optimization_cfg: Mapping[str, Any] | None = None,
    command_prefix: str = "cpsbench",
    delta: float = 0.10,
    eta: float = 0.01,
    q_min: float = 25.0,
    q_max: float = 50000.0,
) -> dict[str, Any]:
    """Adaptive conformal interval baseline with deterministic LP proposals."""
    del command_prefix

    load_forecast_arr = _as_array(load_forecast)
    renewables_forecast_arr = _as_array(renewables_forecast, horizon=len(load_forecast_arr))
    load_true_arr = _as_array(load_true, horizon=len(load_forecast_arr))
    price_arr = _as_array(price, horizon=len(load_forecast_arr))
    horizon = len(load_forecast_arr)

    if horizon == 0:
        raise ValueError("ACI dispatch requires a non-empty horizon")
    if len(telemetry_events) != horizon:
        raise ValueError("telemetry_events length must match forecast horizon")
    if not (0.0 < float(delta) < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if float(eta) <= 0.0:
        raise ValueError("eta must be > 0")
    if float(q_min) <= 0.0:
        raise ValueError("q_min must be > 0")
    if float(q_max) < float(q_min):
        raise ValueError("q_max must be >= q_min")

    opt_cfg = dict(optimization_cfg or _load_optimization_cfg())
    dc3s_cfg = _load_dc3s_cfg()
    base = deterministic_lp_dispatch(
        load_forecast=load_forecast_arr,
        renewables_forecast=renewables_forecast_arr,
        price=price_arr,
        carbon=np.zeros_like(price_arr),
        optimization_cfg=opt_cfg,
    )
    constraints = dict(base["constraints"])

    proposed_charge = _as_array(base["proposed_charge_mw"], horizon=horizon)
    proposed_discharge = _as_array(base["proposed_discharge_mw"], horizon=horizon)
    safe_charge = np.zeros(horizon, dtype=float)
    safe_discharge = np.zeros(horizon, dtype=float)
    soc = np.zeros(horizon, dtype=float)
    lower = np.zeros(horizon, dtype=float)
    upper = np.zeros(horizon, dtype=float)

    q_t = max(50.0, float(np.quantile(np.abs(load_true_arr - load_forecast_arr), 0.90)))
    q_t = min(max(q_t, float(q_min)), float(q_max))
    current_soc = float(constraints["initial_soc_mwh"])

    for t in range(horizon):
        yhat = float(load_forecast_arr[t])
        ytrue = float(load_true_arr[t])
        lower_t = max(0.0, yhat - q_t)
        upper_t = yhat + q_t
        lower[t] = lower_t
        upper[t] = upper_t

        uncertainty_set = {
            "lower": [lower_t],
            "upper": [upper_t],
            "meta": {
                "drift_flag": False,
                "inflation_law": "aci",
                "q_t": float(q_t),
                "err": None,
                "delta_t": float(delta),
                "eta": float(eta),
            },
            "renewables_forecast": [float(renewables_forecast_arr[t])],
            "price": [float(price_arr[t])],
        }
        repair_constraints = {
            **constraints,
            "current_soc_mwh": current_soc,
            "last_net_mw": float(safe_discharge[t - 1] - safe_charge[t - 1]) if t > 0 else 0.0,
            "ramp_mw": float(dc3s_cfg.get("shield", {}).get("max_ramp_mw", 0.0) or 0.0),
            "degradation_cost_per_mwh": float(opt_cfg.get("battery", {}).get("degradation_cost_per_mwh", 10.0)),
            "max_grid_import_mw": float(opt_cfg.get("grid", {}).get("max_import_mw", 500.0)),
            "default_price_per_mwh": float(np.mean(price_arr)),
            "risk_weight_worst_case": float(opt_cfg.get("robust", {}).get("risk_weight_worst_case", 1.0)),
        }
        a_star = {
            "charge_mw": float(proposed_charge[t]),
            "discharge_mw": float(proposed_discharge[t]),
        }
        safe_action, _repair_meta = repair_action(
            a_star=a_star,
            state={"current_soc_mwh": current_soc},
            uncertainty_set=uncertainty_set,
            constraints=repair_constraints,
            cfg=dc3s_cfg,
        )

        safe_charge[t] = float(safe_action["charge_mw"])
        safe_discharge[t] = float(safe_action["discharge_mw"])
        eff = max(float(constraints["efficiency"]), 1e-6)
        current_soc = current_soc + eff * safe_charge[t] - safe_discharge[t] / eff
        current_soc = min(float(constraints["max_soc_mwh"]), max(float(constraints["min_soc_mwh"]), current_soc))
        soc[t] = current_soc

        err = 1.0 if abs(ytrue - yhat) > q_t else 0.0
        q_t = min(max(q_t + float(eta) * (err - float(delta)), float(q_min)), float(q_max))

    grid_import = np.maximum(0.0, load_forecast_arr - renewables_forecast_arr - safe_discharge + safe_charge)
    expected_cost = float(np.sum(price_arr * grid_import))
    return {
        "policy": "aci_conformal",
        "dispatch_plan": {
            "battery_charge_mw": safe_charge.tolist(),
            "battery_discharge_mw": safe_discharge.tolist(),
            "soc_mwh": soc.tolist(),
            "grid_mw": grid_import.tolist(),
            "expected_cost_usd": expected_cost,
        },
        "proposed_charge_mw": proposed_charge,
        "proposed_discharge_mw": proposed_discharge,
        "safe_charge_mw": safe_charge,
        "safe_discharge_mw": safe_discharge,
        "soc_mwh": soc,
        "interval_lower": lower,
        "interval_upper": upper,
        "certificates": [None] * horizon,
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
    sigma_sq: float | None = None
    adaptive_state: dict[str, Any] = field(default_factory=dict)


def init_dc3s_loop_state(dc3s_cfg: Mapping[str, Any]) -> DC3SLoopState:
    return DC3SLoopState(detector=PageHinkleyDetector.from_state(None, cfg=dc3s_cfg.get("drift", {})))


def _derive_intervention_reason(repair_meta: Mapping[str, Any], intervened: bool) -> str | None:
    if not intervened:
        return None
    robust_meta = repair_meta.get("robust_meta")
    if isinstance(robust_meta, Mapping):
        reason = robust_meta.get("reason")
        if isinstance(reason, str) and reason:
            return reason
    return "projection_clip"


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


def _dc3s_step_impl(
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
    policy_name: str,
    law_override: str | None,
) -> dict[str, Any]:
    """Single-step DC3S policy with configurable uncertainty law and action repair."""
    opt_cfg = dict(optimization_cfg or _load_optimization_cfg())
    cfg = dict(dc3s_cfg or _load_dc3s_cfg())
    if law_override is not None:
        cfg["law"] = str(law_override)
    inflation_law = str(cfg.get("inflation_law", "linear")).strip().lower()
    if inflation_law not in {"linear", "kappa"}:
        raise ValueError(f"Unsupported inflation_law: {inflation_law}")
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
    ftit_cfg = {**dict(cfg.get("ftit", {})), "law": str(cfg.get("law", "linear")).strip().lower()}

    w_t, flags = compute_reliability(
        event,
        dc3s_state.prev_event,
        expected_cadence_s=cadence,
        reliability_cfg=cfg.get("reliability", {}),
        adaptive_state=dc3s_state.adaptive_state,
        ftit_cfg=ftit_cfg,
    )
    residual = abs(float(load_true_t) - float(load_obs_window[0]))
    drift = dc3s_state.detector.update(residual)
    ftit_state = update_ftit_state(
        adaptive_state=dc3s_state.adaptive_state,
        fault_flags=flags.get("fault_flags", {}),
        constraints={
            **constraints,
            "current_soc_mwh": float(observed_soc_mwh),
        },
        cfg=ftit_cfg,
        stale_tracker=flags.get("stale_tracker"),
        sigma2_observation=float(residual) ** 2,
    )
    dc3s_state.adaptive_state = ftit_state["adaptive_state"]
    q_t = float(q_base if q_base is not None else max(50.0, residual))
    cfg_runtime = dict(cfg)
    if str(cfg_runtime.get("law", "linear")).strip().lower() == "ftit_ro":
        cfg_runtime["ftit_runtime"] = {"sigma2": float(ftit_state["sigma2"])}
    if policy_name == "dc3s_wrapped" and inflation_law == "kappa":
        sigma_sq = float(dc3s_state.sigma_sq) if dc3s_state.sigma_sq is not None else float(residual) ** 2
        lo, hi, meta = build_uncertainty_set_kappa(
            yhat=np.asarray([load_obs_window[0]], dtype=float),
            q=np.asarray([q_t], dtype=float),
            w_t=float(w_t),
            drift_flag=bool(drift.get("drift", False)),
            cfg=cfg_runtime,
            sigma_sq=sigma_sq,
            prev_inflation=dc3s_state.prev_inflation,
        )
    else:
        lo, hi, meta = build_uncertainty_set(
            yhat=np.asarray([load_obs_window[0]], dtype=float),
            q=np.asarray([q_t], dtype=float),
            w_t=float(w_t),
            drift_flag=bool(drift.get("drift", False)),
            cfg=cfg_runtime,
            prev_inflation=dc3s_state.prev_inflation,
        )
    dc3s_state.prev_inflation = float(meta.get("inflation", 1.0))
    meta["gamma_mw"] = float(ftit_state["gamma_mw"])
    meta["e_t_mwh"] = float(ftit_state["e_t_mwh"])
    meta["soc_tube_lower_mwh"] = float(ftit_state["soc_tube_lower_mwh"])
    meta["soc_tube_upper_mwh"] = float(ftit_state["soc_tube_upper_mwh"])

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
    if str(cfg_runtime.get("law", "linear")).strip().lower() == "ftit_ro":
        repair_constraints["ftit_soc_min_mwh"] = float(ftit_state["soc_tube_lower_mwh"])
        repair_constraints["ftit_soc_max_mwh"] = float(ftit_state["soc_tube_upper_mwh"])
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
        cfg=cfg_runtime,
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
        controller=policy_name,
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
        intervention_reason=_derive_intervention_reason(repair_meta, bool(repair_meta.get("repaired", False))),
        reliability_w=float(w_t),
        drift_flag=bool(drift.get("drift", False)),
        inflation=float(meta.get("inflation", 1.0)),
        guarantee_checks_passed=bool(guarantee_ok),
        guarantee_fail_reasons=guarantee_reasons,
        assumptions_version=str(cfg.get("assumptions_version", "dc3s-assumptions-v1")),
        gamma_mw=float(ftit_state["gamma_mw"]),
        e_t_mwh=float(ftit_state["e_t_mwh"]),
        soc_tube_lower_mwh=float(ftit_state["soc_tube_lower_mwh"]),
        soc_tube_upper_mwh=float(ftit_state["soc_tube_upper_mwh"]),
    )
    dc3s_state.prev_event = event
    dc3s_state.prev_hash = str(certificate.get("certificate_hash"))

    return {
        "policy": policy_name,
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
    return _dc3s_step_impl(
        load_obs_window=load_obs_window,
        renew_obs_window=renew_obs_window,
        load_true_t=load_true_t,
        telemetry_event=telemetry_event,
        price_window=price_window,
        observed_soc_mwh=observed_soc_mwh,
        dc3s_state=dc3s_state,
        command_id=command_id,
        optimization_cfg=optimization_cfg,
        dc3s_cfg=dc3s_cfg,
        q_base=q_base,
        policy_name="dc3s_wrapped",
        law_override="linear",
    )


def dc3s_ftit_step(
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
    return _dc3s_step_impl(
        load_obs_window=load_obs_window,
        renew_obs_window=renew_obs_window,
        load_true_t=load_true_t,
        telemetry_event=telemetry_event,
        price_window=price_window,
        observed_soc_mwh=observed_soc_mwh,
        dc3s_state=dc3s_state,
        command_id=command_id,
        optimization_cfg=optimization_cfg,
        dc3s_cfg=dc3s_cfg,
        q_base=q_base,
        policy_name="dc3s_ftit",
        law_override="ftit_ro",
    )
