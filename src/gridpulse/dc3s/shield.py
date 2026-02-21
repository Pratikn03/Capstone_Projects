"""Safety shield that repairs unsafe controller actions."""
from __future__ import annotations

from typing import Any, Mapping

from gridpulse.optimizer.robust_dispatch import RobustDispatchConfig, optimize_robust_dispatch


def _f(x: Any, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _projection_repair(
    a_star: Mapping[str, Any],
    state: Mapping[str, Any],
    uncertainty_set: Mapping[str, Any],
    constraints: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    charge_in = max(0.0, _f(a_star.get("charge_mw"), 0.0))
    discharge_in = max(0.0, _f(a_star.get("discharge_mw"), 0.0))
    net_in = discharge_in - charge_in

    capacity = _f(constraints.get("capacity_mwh"), 10.0)
    min_soc = _f(constraints.get("min_soc_mwh"), 0.0)
    max_soc = _f(constraints.get("max_soc_mwh"), capacity)
    max_power = _f(constraints.get("max_power_mw"), max(_f(constraints.get("max_charge_mw"), 0.0), _f(constraints.get("max_discharge_mw"), 0.0)))
    max_charge = _f(constraints.get("max_charge_mw"), max_power)
    max_discharge = _f(constraints.get("max_discharge_mw"), max_power)
    ramp_mw = _f(constraints.get("ramp_mw"), 0.0)
    last_net = _f(constraints.get("last_net_mw"), 0.0)

    charge_eff = max(1e-6, _f(constraints.get("charge_efficiency"), 1.0))
    discharge_eff = max(1e-6, _f(constraints.get("discharge_efficiency"), 1.0))

    drift_flag = bool(uncertainty_set.get("meta", {}).get("drift_flag", False))
    reserve_soc_pct_drift = _f(cfg.get("shield", {}).get("reserve_soc_pct_drift"), 0.0)
    min_soc_eff = min(max_soc, min_soc + (reserve_soc_pct_drift * capacity if drift_flag else 0.0))

    net = net_in
    if ramp_mw > 0.0:
        net = max(last_net - ramp_mw, min(last_net + ramp_mw, net))

    current_soc = _f(state.get("current_soc_mwh"), 0.0)

    if net >= 0.0:
        feasible_by_soc = max(0.0, (current_soc - min_soc_eff) * discharge_eff)
        discharge = min(net, max_discharge, max_power, feasible_by_soc)
        charge = 0.0
    else:
        feasible_by_soc = max(0.0, (max_soc - current_soc) / charge_eff)
        charge = min(-net, max_charge, max_power, feasible_by_soc)
        discharge = 0.0

    next_soc = current_soc + charge_eff * charge - (discharge / discharge_eff)
    safe = {
        "charge_mw": float(max(0.0, charge)),
        "discharge_mw": float(max(0.0, discharge)),
    }
    repaired = abs(safe["charge_mw"] - charge_in) > 1e-9 or abs(safe["discharge_mw"] - discharge_in) > 1e-9

    meta = {
        "mode": "projection",
        "repaired": bool(repaired),
        "input_charge_mw": float(charge_in),
        "input_discharge_mw": float(discharge_in),
        "net_input_mw": float(net_in),
        "net_after_ramp_mw": float(net),
        "current_soc_mwh": float(current_soc),
        "next_soc_mwh": float(next_soc),
        "effective_min_soc_mwh": float(min_soc_eff),
        "max_soc_mwh": float(max_soc),
    }
    return safe, meta


def _robust_cfg_from_constraints(constraints: Mapping[str, Any]) -> RobustDispatchConfig:
    cfg_obj = constraints.get("robust_config")
    if isinstance(cfg_obj, RobustDispatchConfig):
        return cfg_obj

    capacity = _f(constraints.get("capacity_mwh"), 10.0)
    max_power = _f(constraints.get("max_power_mw"), 5.0)
    max_charge = _f(constraints.get("max_charge_mw"), max_power)
    max_discharge = _f(constraints.get("max_discharge_mw"), max_power)
    efficiency = max(1e-6, _f(constraints.get("charge_efficiency"), _f(constraints.get("efficiency"), 0.95)))
    return RobustDispatchConfig(
        battery_capacity_mwh=capacity,
        battery_max_charge_mw=max_charge,
        battery_max_discharge_mw=max_discharge,
        battery_charge_efficiency=efficiency,
        battery_discharge_efficiency=efficiency,
        battery_initial_soc_mwh=_f(constraints.get("current_soc_mwh"), capacity / 2.0),
        battery_min_soc_mwh=_f(constraints.get("min_soc_mwh"), 0.0),
        battery_max_soc_mwh=_f(constraints.get("max_soc_mwh"), capacity),
        max_grid_import_mw=_f(constraints.get("max_grid_import_mw"), 500.0),
        default_price_per_mwh=_f(constraints.get("default_price_per_mwh"), 60.0),
        degradation_cost_per_mwh=_f(constraints.get("degradation_cost_per_mwh"), 10.0),
        risk_weight_worst_case=_f(constraints.get("risk_weight_worst_case"), 1.0),
        time_step_hours=1.0,
        solver_name=str(constraints.get("solver_name", "appsi_highs")),
    )


def _robust_resolve(
    state: Mapping[str, Any],
    uncertainty_set: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    lower = uncertainty_set.get("lower")
    upper = uncertainty_set.get("upper")
    renewables = uncertainty_set.get("renewables_forecast")
    if lower is None or upper is None or renewables is None:
        return None, {"robust_attempted": False, "reason": "missing_uncertainty_inputs"}

    try:
        result = optimize_robust_dispatch(
            load_lower_bound=lower,
            load_upper_bound=upper,
            renewables_forecast=renewables,
            price=uncertainty_set.get("price"),
            config=_robust_cfg_from_constraints(constraints),
            verbose=False,
        )
    except Exception as exc:
        return None, {"robust_attempted": True, "reason": f"robust_failed:{exc}"}

    if not bool(result.get("feasible", False)):
        return None, {
            "robust_attempted": True,
            "reason": "robust_infeasible",
            "solver_status": result.get("solver_status"),
        }

    charge = float(result.get("battery_charge_mw", [0.0])[0])
    discharge = float(result.get("battery_discharge_mw", [0.0])[0])
    return {"charge_mw": charge, "discharge_mw": discharge}, {
        "robust_attempted": True,
        "reason": "ok",
        "solver_status": result.get("solver_status"),
    }


def repair_action(
    a_star: Mapping[str, Any],
    state: Mapping[str, Any],
    uncertainty_set: Mapping[str, Any],
    constraints: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Repair potentially unsafe action.

    Modes:
    - projection: local clipping against SOC/power/ramp constraints
    - robust_resolve: solve robust action first, then project as final guard
    """
    mode = str(cfg.get("shield", {}).get("mode", "projection"))
    if mode == "robust_resolve":
        robust_action, robust_meta = _robust_resolve(state, uncertainty_set, constraints)
        action_seed = robust_action if robust_action is not None else a_star
        safe, proj_meta = _projection_repair(action_seed, state, uncertainty_set, constraints, cfg)
        proj_meta["mode"] = "robust_resolve"
        proj_meta["robust_meta"] = robust_meta
        proj_meta["seed_action"] = {
            "charge_mw": float(max(0.0, _f(action_seed.get("charge_mw"), 0.0))),
            "discharge_mw": float(max(0.0, _f(action_seed.get("discharge_mw"), 0.0))),
        }
        return safe, proj_meta

    return _projection_repair(a_star, state, uncertainty_set, constraints, cfg)
