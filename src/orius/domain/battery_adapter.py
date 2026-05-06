import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from orius.cpsbench_iot.plant import BatteryPlant
from orius.domain.adapter import Action, DomainAdapter, Optimizer, Plant, UncertaintySet
from orius.optimizer.robust_dispatch import (
    CVaRDispatchConfig,
    RobustDispatchConfig,
    optimize_cvar_dispatch,
    optimize_robust_dispatch,
)
from orius.utils.config import load_config


# Module-level configs — loaded lazily to avoid FileNotFoundError when files are absent.
def _lazy_config(path: str):
    try:
        return load_config(path)
    except FileNotFoundError:
        from orius.utils.config import _AttrDict

        return _AttrDict()


OPTIMIZATION_CFG = _lazy_config("configs/optimization.yaml")
PLANT_CFG = _lazy_config("configs/plant.yaml")
DC3S_CFG = _lazy_config("configs/dc3s.yaml")


def _f(x: Any, default: float) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return v
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
    max_power = _f(
        constraints.get("max_power_mw"),
        max(_f(constraints.get("max_charge_mw"), 0.0), _f(constraints.get("max_discharge_mw"), 0.0)),
    )
    max_charge = _f(constraints.get("max_charge_mw"), max_power)
    max_discharge = _f(constraints.get("max_discharge_mw"), max_power)
    ramp_mw = _f(constraints.get("ramp_mw"), 0.0)
    last_net = _f(constraints.get("last_net_mw"), 0.0)

    charge_eff = max(1e-6, _f(constraints.get("charge_efficiency"), 1.0))
    discharge_eff = max(1e-6, _f(constraints.get("discharge_efficiency"), 1.0))

    drift_flag = bool(uncertainty_set.get("meta", {}).get("drift_flag", False))
    reserve_soc_pct_drift = _f(cfg.get("shield", {}).get("reserve_soc_pct_drift"), 0.0)
    ftit_min = constraints.get("ftit_soc_min_mwh")
    ftit_max = constraints.get("ftit_soc_max_mwh")
    if ftit_min is not None or ftit_max is not None:
        min_soc_eff = _f(ftit_min, min_soc)
        max_soc_eff = _f(ftit_max, max_soc)
    else:
        min_soc_eff = min(max_soc, min_soc + (reserve_soc_pct_drift * capacity if drift_flag else 0.0))
        max_soc_eff = max_soc

    net = net_in
    if ramp_mw > 0.0:
        net = max(last_net - ramp_mw, min(last_net + ramp_mw, net))

    current_soc = _f(state.get("current_soc_mwh"), 0.0)

    if net >= 0.0:
        feasible_by_soc = max(0.0, (current_soc - min_soc_eff) * discharge_eff)
        discharge = min(net, max_discharge, max_power, feasible_by_soc)
        charge = 0.0
    else:
        feasible_by_soc = max(0.0, (max_soc_eff - current_soc) / charge_eff)
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
        "effective_max_soc_mwh": float(max_soc_eff),
        "max_soc_mwh": float(max_soc_eff),
    }
    return safe, meta


def _l2_projection_repair(
    a_star: Mapping[str, Any],
    state: Mapping[str, Any],
    uncertainty_set: Mapping[str, Any],
    constraints: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Minimum-norm (L2) projection onto the battery safe-action polytope.

    Solves two sub-problems (charge-only, discharge-only) to handle the
    mutual-exclusion constraint (cannot charge and discharge simultaneously),
    then picks the solution nearest to the candidate action.

    Reference: PDF Section 4 – DC3S Safety Shield, Definition 18.
    """
    charge_in = max(0.0, _f(a_star.get("charge_mw"), 0.0))
    discharge_in = max(0.0, _f(a_star.get("discharge_mw"), 0.0))

    capacity = _f(constraints.get("capacity_mwh"), 10.0)
    min_soc = _f(constraints.get("min_soc_mwh"), 0.0)
    max_soc = _f(constraints.get("max_soc_mwh"), capacity)
    max_power = _f(
        constraints.get("max_power_mw"),
        max(_f(constraints.get("max_charge_mw"), 0.0), _f(constraints.get("max_discharge_mw"), 0.0)),
    )
    max_charge = _f(constraints.get("max_charge_mw"), max_power)
    max_discharge = _f(constraints.get("max_discharge_mw"), max_power)
    ramp_mw = _f(constraints.get("ramp_mw"), 0.0)
    last_net = _f(constraints.get("last_net_mw"), 0.0)

    charge_eff = max(1e-6, _f(constraints.get("charge_efficiency"), 1.0))
    discharge_eff = max(1e-6, _f(constraints.get("discharge_efficiency"), 1.0))

    drift_flag = bool(uncertainty_set.get("meta", {}).get("drift_flag", False))
    reserve_soc_pct_drift = _f(cfg.get("shield", {}).get("reserve_soc_pct_drift"), 0.0)
    ftit_min = constraints.get("ftit_soc_min_mwh")
    ftit_max = constraints.get("ftit_soc_max_mwh")
    if ftit_min is not None or ftit_max is not None:
        min_soc_eff = _f(ftit_min, min_soc)
        max_soc_eff = _f(ftit_max, max_soc)
    else:
        min_soc_eff = min(max_soc, min_soc + (reserve_soc_pct_drift * capacity if drift_flag else 0.0))
        max_soc_eff = max_soc

    current_soc = _f(state.get("current_soc_mwh"), 0.0)
    a_ref = np.array([charge_in, discharge_in])

    def _solve_charge_only() -> tuple[np.ndarray, float]:
        """Minimise ||[c, 0] - a_ref||^2 s.t. battery constraints."""
        # c in [0, max_charge], ramp: -(c) in [last_net - ramp, last_net + ramp]
        c_lo = 0.0
        c_hi = min(max_charge, max_power)
        # SOC feasibility: current_soc + charge_eff * c <= max_soc_eff
        soc_room = max(0.0, (max_soc_eff - current_soc) / charge_eff)
        c_hi = min(c_hi, soc_room)
        # Ramp: net = -c, must satisfy last_net - ramp <= -c <= last_net + ramp
        if ramp_mw > 0.0:
            c_lo = max(c_lo, -(last_net + ramp_mw))  # -c >= last_net - ramp => c <= -(last_net - ramp)
            c_hi_ramp = -(last_net - ramp_mw)
            c_lo = max(c_lo, 0.0)
            c_hi = min(c_hi, max(0.0, c_hi_ramp))
        if c_lo > c_hi + 1e-12:
            return np.array([0.0, 0.0]), float(np.sum(a_ref**2))
        # Nearest to charge_in in [c_lo, c_hi], discharge=0
        c_opt = float(np.clip(charge_in, c_lo, c_hi))
        sol = np.array([c_opt, 0.0])
        return sol, float(np.sum((sol - a_ref) ** 2))

    def _solve_discharge_only() -> tuple[np.ndarray, float]:
        """Minimise ||[0, d] - a_ref||^2 s.t. battery constraints."""
        d_lo = 0.0
        d_hi = min(max_discharge, max_power)
        # SOC feasibility: current_soc - d / discharge_eff >= min_soc_eff
        soc_room = max(0.0, (current_soc - min_soc_eff) * discharge_eff)
        d_hi = min(d_hi, soc_room)
        # Ramp: net = d, must satisfy last_net - ramp <= d <= last_net + ramp
        if ramp_mw > 0.0:
            d_lo = max(d_lo, last_net - ramp_mw)
            d_hi = min(d_hi, last_net + ramp_mw)
            d_lo = max(d_lo, 0.0)
        if d_lo > d_hi + 1e-12:
            return np.array([0.0, 0.0]), float(np.sum(a_ref**2))
        d_opt = float(np.clip(discharge_in, d_lo, d_hi))
        sol = np.array([0.0, d_opt])
        return sol, float(np.sum((sol - a_ref) ** 2))

    def _solve_idle() -> tuple[np.ndarray, float]:
        sol = np.array([0.0, 0.0])
        return sol, float(np.sum((sol - a_ref) ** 2))

    # Solve all three sub-problems and pick minimum-distance solution
    candidates = [_solve_charge_only(), _solve_discharge_only(), _solve_idle()]
    best_sol, best_dist = min(candidates, key=lambda x: x[1])

    charge_out = float(max(0.0, best_sol[0]))
    discharge_out = float(max(0.0, best_sol[1]))
    next_soc = current_soc + charge_eff * charge_out - (discharge_out / discharge_eff)

    safe = {
        "charge_mw": charge_out,
        "discharge_mw": discharge_out,
    }
    repaired = abs(safe["charge_mw"] - charge_in) > 1e-9 or abs(safe["discharge_mw"] - discharge_in) > 1e-9

    meta = {
        "mode": "l2_projection",
        "repaired": bool(repaired),
        "l2_distance": float(best_dist**0.5),
        "input_charge_mw": float(charge_in),
        "input_discharge_mw": float(discharge_in),
        "net_input_mw": float(discharge_in - charge_in),
        "current_soc_mwh": float(current_soc),
        "next_soc_mwh": float(next_soc),
        "effective_min_soc_mwh": float(min_soc_eff),
        "effective_max_soc_mwh": float(max_soc_eff),
        "max_soc_mwh": float(max_soc_eff),
    }
    return safe, meta


def _safe_landing_repair(
    a_star: Mapping[str, Any],
    state: Mapping[str, Any],
    uncertainty_set: Mapping[str, Any],
    constraints: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    from orius.universal_theory.battery_instantiation import compute_battery_safe_landing_action

    landing_cfg = dict(cfg.get("shield", {}).get("safe_landing", {}))
    current_soc = _f(state.get("current_soc_mwh"), 0.0)
    theorem_constraints = dict(constraints)
    if "safe_margin_mwh" in landing_cfg:
        theorem_constraints["safe_landing_margin_mwh"] = landing_cfg.get("safe_margin_mwh")
    if "safe_margin_pct" in landing_cfg:
        theorem_constraints["safe_landing_margin_pct"] = landing_cfg.get("safe_margin_pct")
    if "target_soc_mwh" in landing_cfg:
        theorem_constraints["safe_landing_target_soc_mwh"] = landing_cfg.get("target_soc_mwh")
    landing = compute_battery_safe_landing_action(
        current_soc_mwh=current_soc,
        constraints=theorem_constraints,
    )
    safe = {
        "charge_mw": float(landing["fallback_action"]["charge_mw"]),
        "discharge_mw": float(landing["fallback_action"]["discharge_mw"]),
    }
    input_charge = max(0.0, _f(a_star.get("charge_mw"), 0.0))
    input_discharge = max(0.0, _f(a_star.get("discharge_mw"), 0.0))
    repaired = (
        abs(safe["charge_mw"] - input_charge) > 1e-9 or abs(safe["discharge_mw"] - input_discharge) > 1e-9
    )
    meta = {
        "mode": "safe_landing",
        "repaired": bool(repaired),
        "intervention_reason": "safe_landing",
        "input_charge_mw": float(input_charge),
        "input_discharge_mw": float(input_discharge),
        "current_soc_mwh": float(landing["current_soc_mwh"]),
        "target_soc_mwh": float(landing["target_soc_mwh"]),
        "safe_zone_min_mwh": float(landing["safe_zone_min_mwh"]),
        "safe_zone_max_mwh": float(landing["safe_zone_max_mwh"]),
        "next_soc_mwh": float(landing["next_soc_nominal_mwh"]),
        "landing_region": str(landing["landing_region"]),
        "w_t": float(uncertainty_set.get("meta", {}).get("w_t", 1.0)),
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
    charge_eff = max(1e-6, _f(constraints.get("charge_efficiency"), _f(constraints.get("efficiency"), 0.95)))
    discharge_eff = max(
        1e-6, _f(constraints.get("discharge_efficiency"), _f(constraints.get("efficiency"), 0.95))
    )
    return RobustDispatchConfig(
        battery_capacity_mwh=capacity,
        battery_max_charge_mw=max_charge,
        battery_max_discharge_mw=max_discharge,
        battery_charge_efficiency=charge_eff,
        battery_discharge_efficiency=discharge_eff,
        battery_initial_soc_mwh=_f(constraints.get("current_soc_mwh"), capacity / 2.0),
        battery_min_soc_mwh=_f(constraints.get("ftit_soc_min_mwh"), _f(constraints.get("min_soc_mwh"), 0.0)),
        battery_max_soc_mwh=_f(
            constraints.get("ftit_soc_max_mwh"), _f(constraints.get("max_soc_mwh"), capacity)
        ),
        max_grid_import_mw=_f(constraints.get("max_grid_import_mw"), 500.0),
        default_price_per_mwh=_f(constraints.get("default_price_per_mwh"), 60.0),
        degradation_cost_per_mwh=_f(constraints.get("degradation_cost_per_mwh"), 10.0),
        risk_weight_worst_case=_f(constraints.get("risk_weight_worst_case"), 1.0),
        time_step_hours=_f(constraints.get("time_step_hours"), 1.0),
        solver_name=str(constraints.get("solver_name", "appsi_highs")),
    )


def _cvar_cfg_from_constraints(
    constraints: Mapping[str, Any],
    *,
    n_scenarios: int,
    scenario_seed: int,
) -> CVaRDispatchConfig:
    robust_cfg = _robust_cfg_from_constraints(constraints)
    return CVaRDispatchConfig(
        battery_capacity_mwh=robust_cfg.battery_capacity_mwh,
        battery_max_charge_mw=robust_cfg.battery_max_charge_mw,
        battery_max_discharge_mw=robust_cfg.battery_max_discharge_mw,
        battery_charge_efficiency=robust_cfg.battery_charge_efficiency,
        battery_discharge_efficiency=robust_cfg.battery_discharge_efficiency,
        battery_initial_soc_mwh=robust_cfg.battery_initial_soc_mwh,
        battery_min_soc_mwh=robust_cfg.battery_min_soc_mwh,
        battery_max_soc_mwh=robust_cfg.battery_max_soc_mwh,
        max_grid_import_mw=robust_cfg.max_grid_import_mw,
        default_price_per_mwh=robust_cfg.default_price_per_mwh,
        degradation_cost_per_mwh=robust_cfg.degradation_cost_per_mwh,
        risk_weight_worst_case=robust_cfg.risk_weight_worst_case,
        time_step_hours=robust_cfg.time_step_hours,
        solver_name=robust_cfg.solver_name,
        beta=_f(constraints.get("cvar_beta"), 0.90),
        n_scenarios=int(max(2, n_scenarios)),
        risk_weight_cvar=_f(constraints.get("cvar_risk_weight"), 1.0),
        scenario_seed=int(scenario_seed),
    )


def repair_battery_action(
    a_star: Mapping[str, Any],
    state: Mapping[str, Any],
    uncertainty_set: Mapping[str, Any],
    constraints: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Battery-specific repair with legacy (a_star, state, uncertainty_set, constraints, cfg) signature.

    Used by runner, baselines, and CPSBench when domain_adapter is not passed.
    """
    cfg = dict(cfg or {})
    constraints = dict(constraints or {})
    uncertainty_set = dict(uncertainty_set or {})
    mode = str(cfg.get("shield", {}).get("mode", "l2_projection"))
    landing_cfg = dict(cfg.get("shield", {}).get("safe_landing", {}))
    landing_threshold = landing_cfg.get("w_threshold")

    if mode != "safe_landing" and landing_cfg.get("auto_activate", False) and landing_threshold is not None:
        w_t = _f(uncertainty_set.get("meta", {}).get("w_t"), 1.0)
        if w_t <= _f(landing_threshold, 0.0):
            mode = "safe_landing"

    if mode == "safe_landing":
        return _safe_landing_repair(a_star, state, uncertainty_set, constraints, cfg)

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

    if mode == "robust_resolve_cvar":
        cvar_action, cvar_meta = _robust_resolve_cvar(state, uncertainty_set, constraints)
        action_seed = cvar_action if cvar_action is not None else a_star
        safe, proj_meta = _projection_repair(action_seed, state, uncertainty_set, constraints, cfg)
        proj_meta["mode"] = "robust_resolve_cvar"
        proj_meta["robust_meta"] = cvar_meta
        proj_meta["seed_action"] = {
            "charge_mw": float(max(0.0, _f(action_seed.get("charge_mw"), 0.0))),
            "discharge_mw": float(max(0.0, _f(action_seed.get("discharge_mw"), 0.0))),
        }
        return safe, proj_meta

    if mode == "l2_projection":
        return _l2_projection_repair(a_star, state, uncertainty_set, constraints, cfg)

    return _projection_repair(a_star, state, uncertainty_set, constraints, cfg)


def _sample_load_scenarios(
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    n_scenarios: int,
    seed: int,
) -> np.ndarray:
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.size != hi.size:
        raise ValueError("lower and upper must have the same length")
    if np.any(lo > hi):
        raise ValueError("lower cannot exceed upper")
    rng = np.random.default_rng(int(seed))
    return rng.uniform(lo, hi, size=(int(n_scenarios), lo.size))


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

    charge_list = result.get("battery_charge_mw", [0.0])
    charge = float(charge_list[0]) if charge_list else 0.0
    discharge_list = result.get("battery_discharge_mw", [0.0])
    discharge = float(discharge_list[0]) if discharge_list else 0.0
    return {"charge_mw": charge, "discharge_mw": discharge}, {
        "robust_attempted": True,
        "reason": "ok",
        "solver_status": result.get("solver_status"),
    }


def _robust_resolve_cvar(
    state: Mapping[str, Any],
    uncertainty_set: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    _ = state
    lower = uncertainty_set.get("lower")
    upper = uncertainty_set.get("upper")
    renewables = uncertainty_set.get("renewables_forecast")
    if lower is None or upper is None or renewables is None:
        return None, {"robust_attempted": False, "reason": "missing_uncertainty_inputs"}

    cvar_n = int(max(2, _f(constraints.get("cvar_n_scenarios"), 20.0)))
    cvar_seed = int(_f(constraints.get("scenario_seed"), 0.0))

    try:
        scenarios = _sample_load_scenarios(
            lower=np.asarray(lower, dtype=float),
            upper=np.asarray(upper, dtype=float),
            n_scenarios=cvar_n,
            seed=cvar_seed,
        )
        result = optimize_cvar_dispatch(
            load_scenarios=scenarios,
            renewables_forecast=renewables,
            price=uncertainty_set.get("price"),
            config=_cvar_cfg_from_constraints(constraints, n_scenarios=cvar_n, scenario_seed=cvar_seed),
            verbose=False,
        )
    except Exception as exc:
        return None, {"robust_attempted": True, "reason": f"cvar_failed:{exc}"}

    if not bool(result.get("feasible", False)):
        return None, {
            "robust_attempted": True,
            "reason": "cvar_infeasible",
            "solver_status": result.get("solver_status"),
            "n_scenarios": int(cvar_n),
        }

    charge_list = result.get("battery_charge_mw", [0.0])
    charge = float(charge_list[0]) if charge_list else 0.0
    discharge_list = result.get("battery_discharge_mw", [0.0])
    discharge = float(discharge_list[0]) if discharge_list else 0.0
    return {"charge_mw": charge, "discharge_mw": discharge}, {
        "robust_attempted": True,
        "reason": "ok",
        "solver_status": result.get("solver_status"),
        "eta": result.get("eta"),
        "cvar_cost": result.get("cvar_cost"),
        "n_scenarios": int(cvar_n),
    }


class BatteryOptimizer(Optimizer):
    """
    A wrapper for the existing battery LP optimizer to conform to the Optimizer interface.
    """

    def __init__(self, config):
        self.config = config

    def get_candidate_action(self, state: Any, forecast: Any) -> Action:
        return 0.0


class BatteryDomainAdapter(DomainAdapter):
    """
    A domain-specific adapter for the battery energy storage system.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        # For now, we use the globally loaded configs.
        # A full implementation would merge the passed 'config' with defaults.
        super().__init__(config or DC3S_CFG)
        self.plant_cfg = PLANT_CFG
        self.optimizer_cfg = OPTIMIZATION_CFG

        self.plant = BatteryPlant(
            soc_init_mw=self.plant_cfg.soc_init_mw,
            c_rate=self.plant_cfg.c_rate,
            capacity_mwh=self.plant_cfg.capacity_mwh,
        )
        self.optimizer = BatteryOptimizer(self.optimizer_cfg)

    def get_plant(self) -> Plant:
        return self.plant

    def get_optimizer(self) -> Optimizer:
        return self.optimizer

    def project_to_safe_set(
        self, candidate_action: Action, uncertainty_set: UncertaintySet, state: Any
    ) -> tuple[Action, dict[str, Any]]:
        # This method now contains the logic from the original `repair_action`
        # in shield.py. It uses helper functions now part of this module.

        # The `constraints` dict is built from the adapter's own knowledge
        # of the domain's physical properties and configuration.
        constraints = {
            "capacity_mwh": self.plant_cfg.capacity_mwh,
            "min_soc_mwh": self.plant_cfg.min_soc_mwh,
            "max_soc_mwh": self.plant_cfg.max_soc_mwh,
            "max_power_mw": self.plant_cfg.max_power_mw,
            "max_charge_mw": self.plant_cfg.max_charge_mw,
            "max_discharge_mw": self.plant_cfg.max_discharge_mw,
            "charge_efficiency": self.plant_cfg.charge_efficiency,
            "discharge_efficiency": self.plant_cfg.discharge_efficiency,
            # These would come from the state object
            "current_soc_mwh": getattr(state, "current_soc_mwh", 0.0),
            "last_net_mw": getattr(state, "last_net_mw", 0.0),
            "ftit_soc_min_mwh": uncertainty_set.get("ftit_soc_min_mwh"),
            "ftit_soc_max_mwh": uncertainty_set.get("ftit_soc_max_mwh"),
        }

        cfg = self.config  # Use the adapter's own config
        mode = str(cfg.get("shield", {}).get("mode", "l2_projection"))
        landing_cfg = dict(cfg.get("shield", {}).get("safe_landing", {}))
        landing_threshold = landing_cfg.get("w_threshold")

        if (
            mode != "safe_landing"
            and landing_cfg.get("auto_activate", False)
            and landing_threshold is not None
        ):
            w_t = _f(uncertainty_set.get("meta", {}).get("w_t"), 1.0)
            if w_t <= _f(landing_threshold, 0.0):
                mode = "safe_landing"

        if mode == "safe_landing":
            return _safe_landing_repair(candidate_action, state, uncertainty_set, constraints, cfg)

        if mode == "robust_resolve":
            robust_action, robust_meta = _robust_resolve(state, uncertainty_set, constraints)
            action_seed = robust_action if robust_action is not None else candidate_action
            safe, proj_meta = _projection_repair(action_seed, state, uncertainty_set, constraints, cfg)
            proj_meta["mode"] = "robust_resolve"
            proj_meta["robust_meta"] = robust_meta
            proj_meta["seed_action"] = {
                "charge_mw": float(max(0.0, _f(action_seed.get("charge_mw"), 0.0))),
                "discharge_mw": float(max(0.0, _f(action_seed.get("discharge_mw"), 0.0))),
            }
            return safe, proj_meta

        if mode == "robust_resolve_cvar":
            cvar_action, cvar_meta = _robust_resolve_cvar(state, uncertainty_set, constraints)
            action_seed = cvar_action if cvar_action is not None else candidate_action
            safe, proj_meta = _projection_repair(action_seed, state, uncertainty_set, constraints, cfg)
            proj_meta["mode"] = "robust_resolve_cvar"
            proj_meta["robust_meta"] = cvar_meta
            proj_meta["seed_action"] = {
                "charge_mw": float(max(0.0, _f(action_seed.get("charge_mw"), 0.0))),
                "discharge_mw": float(max(0.0, _f(action_seed.get("discharge_mw"), 0.0))),
            }
            return safe, proj_meta

        if mode == "l2_projection":
            return _l2_projection_repair(candidate_action, state, uncertainty_set, constraints, cfg)

        # Greedy clipping fallback (legacy mode="projection")
        return _projection_repair(candidate_action, state, uncertainty_set, constraints, cfg)

    def get_metrics(self) -> dict[str, Callable[[Any], float]]:
        from orius.cpsbench_iot.metrics import (
            compute_control_metrics,
            summarize_true_soc_violations,
        )

        return {
            "true_soc_violations": summarize_true_soc_violations,
            "control_metrics": compute_control_metrics,
        }

    def get_oqe_features(self, telemetry: Any) -> dict[str, float]:
        from orius.dc3s.quality import compute_reliability

        w, flags = compute_reliability(telemetry, None)
        return {"w_t": float(w), **flags}
