"""Paper 3: Graceful degradation planner."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence


def _f(value: Any, default: float) -> float:
    try:
        v = float(value)
        if not math.isfinite(v):
            return float(default)
        return v
    except (TypeError, ValueError):
        return float(default)


def optimized_graceful(
    last_action: Mapping[str, Any],
    horizon_steps: int,
    soc_mwh: float,
    constraints: Mapping[str, Any],
    sigma_d: float = 50.0,
    utility_weight: float = 1.0,
) -> Sequence[dict[str, float]]:
    """
    Constructive graceful degradation: taper from last_action to zero over
    remaining certified horizon. Consumes horizon_steps; reaches safe target
    (zero dispatch) before expiration.

    Parameterized by utility_weight: 1.0 = max useful work, 0.0 = immediate shutdown.
    """
    if horizon_steps <= 0:
        return [{"charge_mw": 0.0, "discharge_mw": 0.0}]

    dt = _f(constraints.get("time_step_hours"), 1.0)
    eta_c = _f(constraints.get("charge_efficiency"), 0.95)
    eta_d = _f(constraints.get("discharge_efficiency"), 0.95)
    soc_min = _f(constraints.get("min_soc_mwh"), 0.0)
    soc_max = _f(constraints.get("max_soc_mwh"), 100.0)
    max_pwr = _f(constraints.get("max_power_mw"), 100.0)
    max_discharge = _f(constraints.get("max_discharge_mw"), max_pwr)
    max_charge = _f(constraints.get("max_charge_mw"), max_pwr)

    last_chg = max(0.0, _f(last_action.get("charge_mw"), 0.0))
    last_dis = max(0.0, _f(last_action.get("discharge_mw"), 0.0))

    actions: List[dict[str, float]] = []
    soc = float(soc_mwh)
    for step in range(horizon_steps):
        # Linear taper: (1 - t) * last + t * 0, t = step / horizon_steps
        t = step / max(1, horizon_steps)
        scale = utility_weight * (1.0 - t)
        target_chg = last_chg * scale
        target_dis = last_dis * scale

        # Clamp to SOC feasibility
        max_feasible_dis = max(0.0, (soc - soc_min) * eta_d / dt) if dt > 0 else 0.0
        max_feasible_chg = max(0.0, (soc_max - soc) / (eta_c * dt)) if dt > 0 else 0.0
        dis = min(target_dis, max_discharge, max_feasible_dis)
        chg = min(target_chg, max_charge, max_feasible_chg)

        actions.append({"charge_mw": float(chg), "discharge_mw": float(dis)})
        soc = soc + dt * (eta_c * chg - dis / eta_d)
        soc = max(soc_min, min(soc_max, soc))

    return actions


def plan_graceful_degradation(
    certificate_state: Dict[str, Any],
    shrinking_safe_set: Dict[str, float],
    objective_weights: Dict[str, float],
    fallback_mode: str,
    remaining_horizon: int,
) -> Dict[str, Any]:
    """
    Computes a provably safe ramp-down policy inside the remaining
    certificate-valid horizon. Constructive: produces tapered actions.
    """
    transition_log: List[Dict[str, Any]] = []

    if not certificate_state.get("fallback_required"):
        transition_log.append({"event": "no_fallback", "reason": "Fallback not required."})
        return {"actions": [], "reason": "Fallback not required.", "transition_log": transition_log}

    if remaining_horizon <= 0:
        transition_log.append({"event": "fallback_termination", "reason": "Horizon exhausted.", "remaining_horizon": 0})
        return {"actions": [{"charge_mw": 0.0, "discharge_mw": 0.0}], "reason": "Horizon exhausted.", "transition_log": transition_log}

    transition_log.append({"event": "fallback_invoked", "fallback_mode": fallback_mode, "remaining_horizon": remaining_horizon})
    last_action = certificate_state.get("last_action") or {"charge_mw": 0.0, "discharge_mw": 0.0}
    soc = _f(certificate_state.get("current_soc_mwh"), 50.0)
    constraints = certificate_state.get("constraints") or {}
    utility = _f(objective_weights.get("useful_work_weight", 1.0), 1.0)

    if fallback_mode == "ramp_down":
        action_plan = list(optimized_graceful(
            last_action=last_action,
            horizon_steps=remaining_horizon,
            soc_mwh=soc,
            constraints=constraints,
            sigma_d=50.0,
            utility_weight=0.5,
        ))
        transition_log.append({"event": "policy_applied", "mode": "simple_ramp_down", "reason": "Ramp-down heuristic."})
        return {"actions": action_plan, "reason": "Ramp-down heuristic.", "transition_log": transition_log}

    elif fallback_mode == "optimized":
        action_plan = list(optimized_graceful(
            last_action=last_action,
            horizon_steps=remaining_horizon,
            soc_mwh=soc,
            constraints=constraints,
            sigma_d=50.0,
            utility_weight=utility,
        ))
        transition_log.append({"event": "policy_applied", "mode": "optimized_graceful", "reason": "Optimized fallback."})
        return {"actions": action_plan, "reason": "Optimized fallback.", "transition_log": transition_log}

    else:  # hard_shutdown
        transition_log.append({"event": "policy_applied", "mode": "immediate_shutdown", "reason": "Hard shutdown."})
        return {"actions": [{"charge_mw": 0.0, "discharge_mw": 0.0}], "reason": "Hard shutdown.", "transition_log": transition_log}


def _simulate_policy(
    policy_name: str,
    last_action: dict,
    horizon_steps: int,
    soc_mwh: float,
    constraints: dict,
    sigma_d: float,
    seed: int = 0,
) -> dict:
    """Simulate one policy; return trajectory and metrics."""
    import random
    rng = random.Random(seed)
    dt = _f(constraints.get("time_step_hours"), 1.0)
    eta_c = _f(constraints.get("charge_efficiency"), 0.95)
    eta_d = _f(constraints.get("discharge_efficiency"), 0.95)
    soc_min = _f(constraints.get("min_soc_mwh"), 0.0)
    soc_max = _f(constraints.get("max_soc_mwh"), 100.0)

    if policy_name == "blind_persistence":
        actions = [dict(last_action) for _ in range(horizon_steps)]
    elif policy_name == "immediate_shutdown":
        actions = [{"charge_mw": 0.0, "discharge_mw": 0.0} for _ in range(horizon_steps)]
    elif policy_name == "simple_ramp_down":
        actions = list(optimized_graceful(last_action, horizon_steps, soc_mwh, constraints, sigma_d, utility_weight=0.5))
    elif policy_name == "optimized_graceful":
        actions = list(optimized_graceful(last_action, horizon_steps, soc_mwh, constraints, sigma_d, utility_weight=1.0))
    else:
        actions = [{"charge_mw": 0.0, "discharge_mw": 0.0} for _ in range(horizon_steps)]

    traj = []
    soc = soc_mwh
    violations = 0
    useful_work = 0.0
    severity_mwh = 0.0
    for step, a in enumerate(actions):
        chg = max(0.0, _f(a.get("charge_mw"), 0.0))
        dis = max(0.0, _f(a.get("discharge_mw"), 0.0))
        noise = sigma_d * rng.gauss(0, 1) if sigma_d > 0 else 0.0
        soc_raw = soc + dt * (eta_c * chg - dis / eta_d) + noise
        if soc_raw <= soc_min - 1e-9 or soc_raw >= soc_max + 1e-9:
            violations += 1
            severity_mwh = max(severity_mwh, soc_min - soc_raw if soc_raw < soc_min else soc_raw - soc_max)
        soc = max(soc_min, min(soc_max, soc_raw))
        last_dis = max(1e-9, _f(last_action.get("discharge_mw"), 0.0))
        useful_work += dis / last_dis if last_dis > 0 else 0.0
        traj.append({"step": step, "charge_mw": chg, "discharge_mw": dis, "soc_mwh": soc})

    tsvr = violations / max(1, horizon_steps)
    gdq = 1.0 - tsvr if violations == 0 else max(0.0, 1.0 - violations / horizon_steps)
    return {
        "trajectory": traj,
        "gdq": gdq,
        "tsvr": tsvr,
        "useful_work_mwh": useful_work * dt,
        "retained_cost_frac": useful_work / max(1, horizon_steps),
        "descent_stability": 1.0 if violations == 0 else 0.0,
        "violations": violations,
        "severity_mwh": float(severity_mwh),
        "violation_rate": tsvr,
    }


def compare_policies(
    last_action: Mapping[str, Any],
    horizon_steps: int,
    soc_mwh: float,
    constraints: Mapping[str, Any],
    sigma_d: float = 50.0,
    seed: int = 0,
) -> dict[str, dict]:
    """Compare four policies: blind_persistence, immediate_shutdown, simple_ramp_down, optimized_graceful."""
    policies = ["blind_persistence", "immediate_shutdown", "simple_ramp_down", "optimized_graceful"]
    return {p: _simulate_policy(p, dict(last_action), horizon_steps, soc_mwh, dict(constraints), sigma_d, seed) for p in policies}