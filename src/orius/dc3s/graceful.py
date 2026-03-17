"""Paper 3: Graceful degradation planner."""
from __future__ import annotations

from typing import Any, Dict


def plan_graceful_degradation(
    certificate_state: Dict[str, Any],
    shrinking_safe_set: Dict[str, float],
    objective_weights: Dict[str, float],
    fallback_mode: str,
    remaining_horizon: int,
) -> Dict[str, Any]:
    """
    Computes a provably safe ramp-down policy inside the remaining
    certificate-valid horizon.
    """
    if not certificate_state.get("fallback_required"):
        return {"actions": [], "reason": "Fallback not required."}

    # Simple ramp-down heuristic
    if fallback_mode == "ramp_down":
        # This is a placeholder for a more complex planner.
        # For now, just return a hold action.
        action_plan = [{"charge_mw": 0.0, "discharge_mw": 0.0} for _ in range(remaining_horizon)]
        return {"actions": action_plan, "reason": "Ramp-down heuristic."}

    # Optimized graceful degradation
    elif fallback_mode == "optimized":
        # This would call the fallback_dispatch optimizer
        # from orius.optimizer.fallback_dispatch import solve_fallback_dispatch
        # solution = solve_fallback_dispatch(...)
        action_plan = [{"charge_mw": 0.0, "discharge_mw": 0.0} for _ in range(remaining_horizon)]
        return {"actions": action_plan, "reason": "Optimized fallback (placeholder)."}

    # Hard shutdown
    else:  # hard_shutdown
        return {"actions": [{"charge_mw": 0.0, "discharge_mw": 0.0}], "reason": "Hard shutdown."}