"""Executable witnesses for observation-ambiguity necessity and optimality.

The helpers in this module intentionally implement the corrected theorem
surface: differing safe sets are not enough for an impossibility claim.  The
relevant object is the common safe core of an observation ambiguity class.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any

State = Hashable
Action = Hashable
Observation = Hashable


def _state_set(states: Iterable[State], *, name: str) -> tuple[State, ...]:
    values = tuple(states)
    if not values:
        raise ValueError(f"{name} must contain at least one state.")
    return values


def _action_set(actions: Iterable[Action], *, name: str) -> frozenset[Action]:
    values = frozenset(actions)
    if not values:
        raise ValueError(f"{name} must contain at least one action.")
    return values


def _safe_actions_for(state: State, safe_action_sets: Mapping[State, Iterable[Action]]) -> frozenset[Action]:
    if state not in safe_action_sets:
        raise KeyError(f"Missing safe-action set for state {state!r}.")
    return _action_set(safe_action_sets[state], name=f"safe_action_sets[{state!r}]")


def common_safe_core(
    ambiguity_class: Iterable[State],
    safe_action_sets: Mapping[State, Iterable[Action]],
) -> frozenset[Action]:
    """Return the actions safe for every state in an ambiguity class.

    This is the corrected impossibility object.  If two states have different
    safe sets but their intersection is non-empty, an observation-only policy
    can still choose a common safe action for that class.
    """
    states = _state_set(ambiguity_class, name="ambiguity_class")
    core = set(_safe_actions_for(states[0], safe_action_sets))
    for state in states[1:]:
        core.intersection_update(_safe_actions_for(state, safe_action_sets))
    return frozenset(core)


def observation_only_bayes_lower_bound(
    observation_groups: Mapping[Observation, Iterable[State]],
    action_space: Iterable[Action],
    safe_action_sets: Mapping[State, Iterable[Action]],
    probabilities: Mapping[State, float],
) -> dict[str, Any]:
    """Compute the Bayes lower bound for an observation-only controller.

    For each observation class B(o), the best observation-only release action
    has conditional violation risk min_a P[a notin C(X) | O=o].  Averaging over
    observations gives a lower bound on every policy pi(O).
    """
    actions = _action_set(action_space, name="action_space")
    if not observation_groups:
        raise ValueError("observation_groups must be non-empty.")

    per_observation: list[dict[str, Any]] = []
    total_mass = 0.0
    weighted_risk = 0.0

    for observation, raw_states in observation_groups.items():
        states = _state_set(raw_states, name=f"observation_groups[{observation!r}]")
        masses = []
        for state in states:
            if state not in probabilities:
                raise KeyError(f"Missing probability for state {state!r}.")
            mass = float(probabilities[state])
            if mass < 0.0:
                raise ValueError("probabilities must be non-negative.")
            masses.append(mass)
            _safe_actions_for(state, safe_action_sets)

        group_mass = float(sum(masses))
        if group_mass <= 0.0:
            raise ValueError(f"Observation class {observation!r} has zero probability mass.")

        action_risks: dict[Action, float] = {}
        for action in actions:
            unsafe_mass = 0.0
            for state, mass in zip(states, masses, strict=True):
                if action not in _safe_actions_for(state, safe_action_sets):
                    unsafe_mass += mass
            action_risks[action] = float(unsafe_mass / group_mass)

        best_action = min(action_risks, key=action_risks.__getitem__)
        best_risk = float(action_risks[best_action])
        core = common_safe_core(states, safe_action_sets)
        total_mass += group_mass
        weighted_risk += group_mass * best_risk
        per_observation.append(
            {
                "observation": observation,
                "states": list(states),
                "probability_mass": group_mass,
                "common_safe_core": sorted(core, key=repr),
                "common_safe_core_empty": len(core) == 0,
                "best_observation_only_action": best_action,
                "best_observation_only_risk": best_risk,
                "action_risks": dict(sorted(action_risks.items(), key=lambda item: repr(item[0]))),
            }
        )

    if total_mass <= 0.0:
        raise ValueError("Total probability mass must be positive.")

    return {
        "theorem_id": "T10_T11_ObservationAmbiguitySandwich",
        "lower_bound": float(weighted_risk / total_mass),
        "total_probability_mass": total_mass,
        "per_observation": per_observation,
        "interpretation": (
            "Observation-only controllers are lower-bounded by the Bayes risk "
            "inside each ambiguity class; differing safe sets alone are not "
            "sufficient for an impossibility claim."
        ),
    }


def verify_covered_orius_release(
    true_state: State,
    uncertainty_set: Iterable[State],
    action: Action,
    safe_action_sets: Mapping[State, Iterable[Action]],
    *,
    coverage_miss_probability: float = 0.0,
) -> dict[str, Any]:
    """Verify the ORIUS upper-bound side for a single covered release."""
    states = _state_set(uncertainty_set, name="uncertainty_set")
    alpha = float(coverage_miss_probability)
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("coverage_miss_probability must lie in [0, 1].")

    covered = true_state in states
    safe_for_uncertainty = all(action in _safe_actions_for(state, safe_action_sets) for state in states)
    safe_for_true_state = action in _safe_actions_for(true_state, safe_action_sets)
    deterministic_zero_violation = bool(covered and safe_for_uncertainty)
    probabilistic_bound = alpha if safe_for_uncertainty else None

    return {
        "theorem_id": "T10_T11_ObservationAmbiguitySandwich",
        "true_state_covered": bool(covered),
        "action_safe_for_uncertainty_set": bool(safe_for_uncertainty),
        "action_safe_for_true_state": bool(safe_for_true_state),
        "deterministic_zero_violation_certified": deterministic_zero_violation,
        "coverage_miss_probability": alpha,
        "violation_probability_upper_bound": probabilistic_bound,
        "status": "certified" if deterministic_zero_violation else "not_certified",
        "scope_note": (
            "Zero violation is certified only under actual coverage; with a "
            "probabilistic coverage contract, the bound is alpha."
        ),
    }


def build_observation_ambiguity_contract_summary(
    *,
    observation_groups: Mapping[Observation, Iterable[State]],
    action_space: Iterable[Action],
    safe_action_sets: Mapping[State, Iterable[Action]],
    probabilities: Mapping[State, float],
    true_state: State,
    uncertainty_set: Iterable[State],
    action: Action,
    coverage_miss_probability: float = 0.0,
) -> dict[str, Any]:
    """Build the publication-facing executable summary for the corollary."""
    lower = observation_only_bayes_lower_bound(
        observation_groups=observation_groups,
        action_space=action_space,
        safe_action_sets=safe_action_sets,
        probabilities=probabilities,
    )
    upper = verify_covered_orius_release(
        true_state=true_state,
        uncertainty_set=uncertainty_set,
        action=action,
        safe_action_sets=safe_action_sets,
        coverage_miss_probability=coverage_miss_probability,
    )
    passed = bool(
        lower["lower_bound"] >= 0.0
        and upper["action_safe_for_uncertainty_set"]
        and upper["violation_probability_upper_bound"] is not None
    )
    return {
        "theorem_id": "T10_T11_ObservationAmbiguitySandwich",
        "source_theorems": ["T10", "T11"],
        "theorem_type": "supporting_optimality_corollary",
        "lower_bound": lower,
        "orius_upper_bound": upper,
        "all_executable_checks_passed": passed,
        "status": "runtime_linked" if passed else "contract_violation",
        "claim_boundary": (
            "Safety-optimal under covered observation ambiguity; not a global "
            "optimality theorem for every physical-AI system."
        ),
    }


__all__ = [
    "build_observation_ambiguity_contract_summary",
    "common_safe_core",
    "observation_only_bayes_lower_bound",
    "verify_covered_orius_release",
]
