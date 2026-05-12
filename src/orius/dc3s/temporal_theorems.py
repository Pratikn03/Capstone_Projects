"""Backward-compatible imports for battery-specific temporal helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from orius.universal_theory.battery_instantiation import (
    certificate_expiration_bound,
    certificate_half_life,
    certificate_validity_horizon as _battery_certificate_validity_horizon,
    evaluate_graceful_degradation_dominance,
    forward_tube,
    should_expire_certificate,
    should_renew_certificate,
    zero_dispatch_fallback,
)
from orius.universal_theory.battery_instantiation import (
    validate_battery_fallback as certify_fallback_existence,
)


def certificate_validity_horizon(
    *,
    interval_lower_mwh: float,
    interval_upper_mwh: float,
    safe_action: Mapping[str, Any],
    constraints: Mapping[str, Any],
    sigma_d: float,
    max_steps: int = 4096,
) -> dict[str, float | int]:
    """Battery-compatible T5 finite-horizon certificate helper."""

    return _battery_certificate_validity_horizon(
        interval_lower_mwh=interval_lower_mwh,
        interval_upper_mwh=interval_upper_mwh,
        safe_action=safe_action,
        constraints=constraints,
        sigma_d=sigma_d,
        max_steps=max_steps,
    )


def forward_reachable_tube(
    initial_set: Sequence[tuple[float, float]],
    action_deltas: Sequence[float],
    *,
    drift_radius_per_step: float = 0.0,
) -> list[tuple[float, float]]:
    """Compute a scalar interval reachable tube for T5-style certificates.

    The generic helper is intentionally simple and domain-neutral: each state
    coordinate is represented by an interval, actions are additive deltas, and
    bounded drift expands the radius linearly with the step index.
    """

    if drift_radius_per_step < 0.0:
        raise ValueError("drift_radius_per_step must be non-negative.")
    lower = min(float(lo) for lo, _ in initial_set)
    upper = max(float(hi) for _, hi in initial_set)
    tube = [(lower, upper)]
    cumulative = 0.0
    for step, delta in enumerate(action_deltas, start=1):
        cumulative += float(delta)
        radius = float(drift_radius_per_step) * step
        tube.append((lower + cumulative - radius, upper + cumulative + radius))
    return tube


def validate_certificate_horizon(
    reachable_tube: Sequence[tuple[float, float]],
    *,
    safe_lower: float,
    safe_upper: float,
) -> dict[str, int | bool]:
    """Return the largest prefix horizon whose tube remains inside the safe set."""

    if safe_lower > safe_upper:
        raise ValueError("safe_lower must be <= safe_upper.")
    horizon = -1
    for idx, (lower, upper) in enumerate(reachable_tube):
        if float(lower) < float(safe_lower) or float(upper) > float(safe_upper):
            break
        horizon = idx
    return {
        "valid": bool(horizon >= 0),
        "horizon": max(0, horizon),
        "fails_closed": bool(horizon < 1),
    }


def certificate_invalidating_event(
    *,
    contradictory_observation: bool = False,
    model_version_changed: bool = False,
    action_sequence_changed: bool = False,
    reliability_below_floor: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Detect certificate-invalidating events named by the T5 claim boundary."""

    reasons = []
    if contradictory_observation:
        reasons.append("contradictory_observation")
    if model_version_changed:
        reasons.append("model_version_changed")
    if action_sequence_changed:
        reasons.append("action_sequence_changed")
    if reliability_below_floor:
        reasons.append("reliability_below_floor")
    return {
        "invalidates_certificate": bool(reasons),
        "reasons": reasons,
        "metadata": dict(metadata or {}),
    }


__all__ = [
    "certificate_expiration_bound",
    "certificate_half_life",
    "certificate_invalidating_event",
    "certificate_validity_horizon",
    "certify_fallback_existence",
    "evaluate_graceful_degradation_dominance",
    "forward_reachable_tube",
    "forward_tube",
    "should_expire_certificate",
    "should_renew_certificate",
    "validate_certificate_horizon",
    "zero_dispatch_fallback",
]
