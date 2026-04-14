from __future__ import annotations

from .state import ShiftAwareConfig, ShiftValidityState


def _penalty(x: float) -> float:
    return min(1.0, max(0.0, float(x)))


def compute_validity_score(
    *,
    reliability_score: float,
    drift_magnitude: float,
    normalized_residual: float,
    under_coverage_gap: float,
    adaptation_instability: float,
    config: ShiftAwareConfig,
) -> ShiftValidityState:
    w = dict(config.validity_weights)
    penalties = {
        "reliability": _penalty(1.0 - reliability_score),
        "drift": _penalty(drift_magnitude),
        "residual": _penalty(normalized_residual),
        "subgroup": _penalty(under_coverage_gap),
        "adaptation": _penalty(adaptation_instability),
    }
    weighted_penalty = sum(float(w.get(k, 0.0)) * penalties[k] for k in penalties)
    denom = max(1e-9, sum(float(w.get(k, 0.0)) for k in penalties))
    validity = 1.0 - weighted_penalty / denom
    validity = min(1.0, max(0.0, validity))
    if validity <= config.invalid_threshold:
        status = "invalid"
    elif validity <= config.degraded_threshold:
        status = "degraded"
    elif validity <= config.watch_threshold:
        status = "watch"
    else:
        status = "nominal"

    component_scores = {f"{k}_score": 1.0 - penalties[k] for k in sorted(penalties)}
    return ShiftValidityState(
        validity_score=validity,
        validity_status=status,
        normalized_residual=normalized_residual,
        under_coverage_gap=under_coverage_gap,
        drift_magnitude=drift_magnitude,
        adaptation_instability=adaptation_instability,
        reliability_score=reliability_score,
        component_scores=component_scores,
    )
