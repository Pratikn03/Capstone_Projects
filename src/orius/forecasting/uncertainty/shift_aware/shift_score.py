from __future__ import annotations

from .state import ShiftAwareConfig, ShiftValidityState


def _component_goodness(x: float) -> float:
    return float(min(max(1.0 - x, 0.0), 1.0))


def compute_validity_score(
    *,
    reliability_score: float,
    drift_magnitude: float,
    normalized_residual: float,
    subgroup_under_coverage_gap: float,
    adaptation_instability: float,
    cfg: ShiftAwareConfig,
) -> ShiftValidityState:
    w = cfg.validity_weights
    residual_bad = min(max(normalized_residual, 0.0), 1.0)
    subgroup_bad = min(max(subgroup_under_coverage_gap / max(1.0 - cfg.coverage_target, 1e-6), 0.0), 1.0)
    drift_bad = min(max(drift_magnitude, 0.0), 1.0)
    adapt_bad = min(max(adaptation_instability, 0.0), 1.0)
    rel_bad = min(max(1.0 - reliability_score, 0.0), 1.0)

    score = (
        w["residual"] * _component_goodness(residual_bad)
        + w["subgroup"] * _component_goodness(subgroup_bad)
        + w["drift"] * _component_goodness(drift_bad)
        + w["adaptation"] * _component_goodness(adapt_bad)
        + w["reliability"] * _component_goodness(rel_bad)
    ) / max(sum(w.values()), 1e-12)
    score = float(min(max(score, 0.0), 1.0))

    status = "nominal"
    if score <= cfg.invalid_threshold:
        status = "invalid"
    elif score <= cfg.degraded_threshold:
        status = "degraded"
    elif score <= cfg.watch_threshold:
        status = "watch"

    return ShiftValidityState(
        validity_score=score,
        validity_status=status,
        normalized_residual=normalized_residual,
        subgroup_gap=subgroup_under_coverage_gap,
        drift_score=drift_magnitude,
        adaptation_instability=adaptation_instability,
        reliability_interaction=rel_bad,
        shift_alert_flag=status in {"degraded", "invalid"},
    )
