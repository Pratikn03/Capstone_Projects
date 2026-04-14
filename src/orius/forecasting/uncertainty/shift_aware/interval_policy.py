from __future__ import annotations

from .state import ShiftAwareConfig, ShiftAwareIntervalDecision, ShiftValidityState


def apply_interval_policy(
    *,
    y_hat: float,
    base_half_width: float,
    reliability_score: float,
    drift_signal: float,
    adaptive_quantile: float,
    under_coverage_gap: float,
    validity: ShiftValidityState,
    policy_name: str,
    config: ShiftAwareConfig,
    coverage_group_key: str = "unknown",
) -> ShiftAwareIntervalDecision:
    bw = max(0.0, float(base_half_width))
    rel_bad = max(0.0, 1.0 - float(reliability_score))
    drift_bad = max(0.0, float(drift_signal))
    validity_bad = max(0.0, 1.0 - float(validity.validity_score))
    subgroup_bad = max(0.0, float(under_coverage_gap))

    if policy_name == "legacy_rac_cert":
        inflation = max(1.0, float(adaptive_quantile))
    elif policy_name == "shift_aware_piecewise":
        inflation = 1.0 + max(rel_bad, validity_bad, drift_bad, subgroup_bad) * 2.0
        inflation *= max(1.0, float(adaptive_quantile))
    elif policy_name == "shift_aware_mondrian":
        inflation = 1.0 + 1.5 * subgroup_bad + 1.2 * rel_bad + drift_bad + validity_bad
        inflation *= max(1.0, float(adaptive_quantile))
    else:  # shift_aware_linear
        inflation = (
            1.0
            + config.linear_reliability_gain * rel_bad
            + config.linear_validity_gain * validity_bad
            + config.linear_drift_gain * drift_bad
            + config.linear_under_coverage_gain * subgroup_bad
        )
        inflation *= max(1.0, float(adaptive_quantile))

    inflation = min(config.widening_cap, max(1.0, float(inflation)))
    half = bw * inflation
    lower = float(y_hat) - half
    upper = float(y_hat) + half
    return ShiftAwareIntervalDecision(
        lower=lower,
        upper=upper,
        base_half_width=bw,
        adjusted_half_width=half,
        inflation_multiplier=inflation,
        adaptive_quantile=float(adaptive_quantile),
        validity_score=float(validity.validity_score),
        validity_status=validity.validity_status,
        under_coverage_gap=float(under_coverage_gap),
        applied_policy=policy_name,
        coverage_group_key=coverage_group_key,
        shift_alert_flag=validity.validity_status in {"degraded", "invalid"},
    )
