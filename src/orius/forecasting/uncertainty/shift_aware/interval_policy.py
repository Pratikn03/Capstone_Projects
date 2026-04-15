from __future__ import annotations

from .state import ShiftAwareConfig, ShiftAwareIntervalDecision, ShiftValidityState


def _severity_multiplier(
    *,
    reliability_score: float,
    drift_signal: float,
    under_coverage_gap: float,
    validity_score: float,
    adaptive_alpha: float,
    policy: str,
) -> float:
    rel_bad = max(0.0, 1.0 - reliability_score)
    valid_bad = max(0.0, 1.0 - validity_score)
    drift_bad = max(0.0, drift_signal)
    uc_bad = max(0.0, under_coverage_gap)
    alpha_bad = max(0.0, adaptive_alpha)

    if policy == "shift_aware_piecewise":
        severe = 1.0 if (valid_bad > 0.55 or drift_bad > 0.5 or uc_bad > 0.05) else 0.0
        return 1.0 + 0.8 * rel_bad + 0.9 * valid_bad + 0.7 * drift_bad + 1.2 * uc_bad + 0.4 * severe
    if policy == "shift_aware_mondrian":
        return 1.0 + 0.7 * rel_bad + 0.8 * valid_bad + 0.6 * drift_bad + 1.5 * uc_bad + 0.2 * alpha_bad
    return 1.0 + 0.8 * rel_bad + 0.9 * valid_bad + 0.7 * drift_bad + 1.0 * uc_bad + 0.3 * alpha_bad


def apply_interval_policy(
    *,
    y_hat: float,
    base_half_width: float,
    reliability_score: float,
    drift_signal: float,
    adaptive_quantile: float,
    subgroup_under_coverage_gap: float,
    validity: ShiftValidityState,
    cfg: ShiftAwareConfig,
    coverage_group_key: str = "global",
) -> ShiftAwareIntervalDecision:
    policy = cfg.policy_mode
    m = _severity_multiplier(
        reliability_score=reliability_score,
        drift_signal=drift_signal,
        under_coverage_gap=subgroup_under_coverage_gap,
        validity_score=validity.validity_score,
        adaptive_alpha=adaptive_quantile,
        policy=policy,
    )
    m = float(min(max(m, 1.0), cfg.max_inflation_multiplier))
    adjusted = float(max(base_half_width, base_half_width * m))
    return ShiftAwareIntervalDecision(
        lower=float(y_hat - adjusted),
        upper=float(y_hat + adjusted),
        base_half_width=float(base_half_width),
        adjusted_half_width=adjusted,
        inflation_multiplier=float(max(adjusted / max(base_half_width, 1e-9), 1.0)),
        adaptive_quantile=float(adaptive_quantile),
        validity_score=float(validity.validity_score),
        validity_status=validity.validity_status,
        under_coverage_gap=float(subgroup_under_coverage_gap),
        applied_policy=str(policy),
        coverage_group_key=coverage_group_key,
        shift_alert_flag=bool(validity.shift_alert_flag),
    )
