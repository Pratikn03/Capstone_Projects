"""Domain runtime certificate-validity semantics for bounded T6/T11 rows."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class DomainValiditySemantics:
    validity_horizon_H_t: int
    half_life_steps: int
    expires_at_step: int
    validity_scope: str
    validity_theorem_id: str
    validity_theorem_contract: str
    guarantee_checks_passed: bool
    guarantee_fail_reasons: tuple[str, ...]

    def certificate_fields(self) -> dict[str, Any]:
        return {
            "validity_horizon_H_t": self.validity_horizon_H_t,
            "half_life_steps": self.half_life_steps,
            "expires_at_step": self.expires_at_step,
            "validity_scope": self.validity_scope,
            "validity_theorem_id": self.validity_theorem_id,
            "validity_theorem_contract": self.validity_theorem_contract,
            "guarantee_checks_passed": self.guarantee_checks_passed,
            "guarantee_fail_reasons": list(self.guarantee_fail_reasons),
        }


def _f(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _theorem_horizon(
    *,
    margin: float,
    sigma_d: float,
    delta: float,
    max_horizon: int,
) -> int:
    if margin <= 0.0 or sigma_d <= 0.0 or not (0.0 < delta < 1.0):
        return 0
    raw = math.floor((margin**2) / (2.0 * (sigma_d**2) * math.log(2.0 / delta)))
    return int(max(0, min(int(max_horizon), raw)))


def _healthcare_margin(uncertainty: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    hc = cfg.get("healthcare", {}) if isinstance(cfg.get("healthcare"), Mapping) else {}
    spo2_min = _f(cfg.get("spo2_min_pct", hc.get("spo2_min_pct")), 90.0)
    hr_min = _f(cfg.get("hr_min_bpm", hc.get("hr_min_bpm")), 40.0)
    hr_max = _f(cfg.get("hr_max_bpm", hc.get("hr_max_bpm")), 120.0)
    rr_min = _f(cfg.get("rr_min", hc.get("rr_min")), 8.0)
    rr_max = _f(cfg.get("rr_max", hc.get("rr_max")), 30.0)
    return float(
        min(
            _f(uncertainty.get("spo2_lower_pct"), spo2_min) - spo2_min,
            _f(uncertainty.get("forecast_spo2_lower_pct"), spo2_min) - spo2_min,
            _f(uncertainty.get("hr_lower_bpm"), hr_min) - hr_min,
            hr_max - _f(uncertainty.get("hr_upper_bpm"), hr_max),
            _f(uncertainty.get("rr_lower"), rr_min) - rr_min,
            rr_max - _f(uncertainty.get("rr_upper"), rr_max),
        )
    )


def _av_margin(uncertainty: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    raw_true_margin = cfg.get("true_margin")
    if raw_true_margin not in (None, ""):
        parsed_true_margin = _f(raw_true_margin, float("nan"))
        if math.isfinite(parsed_true_margin):
            return parsed_true_margin
    min_headway = _f(cfg.get("min_headway_m"), 8.0)
    ttc_min = _f(cfg.get("ttc_min_s"), 3.0)
    gap_margin = _f(uncertainty.get("gap_lower_m"), min_headway) - min_headway
    ttc = _f(uncertainty.get("current_ttc_s"), float("inf"))
    ttc_margin = 1e6 if not math.isfinite(ttc) else ttc - ttc_min
    return float(min(gap_margin, ttc_margin))


def _fallback_action_satisfies(domain: str, safe_action: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    if domain == "healthcare":
        return _f(safe_action.get("alert_level"), 0.0) >= 0.999
    if domain == "av":
        brake_accel = _f(cfg.get("fallback_accel_mps2"), -6.0)
        return _f(safe_action.get("acceleration_mps2"), 0.0) <= brake_accel + 1e-9
    return False


def _projected_release_satisfies(repair: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    if not bool(repair.get("projected_release", False)):
        return False
    if not bool(repair.get("allow_single_step_projected_validity", False)):
        return False
    margin = _f(repair.get("projected_release_margin", cfg.get("projected_release_margin")), 0.0)
    return margin > 0.0


def _av_near_failsafe_projected_release_satisfies(
    safe_action: Mapping[str, Any],
    repair: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> bool:
    if not bool(repair.get("projected_release", False)):
        return False
    if not bool(repair.get("allow_near_failsafe_projected_validity", False)):
        return False
    brake_accel = _f(cfg.get("fallback_accel_mps2"), -6.0)
    epsilon = max(0.0, _f(cfg.get("near_failsafe_projection_epsilon_mps2"), 0.25))
    accel = _f(safe_action.get("acceleration_mps2"), 0.0)
    return accel <= brake_accel + epsilon


def domain_certificate_validity_semantics(
    *,
    domain: str,
    safe_action: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    reliability_w: float,
    validity_status: str,
    step_index: int,
    repair_meta: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> DomainValiditySemantics:
    """Return theorem-facing validity metadata for promoted domain certificates.

    Normal hold releases use a T6-style subgaussian first-passage lower bound
    when the domain margin is positive. Fallback releases are valid for one
    step only when the emitted action is the domain fail-safe action.
    """

    domain_key = "av" if str(domain).lower() in {"av", "vehicle", "orius_av"} else "healthcare"
    repair = dict(repair_meta or {})
    fallback_required = bool(repair.get("fallback_required", False)) or repair.get("mode") == "fallback"
    delta = _f(cfg.get("validity_delta"), 0.05)
    sigma_d = _f(cfg.get("validity_sigma_d"), 1.0)
    max_horizon = int(_f(cfg.get("max_validity_horizon_steps"), 10 if domain_key == "av" else 8))

    if fallback_required:
        if _fallback_action_satisfies(domain_key, safe_action, cfg):
            return DomainValiditySemantics(
                validity_horizon_H_t=1,
                half_life_steps=1,
                expires_at_step=int(step_index) + 1,
                validity_scope="single_step_fallback",
                validity_theorem_id="T6_T11_domain_fallback",
                validity_theorem_contract="bounded_fail_safe_release",
                guarantee_checks_passed=True,
                guarantee_fail_reasons=(),
            )
        return DomainValiditySemantics(
            validity_horizon_H_t=0,
            half_life_steps=0,
            expires_at_step=int(step_index),
            validity_scope="invalid_fallback",
            validity_theorem_id="T6_T11_domain_fallback",
            validity_theorem_contract="bounded_fail_safe_release",
            guarantee_checks_passed=False,
            guarantee_fail_reasons=("fallback_action_not_fail_safe",),
        )

    if domain_key == "av" and _av_near_failsafe_projected_release_satisfies(safe_action, repair, cfg):
        return DomainValiditySemantics(
            validity_horizon_H_t=1,
            half_life_steps=1,
            expires_at_step=int(step_index) + 1,
            validity_scope="single_step_projected_release",
            validity_theorem_id="T6_T11_domain_projected_release",
            validity_theorem_contract="bounded_projected_safe_release",
            guarantee_checks_passed=True,
            guarantee_fail_reasons=(),
        )

    if _projected_release_satisfies(repair, cfg):
        return DomainValiditySemantics(
            validity_horizon_H_t=1,
            half_life_steps=1,
            expires_at_step=int(step_index) + 1,
            validity_scope="single_step_projected_release",
            validity_theorem_id="T6_T11_domain_projected_release",
            validity_theorem_contract="bounded_projected_safe_release",
            guarantee_checks_passed=True,
            guarantee_fail_reasons=(),
        )

    if validity_status in {"invalid", "degraded"} or reliability_w <= 0.0:
        return DomainValiditySemantics(
            validity_horizon_H_t=0,
            half_life_steps=0,
            expires_at_step=int(step_index),
            validity_scope="invalid_hold",
            validity_theorem_id="T6",
            validity_theorem_contract="closed_form_first_passage_lower_bound",
            guarantee_checks_passed=False,
            guarantee_fail_reasons=("non_fallback_degraded_validity",),
        )

    margin = _av_margin(uncertainty, cfg) if domain_key == "av" else _healthcare_margin(uncertainty, cfg)
    horizon = _theorem_horizon(
        margin=margin,
        sigma_d=sigma_d,
        delta=delta,
        max_horizon=max_horizon,
    )
    if horizon <= 0:
        return DomainValiditySemantics(
            validity_horizon_H_t=0,
            half_life_steps=0,
            expires_at_step=int(step_index),
            validity_scope="invalid_hold",
            validity_theorem_id="T6",
            validity_theorem_contract="closed_form_first_passage_lower_bound",
            guarantee_checks_passed=False,
            guarantee_fail_reasons=("nonpositive_domain_margin",),
        )
    return DomainValiditySemantics(
        validity_horizon_H_t=horizon,
        half_life_steps=max(1, horizon // 2),
        expires_at_step=int(step_index) + horizon,
        validity_scope="multi_step_hold",
        validity_theorem_id="T6",
        validity_theorem_contract="closed_form_first_passage_lower_bound",
        guarantee_checks_passed=True,
        guarantee_fail_reasons=(),
    )


__all__ = ["DomainValiditySemantics", "domain_certificate_validity_semantics"]
