"""Healthcare domain adapter for ORIUS Universal Framework.

Vital signs: HR, SpO2, respiratory rate. Safety predicates:
- hr_bpm in [hr_min, hr_max]
- spo2_pct >= spo2_min
- respiratory_rate in [rr_min, rr_max]
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from orius.dc3s.domain_adapter import DomainAdapter
from orius.dc3s.certificate import make_certificate, recompute_certificate_hash
from orius.universal_theory.domain_validity import domain_certificate_validity_semantics
from orius.universal_framework.reliability_runtime import assess_domain_reliability
from orius.universal_framework.runtime_evidence import resolve_runtime_evidence


def _f(x: Any, default: float) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return float(default)
        return v
    except (TypeError, ValueError):
        return float(default)


def _validity_status(reliability_w: float, drift_flag: bool) -> str:
    if reliability_w < 0.20:
        return "invalid"
    if drift_flag or reliability_w < 0.45:
        return "degraded"
    if reliability_w < 0.75:
        return "watch"
    return "nominal"


class HealthcareDomainAdapter(DomainAdapter):
    """DomainAdapter for healthcare vital signs monitoring."""

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        self._cfg = dict(cfg or {})
        self.domain_id = "healthcare"
        hc = self._cfg.get("healthcare", {})
        self._hr_min = _f(hc.get("hr_min_bpm"), 40.0)
        self._hr_max = _f(hc.get("hr_max_bpm"), 120.0)
        self._spo2_min = _f(hc.get("spo2_min_pct"), 90.0)
        self._rr_min = _f(hc.get("rr_min"), 8.0)
        self._rr_max = _f(hc.get("rr_max"), 30.0)
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 1.0)
        evidence = resolve_runtime_evidence(self.domain_id, self._cfg)
        self._runtime_surface = evidence.runtime_surface
        self._closure_tier = evidence.closure_tier
        self._maturity_tier = evidence.maturity_tier
        self._fallback_policy = evidence.fallback_policy
        self._exact_blocker = evidence.exact_blocker

    def capability_profile(self) -> Mapping[str, Any]:
        return {
            "safety_surface_type": "vital_alert_envelope",
            "repair_mode": "piecewise_hold_or_max_alert_release",
            "fallback_mode": "max_alert",
            "supports_multi_agent_eval": False,
            "supports_certos_eval": True,
            "runtime_surface": self._runtime_surface,
            "closure_tier": self._closure_tier,
            "maturity_tier": self._maturity_tier,
            "fallback_policy": self._fallback_policy,
            "exact_blocker": self._exact_blocker,
        }

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        """Parse raw vital signs packet into state vector z_t."""
        out = {}
        missing_fields: list[str] = []
        held_fields: list[str] = []
        for k in (
            "hr_bpm",
            "spo2_pct",
            "respiratory_rate",
            "reliability",
            "forecast_spo2_pct",
            "patient_id",
            "ts_utc",
        ):
            v = raw_packet.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                missing_fields.append(k)
                hold_value = raw_packet.get(f"_hold_{k}", 0.0 if k != "ts_utc" else "")
                if f"_hold_{k}" in raw_packet:
                    held_fields.append(k)
                out[k] = hold_value
            else:
                out[k] = v
        if "load_mw" not in out:
            out["load_mw"] = out.get("hr_bpm", 0.0)
        if "ts_utc" not in out and "timestamp" in raw_packet:
            out["ts_utc"] = str(raw_packet["timestamp"])
        out["is_critical"] = bool(raw_packet.get("is_critical", False))
        out["telemetry_missing_count"] = int(len(missing_fields))
        out["telemetry_missing_fields"] = ",".join(missing_fields)
        out["telemetry_used_hold"] = bool(held_fields)
        out["telemetry_held_fields"] = ",".join(held_fields)
        return out

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        """Compute reliability from healthcare-native telemetry signals."""
        w_t, flags = assess_domain_reliability(
            domain_id=self.domain_id,
            state=state,
            history=history,
            feature_sources={
                "heart_rate_bpm": "hr_bpm",
                "spo2_pct": "spo2_pct",
                "respiratory_rate_bpm": "respiratory_rate",
            },
            expected_cadence_s=self._expected_cadence_s,
            reliability_cfg=self._cfg.get("reliability", {}),
            ftit_cfg=self._cfg.get("ftit", {}),
            runtime_surface=self._runtime_surface,
            closure_tier=self._closure_tier,
        )
        source_reliability = float(max(0.05, min(1.0, _f(state.get("reliability"), 1.0))))
        missing_count = int(_f(state.get("telemetry_missing_count"), 0.0))
        missing_penalty = 0.05 if missing_count > 0 else 1.0
        w_t = float(max(0.05, min(1.0, w_t * source_reliability * missing_penalty)))
        flags["dataset_reliability"] = source_reliability
        flags["telemetry_missing_count"] = missing_count
        flags["telemetry_missing_fields"] = str(state.get("telemetry_missing_fields", ""))
        flags["telemetry_used_hold"] = bool(state.get("telemetry_used_hold", False))
        flags["w_t"] = w_t
        return float(w_t), flags

    def build_uncertainty_set(
        self,
        state: Mapping[str, Any],
        reliability_w: float,
        quantile: float,
        *,
        cfg: Mapping[str, Any],
        drift_flag: bool | None = None,
        prev_meta: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Conformal box over (hr_bpm, spo2_pct, respiratory_rate) inflated by w_t."""
        hr = _f(state.get("hr_bpm"), 70.0)
        spo2 = _f(state.get("spo2_pct"), 97.0)
        forecast_spo2 = _f(state.get("forecast_spo2_pct"), spo2)
        rr = _f(state.get("respiratory_rate"), 14.0)
        w = max(0.05, float(reliability_w))
        margin = max(0.5, float(quantile) / (w + 1e-9))
        margin = min(margin, 5.0)
        validity_status = _validity_status(reliability_w=w, drift_flag=bool(drift_flag))
        telemetry_missing_count = int(_f(state.get("telemetry_missing_count"), 0.0))
        telemetry_used_hold = bool(state.get("telemetry_used_hold", False))
        if telemetry_missing_count > 0:
            validity_status = "invalid"
        uncertainty = {
            "hr_lower_bpm": max(0.0, hr - margin),
            "hr_upper_bpm": hr + margin,
            "spo2_lower_pct": max(0.0, min(spo2, forecast_spo2) - margin),
            "spo2_upper_pct": min(100.0, max(spo2, forecast_spo2) + margin),
            "forecast_spo2_lower_pct": max(0.0, forecast_spo2 - margin),
            "forecast_spo2_upper_pct": min(100.0, forecast_spo2 + margin),
            "rr_lower": max(0.0, rr - margin),
            "rr_upper": rr + margin,
            "hr_bpm": hr,
            "spo2_pct": spo2,
            "forecast_spo2_pct": forecast_spo2,
            "respiratory_rate": rr,
            "meta": {
                "inflation": margin,
                "w_t": w,
                "validity_status": validity_status,
                "telemetry_missing_count": telemetry_missing_count,
                "telemetry_used_hold": telemetry_used_hold,
                "telemetry_missing_fields": str(state.get("telemetry_missing_fields", "")),
            },
        }
        meta = {
            "inflation": margin,
            "w_t": w,
            "validity_status": validity_status,
            "telemetry_missing_count": telemetry_missing_count,
            "telemetry_used_hold": telemetry_used_hold,
            "telemetry_missing_fields": str(state.get("telemetry_missing_fields", "")),
        }
        return uncertainty, meta

    def tighten_action_set(
        self,
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        *,
        cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Piecewise hold-or-release contract for bounded monitoring."""
        unc = dict(uncertainty)
        cstr = dict(constraints)
        spo2_lo = _f(unc.get("spo2_lower_pct"), self._spo2_min)
        spo2_hi = _f(unc.get("spo2_upper_pct"), 100.0)
        forecast_spo2_lo = _f(unc.get("forecast_spo2_lower_pct"), spo2_lo)
        hr_lo = _f(unc.get("hr_lower_bpm"), self._hr_min)
        hr_hi = _f(unc.get("hr_upper_bpm"), self._hr_max)
        rr_lo = _f(unc.get("rr_lower"), self._rr_min)
        rr_hi = _f(unc.get("rr_upper"), self._rr_max)
        spo2_min = _f(cstr.get("spo2_min_pct"), self._spo2_min)
        hr_min = _f(cstr.get("hr_min_bpm"), self._hr_min)
        hr_max = _f(cstr.get("hr_max_bpm"), self._hr_max)
        rr_min = _f(cstr.get("rr_min"), self._rr_min)
        rr_max = _f(cstr.get("rr_max"), self._rr_max)
        meta = dict(unc.get("meta", {}))
        validity_status = str(meta.get("validity_status", "nominal"))
        reliability_w = _f(meta.get("w_t"), 1.0)
        telemetry_missing_count = int(_f(meta.get("telemetry_missing_count"), 0.0))
        telemetry_used_hold = bool(meta.get("telemetry_used_hold", False))
        unsafe_interval = (
            spo2_lo < spo2_min
            or forecast_spo2_lo < spo2_min
            or spo2_hi > 100.0
            or hr_lo < hr_min
            or hr_hi > hr_max
            or rr_lo < rr_min
            or rr_hi > rr_max
        )
        validity_margin = min(
            spo2_lo - spo2_min,
            forecast_spo2_lo - spo2_min,
            hr_lo - hr_min,
            hr_max - hr_hi,
            rr_lo - rr_min,
            rr_max - rr_hi,
        )
        current_spo2 = _f(unc.get("spo2_pct"), (spo2_lo + spo2_hi) / 2.0)
        current_forecast_spo2 = _f(unc.get("forecast_spo2_pct"), forecast_spo2_lo)
        current_hr = _f(unc.get("hr_bpm"), (hr_lo + hr_hi) / 2.0)
        current_rr = _f(unc.get("respiratory_rate"), (rr_lo + rr_hi) / 2.0)
        current_margin = min(
            current_spo2 - spo2_min,
            current_forecast_spo2 - spo2_min,
            current_hr - hr_min,
            hr_max - current_hr,
            current_rr - rr_min,
            rr_max - current_rr,
        )
        current_vitals_unsafe = (
            current_spo2 < spo2_min
            or current_forecast_spo2 < spo2_min
            or current_spo2 > 100.0
            or current_hr < hr_min
            or current_hr > hr_max
            or current_rr < rr_min
            or current_rr > rr_max
        )
        degraded_validity = (
            validity_status in {"invalid", "degraded"}
            or reliability_w < 0.20
            or telemetry_missing_count > 0
            or telemetry_used_hold
            or validity_margin <= 0.0
        )
        projection_margin_floor = _f(self._cfg.get("graded_alert_projection_margin"), 1.5)
        projected_release = bool(
            (unsafe_interval or degraded_validity)
            and not telemetry_used_hold
            and telemetry_missing_count == 0
            and not current_vitals_unsafe
            and current_margin >= projection_margin_floor
        )
        fallback_required = bool((unsafe_interval or degraded_validity) and not projected_release)
        if telemetry_missing_count > 0 or telemetry_used_hold:
            fallback_reason = "telemetry_missing_or_held"
        elif current_vitals_unsafe:
            fallback_reason = "unsafe_current_vitals"
        elif projected_release:
            fallback_reason = "graded_alert_projection"
        elif unsafe_interval:
            fallback_reason = "unsafe_vital_interval"
        elif validity_margin <= 0.0:
            fallback_reason = "nonpositive_certificate_margin"
        else:
            fallback_reason = "certificate_validity_degraded"
        fallback_region = "max_alert_release" if fallback_required else ("graded_alert_release" if projected_release else "hold_region")
        if fallback_required:
            alert_lower = 1.0
        elif projected_release:
            reliability_penalty = max(0.0, 0.65 - reliability_w)
            margin_credit = min(current_margin, 20.0) / 20.0
            alert_lower = max(0.20, min(0.85, 0.35 + reliability_penalty + 0.25 * (1.0 - margin_credit)))
        else:
            alert_lower = 0.0
        alert_upper = 1.0
        return {
            "uncertainty": unc,
            "constraints": cstr,
            "cfg": dict(cfg),
            "alert_level_lower": float(alert_lower),
            "alert_level_upper": float(alert_upper),
            "fallback_action": {"alert_level": 1.0},
            "projection_surface": "healthcare_fail_safe_release",
            "viable": not fallback_required,
            "fallback_required": fallback_required,
            "fallback_reason": fallback_reason,
            "fallback_region": fallback_region,
            "theorem_contract": "healthcare_fail_safe_release",
            "forecast_spo2_lower_pct": float(forecast_spo2_lo),
            "validity_margin": float(validity_margin),
            "current_vital_margin": float(current_margin),
            "projected_release": bool(projected_release),
            "projected_release_margin": float(current_margin),
            "telemetry_missing_count": telemetry_missing_count,
        }

    def repair_action(
        self,
        candidate_action: Mapping[str, Any],
        tightened_set: Mapping[str, Any],
        *,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        cfg: Mapping[str, Any],
    ) -> tuple[Mapping[str, float], Mapping[str, Any]]:
        """Piecewise hold-or-release repair for bounded monitoring."""
        if "alert_level_lower" not in tightened_set or "alert_level_upper" not in tightened_set:
            tightened_set = self.tighten_action_set(
                uncertainty=tightened_set.get("uncertainty", uncertainty),
                constraints=tightened_set.get("constraints", constraints),
                cfg=cfg,
            )
        if bool(tightened_set.get("fallback_required", False)) or tightened_set.get("viable") is False:
            fallback_action = dict(tightened_set.get("fallback_action", {"alert_level": 1.0}))
            return {"alert_level": float(_f(fallback_action.get("alert_level"), 1.0))}, {
                "mode": "fallback",
                "repaired": True,
                "original_alert_level": _f(candidate_action.get("alert_level", candidate_action.get("alarm_level", 0.0)), 0.0),
                "intervention_reason": str(tightened_set.get("fallback_reason", "max_alert_release")),
                "fallback_region": str(tightened_set.get("fallback_region", "max_alert_release")),
                "fallback_required": True,
                "theorem_contract": str(tightened_set.get("theorem_contract", "healthcare_fail_safe_release")),
            }
        alert = _f(candidate_action.get("alert_level", candidate_action.get("alarm_level", 0.0)), 0.0)
        alert_lo = _f(tightened_set.get("alert_level_lower"), 0.0)
        alert_hi = _f(tightened_set.get("alert_level_upper"), 1.0)
        alert_safe = max(alert_lo, min(alert_hi, alert))
        repaired = abs(alert_safe - alert) > 1e-9
        meta = {
            "mode": "projection" if bool(tightened_set.get("projected_release", False)) else "hold",
            "repaired": repaired,
            "original_alert_level": alert,
            "repair_surface": str(tightened_set.get("projection_surface", "healthcare_fail_safe_release")),
            "fallback_region": str(tightened_set.get("fallback_region", "hold_region")),
            "fallback_required": False,
            "theorem_contract": str(tightened_set.get("theorem_contract", "healthcare_fail_safe_release")),
            "projected_release": bool(tightened_set.get("projected_release", False)),
            "allow_single_step_projected_validity": bool(tightened_set.get("projected_release", False)),
            "projected_release_margin": float(tightened_set.get("projected_release_margin", 0.0)),
        }
        if repaired:
            meta["intervention_reason"] = (
                "graded_alert_projection"
                if bool(tightened_set.get("projected_release", False))
                else "alert_clamp"
            )
        return {"alert_level": float(alert_safe)}, meta

    def emit_certificate(
        self,
        *,
        command_id: str,
        device_id: str,
        zone_id: str,
        controller: str,
        proposed_action: Mapping[str, Any],
        safe_action: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
        reliability: Mapping[str, Any],
        drift: Mapping[str, Any],
        cfg: Mapping[str, Any],
        prev_hash: str | None = None,
        dispatch_plan: Mapping[str, Any] | None = None,
        repair_meta: Mapping[str, Any] | None = None,
        guarantee_meta: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Emit certificate with healthcare action fields."""
        model_hash = str(cfg.get("model_hash", ""))
        config_hash = str(cfg.get("config_hash", ""))
        intervened = bool(repair_meta.get("repaired")) if isinstance(repair_meta, Mapping) else None
        intervention_reason = repair_meta.get("intervention_reason") if isinstance(repair_meta, Mapping) else None
        reliability_w = float(reliability.get("w_t", reliability.get("w", 1.0)))
        drift_flag = bool(drift.get("drift", False))
        inflation = float(uncertainty.get("meta", {}).get("inflation", 1.0))
        validity_status = str(
            cfg.get(
                "validity_status",
                uncertainty.get("meta", {}).get(
                    "validity_status",
                    _validity_status(reliability_w=reliability_w, drift_flag=drift_flag),
                ),
            )
        )
        step_index = int(cfg.get("step_index", 0) or 0)
        validity = domain_certificate_validity_semantics(
            domain="healthcare",
            safe_action=safe_action,
            uncertainty=uncertainty,
            reliability_w=reliability_w,
            validity_status=validity_status,
            step_index=step_index,
            repair_meta=repair_meta,
            cfg=cfg,
        )
        certificate = make_certificate(
            command_id=command_id,
            device_id=device_id,
            zone_id=zone_id,
            controller=controller,
            proposed_action=dict(proposed_action),
            safe_action=dict(safe_action),
            uncertainty=uncertainty,
            reliability=reliability,
            drift=drift,
            model_hash=model_hash,
            config_hash=config_hash,
            prev_hash=prev_hash,
            dispatch_plan=dispatch_plan,
            intervened=intervened,
            intervention_reason=intervention_reason,
            reliability_w=reliability_w,
            drift_flag=drift_flag,
            inflation=inflation,
            validity_score=reliability_w,
            guarantee_checks_passed=validity.guarantee_checks_passed,
            guarantee_fail_reasons=list(validity.guarantee_fail_reasons),
            assumptions_version="healthcare_monitoring_runtime_v1",
            validity_horizon_H_t=validity.validity_horizon_H_t,
            half_life_steps=validity.half_life_steps,
            expires_at_step=validity.expires_at_step,
            validity_status=validity_status,
            runtime_surface=self._runtime_surface,
            closure_tier=self._closure_tier,
            reliability_feature_basis=reliability.get("reliability_feature_basis"),
        )
        certificate = dict(certificate)
        certificate["fallback_mode"] = (
            "max_alert"
            if isinstance(repair_meta, Mapping) and repair_meta.get("mode") == "fallback"
            else "projected_release"
            if isinstance(repair_meta, Mapping) and repair_meta.get("mode") == "projection"
            else "hold"
        )
        certificate["fallback_region"] = repair_meta.get("fallback_region") if isinstance(repair_meta, Mapping) else None
        certificate["fallback_required"] = (
            bool(repair_meta.get("fallback_required", False)) if isinstance(repair_meta, Mapping) else False
        )
        certificate["theorem_contract"] = (
            repair_meta.get("theorem_contract", "healthcare_fail_safe_release")
            if isinstance(repair_meta, Mapping)
            else "healthcare_fail_safe_release"
        )
        certificate["validity_scope"] = validity.validity_scope
        certificate["validity_theorem_id"] = validity.validity_theorem_id
        certificate["validity_theorem_contract"] = validity.validity_theorem_contract
        certificate["projected_release"] = (
            bool(repair_meta.get("projected_release", False)) if isinstance(repair_meta, Mapping) else False
        )
        certificate["projected_release_margin"] = (
            float(repair_meta.get("projected_release_margin", 0.0)) if isinstance(repair_meta, Mapping) else 0.0
        )
        certificate["certificate_hash"] = recompute_certificate_hash(certificate)
        return certificate
