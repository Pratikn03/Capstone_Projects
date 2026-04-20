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
from orius.dc3s.certificate import make_certificate
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
            "repair_mode": "one_dim_projection",
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
        for k in ("hr_bpm", "spo2_pct", "respiratory_rate", "ts_utc"):
            v = raw_packet.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[k] = raw_packet.get(f"_hold_{k}", 0.0 if k != "ts_utc" else "")
            else:
                out[k] = v
        if "load_mw" not in out:
            out["load_mw"] = out.get("hr_bpm", 0.0)
        if "ts_utc" not in out and "timestamp" in raw_packet:
            out["ts_utc"] = str(raw_packet["timestamp"])
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
        rr = _f(state.get("respiratory_rate"), 14.0)
        w = max(0.05, float(reliability_w))
        margin = max(0.5, float(quantile) / (w + 1e-9))
        margin = min(margin, 5.0)
        uncertainty = {
            "hr_lower_bpm": max(0.0, hr - margin),
            "hr_upper_bpm": hr + margin,
            "spo2_lower_pct": max(0.0, spo2 - margin),
            "spo2_upper_pct": min(100.0, spo2 + margin),
            "rr_lower": max(0.0, rr - margin),
            "rr_upper": rr + margin,
            "meta": {"inflation": margin, "w_t": w},
        }
        meta = {"inflation": margin, "w_t": w}
        return uncertainty, meta

    def tighten_action_set(
        self,
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        *,
        cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Feasible alarm/alert set from vital sign limits."""
        unc = dict(uncertainty)
        cstr = dict(constraints)
        spo2_lo = _f(unc.get("spo2_lower_pct"), self._spo2_min)
        spo2_min = _f(cstr.get("spo2_min_pct"), self._spo2_min)
        alert_lower = 0.8 if spo2_lo < spo2_min else 0.0
        alert_upper = 1.0
        if alert_lower > alert_upper:
            alert_lower = alert_upper = 1.0
        return {
            "uncertainty": unc,
            "constraints": cstr,
            "cfg": dict(cfg),
            "alert_level_lower": float(alert_lower),
            "alert_level_upper": float(alert_upper),
            "fallback_action": {"alert_level": 1.0},
            "projection_surface": "vital_alert_box",
            "viable": True,
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
        """Clip alarm threshold to safe bounds. Healthcare: alert_level in [0,1]."""
        if "alert_level_lower" not in tightened_set or "alert_level_upper" not in tightened_set:
            tightened_set = self.tighten_action_set(
                uncertainty=tightened_set.get("uncertainty", uncertainty),
                constraints=tightened_set.get("constraints", constraints),
                cfg=cfg,
            )
        alert = _f(candidate_action.get("alert_level", candidate_action.get("alarm_level", 0.0)), 0.0)
        alert_lo = _f(tightened_set.get("alert_level_lower"), 0.0)
        alert_hi = _f(tightened_set.get("alert_level_upper"), 1.0)
        alert_safe = max(alert_lo, min(alert_hi, alert))
        repaired = abs(alert_safe - alert) > 1e-9
        meta = {
            "mode": "projection",
            "repaired": repaired,
            "original_alert_level": alert,
            "repair_surface": str(tightened_set.get("projection_surface", "vital_alert_box")),
        }
        if repaired:
            meta["intervention_reason"] = "alert_clamp"
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
        return make_certificate(
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
            runtime_surface=self._runtime_surface,
            closure_tier=self._closure_tier,
            reliability_feature_basis=reliability.get("reliability_feature_basis"),
        )
