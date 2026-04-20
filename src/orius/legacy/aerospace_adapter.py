"""Aerospace domain adapter for ORIUS Universal Framework.

Placeholder for thesis domain 4 (Aerospace). Safety predicates:
- altitude_m above ground
- airspeed_kt in [v_min, v_max]
- bank_angle_deg in [-max_bank, max_bank]
- fuel_remaining_pct >= fuel_min
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


class AerospaceDomainAdapter(DomainAdapter):
    """DomainAdapter for aerospace flight envelope monitoring (placeholder)."""

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        self._cfg = dict(cfg or {})
        self.domain_id = "aerospace"
        ac = self._cfg.get("aerospace", {})
        self._v_min_kt = _f(ac.get("v_min_kt"), 60.0)
        self._v_max_kt = _f(ac.get("v_max_kt"), 350.0)
        self._max_bank_deg = _f(ac.get("max_bank_deg"), 30.0)
        self._fuel_min_pct = _f(ac.get("fuel_min_pct"), 10.0)
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 1.0)
        evidence = resolve_runtime_evidence(self.domain_id, self._cfg)
        self._runtime_surface = evidence.runtime_surface
        self._closure_tier = evidence.closure_tier
        self._maturity_tier = evidence.maturity_tier
        self._fallback_policy = evidence.fallback_policy
        self._exact_blocker = evidence.exact_blocker

    def capability_profile(self) -> Mapping[str, Any]:
        return {
            "safety_surface_type": "approach_energy_envelope",
            "repair_mode": "bounded_projection",
            "fallback_mode": "envelope_hold",
            "supports_multi_agent_eval": False,
            "supports_certos_eval": True,
            "runtime_surface": self._runtime_surface,
            "closure_tier": self._closure_tier,
            "maturity_tier": self._maturity_tier,
            "fallback_policy": self._fallback_policy,
            "exact_blocker": self._exact_blocker,
        }

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        """Parse raw flight telemetry into state vector z_t."""
        out = {}
        for k in ("altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct", "ts_utc"):
            v = raw_packet.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[k] = raw_packet.get(f"_hold_{k}", 0.0 if k != "ts_utc" else "")
            else:
                out[k] = v
        if "load_mw" not in out:
            out["load_mw"] = out.get("airspeed_kt", 0.0)
        if "ts_utc" not in out and "timestamp" in raw_packet:
            out["ts_utc"] = str(raw_packet["timestamp"])
        return out

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        """Compute reliability from aerospace-native telemetry signals."""
        w_t, flags = assess_domain_reliability(
            domain_id=self.domain_id,
            state=state,
            history=history,
            feature_sources={
                "altitude_m": "altitude_m",
                "airspeed_kt": "airspeed_kt",
                "bank_angle_deg": "bank_angle_deg",
                "fuel_remaining_pct": "fuel_remaining_pct",
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
        """Conformal box over (altitude, airspeed, bank, fuel) inflated by w_t."""
        alt = _f(state.get("altitude_m"), 1000.0)
        v = _f(state.get("airspeed_kt"), 150.0)
        bank = _f(state.get("bank_angle_deg"), 0.0)
        fuel = _f(state.get("fuel_remaining_pct"), 50.0)
        w = max(0.05, float(reliability_w))
        margin = max(0.5, float(quantile) / (w + 1e-9))
        margin = min(margin, 10.0)
        uncertainty = {
            "alt_lower_m": max(0.0, alt - margin * 100),
            "alt_upper_m": alt + margin * 100,
            "v_lower_kt": max(0.0, v - margin),
            "v_upper_kt": v + margin,
            "bank_lower_deg": max(-90.0, bank - margin),
            "bank_upper_deg": min(90.0, bank + margin),
            "fuel_lower_pct": max(0.0, fuel - margin),
            "fuel_upper_pct": min(100.0, fuel + margin),
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
        """Feasible command set from flight envelope limits."""
        unc = dict(uncertainty)
        cstr = dict(constraints)
        v_lo = _f(unc.get("v_lower_kt"), self._v_min_kt)
        fuel_lo = _f(unc.get("fuel_lower_pct"), self._fuel_min_pct)
        max_bank = _f(cstr.get("max_bank_deg"), self._max_bank_deg)
        throttle_lower = 0.0
        throttle_upper = 1.0
        active_constraints: list[str] = []
        if v_lo < self._v_min_kt:
            throttle_upper = min(throttle_upper, 0.8)
            active_constraints.append("low_speed_uncertainty")
        if fuel_lo < self._fuel_min_pct:
            throttle_upper = min(throttle_upper, 0.5)
            active_constraints.append("low_fuel_uncertainty")
        fallback_throttle = 0.8 if v_lo < self._v_min_kt else 0.5 if fuel_lo < self._fuel_min_pct else 0.6
        if throttle_lower > throttle_upper:
            throttle_lower = throttle_upper = fallback_throttle
        return {
            "uncertainty": unc,
            "constraints": cstr,
            "cfg": dict(cfg),
            "throttle_lower": float(throttle_lower),
            "throttle_upper": float(throttle_upper),
            "bank_deg_lower": float(-max_bank),
            "bank_deg_upper": float(max_bank),
            "fallback_action": {"throttle": float(fallback_throttle), "bank_deg": 0.0},
            "projection_surface": "approach_energy_envelope",
            "active_constraints": active_constraints,
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
        """Clip command to safe flight envelope. Aerospace: throttle, bank, pitch."""
        if "throttle_lower" not in tightened_set or "throttle_upper" not in tightened_set:
            tightened_set = self.tighten_action_set(
                uncertainty=tightened_set.get("uncertainty", uncertainty),
                constraints=tightened_set.get("constraints", constraints),
                cfg=cfg,
            )
        throttle = _f(candidate_action.get("throttle", candidate_action.get("throttle_pct", 0.0)), 0.0)
        bank_cmd = _f(candidate_action.get("bank_deg", 0.0), 0.0)
        unc = tightened_set.get("uncertainty", uncertainty)
        v_lo = _f(unc.get("v_lower_kt"), self._v_min_kt)
        fuel_lo = _f(unc.get("fuel_lower_pct"), self._fuel_min_pct)
        throttle_safe = max(
            _f(tightened_set.get("throttle_lower"), 0.0),
            min(_f(tightened_set.get("throttle_upper"), 1.0), throttle),
        )
        bank_safe = max(
            _f(tightened_set.get("bank_deg_lower"), -self._max_bank_deg),
            min(_f(tightened_set.get("bank_deg_upper"), self._max_bank_deg), bank_cmd),
        )
        active_constraints = [str(item) for item in tightened_set.get("active_constraints", ())]
        v_actual = _f(state.get("airspeed_kt", self._v_min_kt), self._v_min_kt)
        reason = None
        if v_actual < self._v_min_kt:
            # Recovery mode: actual airspeed is below stall — enforce minimum throttle
            throttle_safe = max(throttle_safe, 0.8)
            reason = "stall_recovery"
        elif v_lo < self._v_min_kt:
            # Uncertainty suggests possible low-speed condition — cap throttle conservatively
            throttle_safe = min(throttle_safe, 0.8)
            reason = reason or "low_speed_uncertainty"
        if fuel_lo < self._fuel_min_pct:
            throttle_safe = min(throttle_safe, 0.5)
            reason = reason or "low_fuel_uncertainty"
        repaired = (
            abs(throttle_safe - throttle) > 1e-9 or abs(bank_safe - bank_cmd) > 1e-9
        )
        meta = {
            "mode": "projection",
            "repaired": repaired,
            "original_throttle": throttle,
            "original_bank": bank_cmd,
            "repair_surface": str(tightened_set.get("projection_surface", "approach_energy_envelope")),
        }
        if repaired:
            meta["intervention_reason"] = reason or (
                "low_speed_uncertainty"
                if "low_speed_uncertainty" in active_constraints
                else "low_fuel_uncertainty"
                if "low_fuel_uncertainty" in active_constraints
                else "envelope_clamp"
            )
        return {
            "throttle": float(throttle_safe),
            "bank_deg": float(bank_safe),
        }, meta

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
        """Emit certificate with aerospace action fields."""
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
