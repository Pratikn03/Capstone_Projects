"""Industrial domain adapter for ORIUS Universal Framework.

Process control: temperature, pressure, power. Safety predicates:
- temp_c in [temp_min, temp_max]
- pressure_mbar in [p_min, p_max]
- power_mw in [0, power_max]
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
        return float(x)
    except (TypeError, ValueError):
        return float(default)


class IndustrialDomainAdapter(DomainAdapter):
    """DomainAdapter for industrial process control (CCPP-style)."""

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        self._cfg = dict(cfg or {})
        self.domain_id = "industrial"
        ind = self._cfg.get("industrial", {})
        self._temp_min = _f(ind.get("temp_min_c"), 0.0)
        self._temp_max = _f(ind.get("temp_max_c"), 50.0)
        self._pressure_min = _f(ind.get("pressure_min_mbar"), 990.0)
        self._pressure_max = _f(ind.get("pressure_max_mbar"), 1040.0)
        self._power_max = _f(ind.get("power_max_mw"), 500.0)
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 3600.0)
        evidence = resolve_runtime_evidence(self.domain_id, self._cfg)
        self._runtime_surface = evidence.runtime_surface
        self._closure_tier = evidence.closure_tier
        self._maturity_tier = evidence.maturity_tier
        self._fallback_policy = evidence.fallback_policy
        self._exact_blocker = evidence.exact_blocker

    def capability_profile(self) -> Mapping[str, Any]:
        return {
            "safety_surface_type": "power_temperature_envelope",
            "repair_mode": "one_dim_projection",
            "fallback_mode": "power_cap",
            "supports_multi_agent_eval": True,
            "supports_certos_eval": True,
            "runtime_surface": self._runtime_surface,
            "closure_tier": self._closure_tier,
            "maturity_tier": self._maturity_tier,
            "fallback_policy": self._fallback_policy,
            "exact_blocker": self._exact_blocker,
        }

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        """Parse raw industrial packet into state vector z_t."""
        out = {}
        for k in ("temp_c", "vacuum_cmhg", "pressure_mbar", "humidity_pct", "power_mw", "ts_utc"):
            v = raw_packet.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[k] = raw_packet.get(f"_hold_{k}", 0.0 if k != "ts_utc" else "")
            else:
                out[k] = v
        if "load_mw" not in out:
            out["load_mw"] = out.get("power_mw", 0.0)
        if "ts_utc" not in out and "timestamp" in raw_packet:
            out["ts_utc"] = str(raw_packet["timestamp"])
        return out

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        """Compute reliability from industrial-native telemetry signals."""
        w_t, flags = assess_domain_reliability(
            domain_id=self.domain_id,
            state=state,
            history=history,
            feature_sources={
                "power_output_mw": "power_mw",
                "temperature_c": "temp_c",
                "pressure_mbar": "pressure_mbar",
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
        """Conformal box over (temp_c, pressure_mbar, power_mw) inflated by w_t."""
        temp = _f(state.get("temp_c"), 25.0)
        pressure = _f(state.get("pressure_mbar"), 1010.0)
        power = _f(state.get("power_mw"), 450.0)
        w = max(0.05, float(reliability_w))
        margin = max(0.5, float(quantile) / (w + 1e-9))
        margin = min(margin, 10.0)
        uncertainty = {
            "temp_lower_c": temp - margin,
            "temp_upper_c": temp + margin,
            "pressure_lower_mbar": pressure - margin,
            "pressure_upper_mbar": pressure + margin,
            "power_lower_mw": max(0.0, power - margin),
            "power_upper_mw": power + margin,
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
        """Feasible setpoint set from temp/pressure/power limits."""
        unc = dict(uncertainty)
        cstr = dict(constraints)
        power_lo = max(0.0, _f(unc.get("power_lower_mw"), 0.0))
        power_hi = min(
            _f(cstr.get("power_max_mw"), self._power_max),
            _f(unc.get("power_upper_mw"), self._power_max),
        )
        viable = power_lo <= power_hi + 1e-9
        if not viable:
            power_lo = power_hi = max(0.0, min(_f(cstr.get("power_max_mw"), self._power_max), power_lo))
            viable = True
        return {
            "uncertainty": unc,
            "constraints": cstr,
            "cfg": dict(cfg),
            "power_setpoint_lower_mw": float(power_lo),
            "power_setpoint_upper_mw": float(power_hi),
            "fallback_action": {"power_setpoint_mw": float(power_lo)},
            "projection_surface": "power_envelope_box",
            "viable": bool(viable),
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
        """Clip setpoint to safe bounds. Industrial: power_mw in [0, power_max]."""
        if "power_setpoint_lower_mw" not in tightened_set or "power_setpoint_upper_mw" not in tightened_set:
            tightened_set = self.tighten_action_set(
                uncertainty=tightened_set.get("uncertainty", uncertainty),
                constraints=tightened_set.get("constraints", constraints),
                cfg=cfg,
            )
        power = _f(candidate_action.get("power_setpoint_mw", candidate_action.get("power_mw", 0.0)), 0.0)
        power_lo = _f(tightened_set.get("power_setpoint_lower_mw"), 0.0)
        power_hi = _f(tightened_set.get("power_setpoint_upper_mw"), self._power_max)
        power_safe = max(power_lo, min(power_hi, power))
        repaired = abs(power_safe - power) > 1e-9
        meta = {
            "mode": "projection",
            "repaired": repaired,
            "original_power_mw": power,
            "repair_surface": str(tightened_set.get("projection_surface", "power_envelope_box")),
        }
        if repaired:
            meta["intervention_reason"] = "power_clamp"
        return {"power_setpoint_mw": float(power_safe)}, meta

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
        """Emit certificate with industrial action fields."""
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
