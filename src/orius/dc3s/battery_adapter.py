"""BatteryDomainAdapter: concrete DomainAdapter for the battery domain.

This wraps the existing DC3S battery logic behind the DomainAdapter
interface so that:
  - behavior remains unchanged for the current battery implementation, and
  - a vehicles (or other CPS) adapter can be added alongside without
    touching the ORIUS core.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .calibration import build_uncertainty_set
from .certificate import make_certificate
from .domain_adapter import DomainAdapter
from .quality import compute_reliability
from orius.domain.battery_adapter import repair_battery_action


class BatteryDomainAdapter(DomainAdapter):
    """DomainAdapter implementation using the existing battery/DC3S logic."""

    def capability_profile(self) -> Mapping[str, Any]:
        return {
            "safety_surface_type": "soc_power_envelope",
            "repair_mode": "one_dim_projection",
            "fallback_mode": "safe_hold",
            "supports_multi_agent_eval": True,
            "supports_certos_eval": True,
        }

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        # For the current battery path, telemetry is already a dict with
        # numeric fields consumed directly by compute_reliability and
        # downstream components, so we pass it through.
        return dict(raw_packet)

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        last_event = history[-1] if history else None
        expected_cadence_s = float(state.get("expected_cadence_s", 3600.0))
        reliability_cfg = state.get("reliability_cfg", {})
        ftit_cfg = state.get("ftit_cfg", {})
        w_t, flags = compute_reliability(
            state,
            last_event,
            expected_cadence_s=expected_cadence_s,
            reliability_cfg=reliability_cfg,
            adaptive_state=state.get("adaptive_state"),
            ftit_cfg=ftit_cfg,
        )
        return float(w_t), {"flags": flags}

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
        yhat = state.get("yhat_load")
        q = quantile
        if yhat is None:
            raise ValueError("BatteryDomainAdapter.build_uncertainty_set requires yhat_load in state.")
        prev_inflation = None
        if prev_meta is not None:
            prev_inflation = prev_meta.get("inflation")
        lower, upper, meta = build_uncertainty_set(
            yhat=yhat,
            q=q,
            w_t=float(reliability_w),
            drift_flag=bool(drift_flag) if drift_flag is not None else False,
            cfg=cfg,
            prev_inflation=prev_inflation,
        )
        uncertainty = {
            "lower": lower,
            "upper": upper,
            "meta": dict(meta),
        }
        return uncertainty, dict(meta)

    def tighten_action_set(
        self,
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        *,
        cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        max_power = float(constraints.get("max_power_mw", 0.0))
        max_charge = float(constraints.get("max_charge_mw", max_power))
        max_discharge = float(constraints.get("max_discharge_mw", max_power))
        return {
            "uncertainty": dict(uncertainty),
            "constraints": dict(constraints),
            "cfg": dict(cfg),
            "charge_mw_lower": 0.0,
            "charge_mw_upper": max(0.0, max_charge),
            "discharge_mw_lower": 0.0,
            "discharge_mw_upper": max(0.0, max_discharge),
            "fallback_action": {"charge_mw": 0.0, "discharge_mw": 0.0},
            "projection_surface": "soc_power_envelope",
            "viable": True,
        }

    def project_to_safe_set(
        self,
        candidate_action: Mapping[str, Any],
        uncertainty_set: Mapping[str, Any],
        state: Any,
    ) -> tuple[Mapping[str, float], Mapping[str, Any]]:
        """Shield-compatible projection: clamp action to SOC-safe region."""
        def _state_value(key: str, default: float) -> float:
            if isinstance(state, Mapping):
                return float(state.get(key, default))
            return float(getattr(state, key, default))

        charge = float(candidate_action.get("charge_mw", 0.0))
        discharge = float(candidate_action.get("discharge_mw", 0.0))

        soc = _state_value("current_soc_mwh", 5000.0)
        cap = _state_value("capacity_mwh", 10000.0)
        soc_min = float(uncertainty_set.get("ftit_soc_min_mwh", _state_value("min_soc_mwh", 0.0)))
        soc_max = float(uncertainty_set.get("ftit_soc_max_mwh", _state_value("max_soc_mwh", cap)))
        max_pw = _state_value("max_power_mw", 200.0)

        room_up = max(0.0, soc_max - soc)
        room_dn = max(0.0, soc - soc_min)

        safe_charge = min(max(charge, 0.0), max_pw, room_up)
        safe_discharge = min(max(discharge, 0.0), max_pw, room_dn)

        # Only one direction active
        if safe_charge > 0 and safe_discharge > 0:
            if charge >= discharge:
                safe_discharge = 0.0
            else:
                safe_charge = 0.0

        repaired = (safe_charge != charge) or (safe_discharge != discharge)
        safe = {"charge_mw": safe_charge, "discharge_mw": safe_discharge}
        meta = {
            "mode": "projection",
            "repaired": repaired,
            "original_charge_mw": charge,
            "original_discharge_mw": discharge,
        }
        if repaired:
            meta["intervention_reason"] = "soc_bound_clamp"
        return safe, meta

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
        # Delegate to battery repair logic (legacy signature).
        safe, meta = repair_battery_action(
            a_star=candidate_action,
            state=state,
            uncertainty_set=uncertainty,
            constraints=constraints,
            cfg=cfg,
        )
        return safe, meta

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
        # The existing make_certificate helper already defines the
        # battery certificate payload; we call it here to keep the
        # on-disk format and chain semantics identical.
        model_hash = str(cfg.get("model_hash", ""))
        config_hash = str(cfg.get("config_hash", ""))
        intervened = bool(repair_meta.get("repaired")) if isinstance(repair_meta, Mapping) else None
        intervention_reason = None
        if isinstance(repair_meta, Mapping):
            reason = repair_meta.get("intervention_reason")
            if isinstance(reason, str) and reason:
                intervention_reason = reason
        reliability_w = float(reliability.get("w_t", reliability.get("w", 1.0)))
        drift_flag = bool(drift.get("drift", False))
        inflation = float(uncertainty.get("meta", {}).get("inflation", 1.0))
        guarantee_checks_passed = None
        guarantee_fail_reasons = None
        true_soc_violation_after_apply = None
        if isinstance(guarantee_meta, Mapping):
            guarantee_checks_passed = guarantee_meta.get("guarantee_checks_passed")
            guarantee_fail_reasons = guarantee_meta.get("guarantee_fail_reasons")
            true_soc_violation_after_apply = guarantee_meta.get("true_soc_violation_after_apply")

        certificate = make_certificate(
            command_id=command_id,
            device_id=device_id,
            zone_id=zone_id,
            controller=controller,
            proposed_action=proposed_action,
            safe_action=safe_action,
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
            guarantee_checks_passed=guarantee_checks_passed,
            guarantee_fail_reasons=list(guarantee_fail_reasons or []),
            true_soc_violation_after_apply=true_soc_violation_after_apply,
            assumptions_version=str(cfg.get("assumptions_version")) if cfg.get("assumptions_version") is not None else None,
        )
        return certificate
