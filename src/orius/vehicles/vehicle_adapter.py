"""VehicleDomainAdapter: DomainAdapter for 1D longitudinal vehicle control.

Prototype extension. Not part of locked battery thesis claims.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from orius.dc3s.certificate import make_certificate
from orius.dc3s.domain_adapter import DomainAdapter
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


class VehicleDomainAdapter(DomainAdapter):
    """DomainAdapter for 1D longitudinal vehicle (speed control along a lane)."""

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        self._cfg = dict(cfg or {})
        self.domain_id = "av"
        self._default_dt_s = _f(self._cfg.get("vehicles", {}).get("dt_s"), 0.25)
        self._accel_min = _f(self._cfg.get("vehicles", {}).get("accel_min_mps2"), -5.0)
        self._accel_max = _f(self._cfg.get("vehicles", {}).get("accel_max_mps2"), 3.0)
        self._min_headway_m = _f(self._cfg.get("vehicles", {}).get("min_headway_m"), 5.0)
        self._ttc_min_s = _f(
            self._cfg.get("vehicles", {}).get("ttc_min_s"),
            self._cfg.get("vehicles", {}).get("headway_time_s", 2.0),
        )
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 1.0)
        evidence = resolve_runtime_evidence(self.domain_id, self._cfg)
        self._runtime_surface = evidence.runtime_surface
        self._closure_tier = evidence.closure_tier
        self._maturity_tier = evidence.maturity_tier
        self._fallback_policy = evidence.fallback_policy
        self._exact_blocker = evidence.exact_blocker

    def capability_profile(self) -> Mapping[str, Any]:
        return {
            "safety_surface_type": "ttc_entry_barrier",
            "repair_mode": "one_dim_projection",
            "fallback_mode": "full_brake",
            "supports_multi_agent_eval": False,
            "supports_certos_eval": True,
            "runtime_surface": self._runtime_surface,
            "closure_tier": self._closure_tier,
            "maturity_tier": self._maturity_tier,
            "fallback_policy": self._fallback_policy,
            "exact_blocker": self._exact_blocker,
        }

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        """Parse raw vehicle packet into state vector z_t. Zero-order hold for NaN."""
        out = {}
        for k in ("position_m", "speed_mps", "speed_limit_mps", "lead_position_m", "ts_utc"):
            v = raw_packet.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[k] = raw_packet.get(f"_hold_{k}", 0.0 if k != "ts_utc" else "")
            else:
                out[k] = v
        if "load_mw" not in out:
            out["load_mw"] = out.get("speed_mps", 0.0)
        if "ts_utc" not in out and "timestamp" in raw_packet:
            out["ts_utc"] = str(raw_packet["timestamp"])
        # Pass through RSS fields when present (Path B)
        for k in ("lead_present", "lead_rel_x_m", "lead_speed_mps", "rss_safe_gap_m", "rss_violation_true"):
            if k in raw_packet:
                out[k] = raw_packet[k]
        return out

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        """Compute reliability from AV-native telemetry signals."""

        def _lead_gap(payload: Mapping[str, Any]) -> float | None:
            lead_position = payload.get("lead_position_m")
            if lead_position is None:
                return None
            return _f(lead_position, 0.0) - _f(payload.get("position_m"), 0.0)

        w_t, flags = assess_domain_reliability(
            domain_id=self.domain_id,
            state=state,
            history=history,
            feature_sources={
                "ego_speed_mps": "speed_mps",
                "lead_gap_m": _lead_gap,
                "speed_limit_mps": "speed_limit_mps",
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
        """Conformal box over (position_m, speed_mps) inflated by w_t."""
        pos = _f(state.get("position_m"), 0.0)
        spd = _f(state.get("speed_mps"), 0.0)
        w = max(0.05, float(reliability_w))
        margin = max(0.1, float(quantile) / (w + 1e-9))
        margin = min(margin, 10.0)
        uncertainty = {
            "position_lower_m": pos - margin,
            "position_upper_m": pos + margin,
            "speed_lower_mps": max(0.0, spd - margin),
            "speed_upper_mps": spd + margin,
            "meta": {"inflation": margin, "w_t": w},
        }
        lead = state.get("lead_position_m")
        if lead is not None and not (isinstance(lead, float) and math.isnan(lead)):
            lead_f = _f(lead, 0.0)
            uncertainty["lead_position_lower_m"] = lead_f - margin
            uncertainty["lead_position_upper_m"] = lead_f + margin
        meta = {"inflation": margin, "w_t": w}
        return uncertainty, meta

    def tighten_action_set(
        self,
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        *,
        cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Feasible acceleration set A_t from speed-limit and TTC barrier."""
        unc = dict(uncertainty)
        cstr = dict(constraints)
        spd_lo = _f(unc.get("speed_lower_mps"), 0.0)
        spd_hi = _f(unc.get("speed_upper_mps"), cstr.get("speed_limit_mps", 30.0))
        pos_hi = _f(unc.get("position_upper_m"), 0.0)
        lead_lo = unc.get("lead_position_lower_m")
        lead_speed = _f(cstr.get("lead_speed_mps"), 0.0)
        v_limit = _f(cstr.get("speed_limit_mps"), 30.0)
        a_lower = _f(cstr.get("accel_min_mps2"), self._accel_min)
        a_upper = _f(cstr.get("accel_max_mps2"), self._accel_max)
        dt = _f(cstr.get("dt_s"), self._default_dt_s)
        min_headway_m = _f(cstr.get("min_headway_m"), self._min_headway_m)
        ttc_min_s = _f(cstr.get("ttc_min_s"), cstr.get("headway_time_s", self._ttc_min_s))
        active_constraints: list[str] = []
        gap_budget = None
        entry_barrier_triggered = False

        if spd_hi + a_upper * dt > v_limit + 1e-9:
            a_upper = min(a_upper, (v_limit - spd_hi) / max(dt, 1e-9))
            active_constraints.append("speed_limit")
        if spd_lo + a_lower * dt < 0.0:
            a_lower = max(a_lower, -spd_lo / max(dt, 1e-9))
            active_constraints.append("nonnegative_speed")

        if lead_lo is not None and not (isinstance(lead_lo, float) and math.isnan(lead_lo)):
            lead_lo_f = _f(lead_lo, pos_hi)
            gap_budget = lead_lo_f - pos_hi - min_headway_m
            if gap_budget <= 0.0:
                a_lower = a_upper = _f(cstr.get("accel_min_mps2"), self._accel_min)
                active_constraints.append("headway_predictive_entry_barrier")
                entry_barrier_triggered = True
            else:
                max_next_speed = (gap_budget / max(dt + ttc_min_s, 1e-9)) + lead_speed
                a_ttc = (max_next_speed - spd_hi) / max(dt, 1e-9)
                if a_ttc < a_upper - 1e-9:
                    a_upper = a_ttc
                    active_constraints.append("ttc_clamp")

                full_brake_speed = max(lead_speed, spd_hi + a_lower * dt)
                unavoidable_gap_budget = lead_lo_f - (pos_hi + full_brake_speed * dt) - min_headway_m
                if unavoidable_gap_budget <= 0.0:
                    a_lower = a_upper = _f(cstr.get("accel_min_mps2"), self._accel_min)
                    active_constraints.append("headway_predictive_entry_barrier")
                    entry_barrier_triggered = True

        fallback_accel = float(_f(cstr.get("accel_min_mps2"), self._accel_min))
        viable = a_lower <= a_upper + 1e-9
        if not viable:
            a_lower = a_upper = fallback_accel
            active_constraints.append("fallback_collapse")
        return {
            "uncertainty": unc,
            "constraints": cstr,
            "cfg": dict(cfg),
            "acceleration_mps2_lower": float(a_lower),
            "acceleration_mps2_upper": float(a_upper),
            "fallback_action": {"acceleration_mps2": fallback_accel},
            "projection_surface": "ttc_predictive_barrier",
            "active_constraints": active_constraints,
            "entry_barrier_triggered": bool(entry_barrier_triggered),
            "worst_case_gap_budget_m": None if gap_budget is None else float(gap_budget),
            "ttc_min_s": float(ttc_min_s),
            "viable": bool(viable),
            "fallback_forced": bool(not viable),
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
        """Clip acceleration to the vehicle-safe set.

        The repair step enforces both speed-limit and TTC predicates under a
        worst-case one-step uncertainty box. A predictive entry barrier forces
        maximum braking when the observed state is already outside the one-step
        controllable set.
        """
        if "acceleration_mps2_lower" not in tightened_set or "acceleration_mps2_upper" not in tightened_set:
            tightened_set = self.tighten_action_set(
                uncertainty=tightened_set.get("uncertainty", uncertainty),
                constraints=tightened_set.get("constraints", constraints),
                cfg=cfg,
            )
        a = _f(candidate_action.get("acceleration_mps2"), 0.0)
        tightened_set.get("uncertainty", uncertainty)
        cstr = tightened_set.get("constraints", constraints)
        a_min = _f(cstr.get("accel_min_mps2"), self._accel_min)
        a_lower = _f(tightened_set.get("acceleration_mps2_lower"), a_min)
        a_upper = _f(
            tightened_set.get("acceleration_mps2_upper"),
            _f(cstr.get("accel_max_mps2"), self._accel_max),
        )
        a_safe = max(a_lower, min(a_upper, a))
        active_constraints = [str(item) for item in tightened_set.get("active_constraints", ())]
        reason = None
        if abs(a_safe - a) > 1e-9:
            # Prefer the governing safety surface over the generic collapse marker
            # so certificates explain why the set collapsed, not just that it did.
            if "headway_predictive_entry_barrier" in active_constraints:
                reason = "headway_predictive_entry_barrier"
            elif "ttc_clamp" in active_constraints:
                reason = "ttc_clamp"
            elif "speed_limit" in active_constraints:
                reason = "speed_limit_clamp"
            elif "nonnegative_speed" in active_constraints:
                reason = "nonnegative_speed_clamp"
            elif tightened_set.get("fallback_forced"):
                reason = "fallback_collapse"
            else:
                reason = "acceleration_clamp"

        repaired = abs(a_safe - a) > 1e-9
        mode = "fallback" if tightened_set.get("fallback_forced") else "projection"
        meta = {
            "mode": mode,
            "repaired": repaired,
            "original_acceleration_mps2": a,
            "repair_surface": str(tightened_set.get("projection_surface", "ttc_predictive_barrier")),
        }
        if repaired:
            meta["intervention_reason"] = reason or "acceleration_clamp"
        if tightened_set.get("worst_case_gap_budget_m") is not None:
            meta["worst_case_gap_budget_m"] = float(tightened_set["worst_case_gap_budget_m"])
            meta["ttc_min_s"] = float(tightened_set.get("ttc_min_s", self._ttc_min_s))
            meta["entry_barrier_triggered"] = bool(tightened_set.get("entry_barrier_triggered", False))
        return {"acceleration_mps2": float(a_safe)}, meta

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
        """Emit certificate with vehicle action fields. Same structure as battery."""
        model_hash = str(cfg.get("model_hash", ""))
        config_hash = str(cfg.get("config_hash", ""))
        intervened = bool(repair_meta.get("repaired")) if isinstance(repair_meta, Mapping) else None
        intervention_reason = (
            repair_meta.get("intervention_reason") if isinstance(repair_meta, Mapping) else None
        )
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
