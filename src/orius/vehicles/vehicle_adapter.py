"""VehicleDomainAdapter: DomainAdapter for 1D longitudinal vehicle control.

Prototype extension. Not part of locked battery thesis claims.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from orius.dc3s.domain_adapter import DomainAdapter
from orius.dc3s.quality import compute_reliability
from orius.dc3s.certificate import make_certificate


def _f(x: Any, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


class VehicleDomainAdapter(DomainAdapter):
    """DomainAdapter for 1D longitudinal vehicle (speed control along a lane)."""

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        self._cfg = dict(cfg or {})
        self._default_dt_s = _f(self._cfg.get("vehicles", {}).get("dt_s"), 0.25)
        self._accel_min = _f(self._cfg.get("vehicles", {}).get("accel_min_mps2"), -5.0)
        self._accel_max = _f(self._cfg.get("vehicles", {}).get("accel_max_mps2"), 3.0)
        self._min_headway_m = _f(self._cfg.get("vehicles", {}).get("min_headway_m"), 5.0)
        self._ttc_min_s = _f(
            self._cfg.get("vehicles", {}).get("ttc_min_s"),
            self._cfg.get("vehicles", {}).get("headway_time_s", 2.0),
        )
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 1.0)

    def capability_profile(self) -> Mapping[str, Any]:
        return {
            "safety_surface_type": "ttc_entry_barrier",
            "repair_mode": "one_dim_projection",
            "fallback_mode": "full_brake",
            "supports_multi_agent_eval": False,
            "supports_certos_eval": True,
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
        return out

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        """Compute reliability w_t. Reuse battery OQE with vehicle event shape."""
        event = {
            "ts_utc": state.get("ts_utc", ""),
            "load_mw": state.get("speed_mps", state.get("load_mw", 0.0)),
            "renewables_mw": state.get("position_m", 0.0),
        }
        last_event = None
        if history:
            prev = history[-1]
            last_event = {
                "ts_utc": prev.get("ts_utc", ""),
                "load_mw": prev.get("speed_mps", prev.get("load_mw", 0.0)),
                "renewables_mw": prev.get("position_m", 0.0),
            }
        reliability_cfg = self._cfg.get("reliability", {})
        ftit_cfg = self._cfg.get("ftit", {})
        w_t, flags = compute_reliability(
            event,
            last_event,
            expected_cadence_s=self._expected_cadence_s,
            reliability_cfg=reliability_cfg,
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
        return {
            "uncertainty": dict(uncertainty),
            "constraints": dict(constraints),
            "cfg": dict(cfg),
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
        a = _f(candidate_action.get("acceleration_mps2"), 0.0)
        unc = tightened_set.get("uncertainty", uncertainty)
        cstr = tightened_set.get("constraints", constraints)
        spd_lo = _f(unc.get("speed_lower_mps"), 0.0)
        spd_hi = _f(unc.get("speed_upper_mps"), 50.0)
        pos_hi = _f(unc.get("position_upper_m"), state.get("position_m", 0.0))
        lead_lo = unc.get("lead_position_lower_m", state.get("lead_position_m"))
        lead_speed = _f(state.get("lead_speed_mps"), cstr.get("lead_speed_mps", 0.0))
        v_limit = _f(state.get("speed_limit_mps"), cstr.get("speed_limit_mps", 30.0))
        a_min = _f(cstr.get("accel_min_mps2"), self._accel_min)
        a_max = _f(cstr.get("accel_max_mps2"), self._accel_max)
        dt = _f(cstr.get("dt_s"), self._default_dt_s)
        min_headway_m = _f(cstr.get("min_headway_m"), self._min_headway_m)
        ttc_min_s = _f(cstr.get("ttc_min_s"), cstr.get("headway_time_s", self._ttc_min_s))

        a_safe = max(a_min, min(a_max, a))
        reason = None
        ttc_current = None
        ttc_next = None
        entry_barrier_triggered = False

        telemetry_vals = (spd_lo, spd_hi, pos_hi, v_limit)
        if any(isinstance(v, float) and math.isnan(v) for v in telemetry_vals):
            a_safe = a_min
            reason = "telemetry_blackout_brake"

        v_next = spd_hi + a_safe * dt
        if v_next > v_limit + 1e-9:
            a_cap = (v_limit - spd_hi) / max(dt, 1e-9)
            a_safe = min(a_safe, a_cap)
            reason = reason or "speed_limit_clamp"
        v_next_lo = spd_lo + a_safe * dt
        if v_next_lo < 0:
            a_floor = -spd_lo / max(dt, 1e-9)
            a_safe = max(a_safe, a_floor)
            reason = reason or "nonnegative_speed_clamp"

        if lead_lo is not None and not (isinstance(lead_lo, float) and math.isnan(lead_lo)):
            lead_lo_f = _f(lead_lo, pos_hi)
            gap_budget = lead_lo_f - pos_hi - min_headway_m
            if gap_budget <= 0:
                a_safe = a_min
                reason = "predictive_entry_barrier"
                entry_barrier_triggered = True
            else:
                closing_speed = max(spd_hi - lead_speed, 1e-9)
                ttc_current = gap_budget / closing_speed

                max_next_speed = (gap_budget / max(dt + ttc_min_s, 1e-9)) + lead_speed
                a_ttc = (max_next_speed - spd_hi) / max(dt, 1e-9)
                if a_ttc < a_safe - 1e-9:
                    a_safe = a_ttc
                    reason = reason or "ttc_clamp"

                full_brake_speed = max(lead_speed, spd_hi + a_min * dt)
                unavoidable_gap_budget = lead_lo_f - (pos_hi + full_brake_speed * dt) - min_headway_m
                if unavoidable_gap_budget <= 0.0:
                    a_safe = a_min
                    reason = "predictive_entry_barrier"
                    entry_barrier_triggered = True

                next_speed = max(lead_speed, spd_hi + a_safe * dt)
                next_gap_budget = lead_lo_f - (pos_hi + next_speed * dt) - min_headway_m
                if next_gap_budget > 0.0:
                    ttc_next = next_gap_budget / max(next_speed - lead_speed, 1e-9)
                else:
                    ttc_next = 0.0

        a_safe = max(a_min, min(a_max, a_safe))

        repaired = abs(a_safe - a) > 1e-9
        meta = {
            "mode": "projection",
            "repaired": repaired,
            "original_acceleration_mps2": a,
            "repair_surface": "ttc_predictive_barrier",
        }
        if repaired:
            meta["intervention_reason"] = reason or "acceleration_clamp"
        if lead_lo is not None and not (isinstance(lead_lo, float) and math.isnan(lead_lo)):
            meta["worst_case_gap_budget_m"] = float(_f(lead_lo, pos_hi) - pos_hi - min_headway_m)
            meta["ttc_min_s"] = float(ttc_min_s)
            meta["ttc_seconds_current"] = float(ttc_current) if ttc_current is not None else None
            meta["ttc_seconds_next"] = float(ttc_next) if ttc_next is not None else None
            meta["entry_barrier_triggered"] = bool(entry_barrier_triggered)
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
        )
