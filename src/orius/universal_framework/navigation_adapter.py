"""Navigation domain adapter for the ORIUS universal runtime.

This adapter wraps the toy 2D navigation benchmark in the same five-stage
DC3S pipeline used by the battery and vehicle proof paths. Navigation remains
portability-only evidence, but it is now a first-class runtime domain exposed
through the universal registry and API surface.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from orius.dc3s.certificate import make_certificate
from orius.dc3s.domain_adapter import DomainAdapter
from orius.dc3s.quality import compute_reliability


def _f(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class NavigationDomainAdapter(DomainAdapter):
    """DomainAdapter for bounded 2D robot navigation."""

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        self._cfg = dict(cfg or {})
        nav_cfg = self._cfg.get("navigation", {})
        self._arena_size = _f(nav_cfg.get("arena_size"), 10.0)
        self._speed_limit = _f(nav_cfg.get("speed_limit"), 1.0)
        self._obstacle_radius = _f(nav_cfg.get("obstacle_radius"), 1.0)
        centres = nav_cfg.get("obstacle_centres", [(5.0, 5.0)])
        self._obstacle_centres = [
            (float(pair[0]), float(pair[1]))
            for pair in centres
            if isinstance(pair, Sequence) and len(pair) == 2
        ]
        self._dt_s = _f(nav_cfg.get("dt_s"), 0.25)
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 1.0)

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "x": raw_packet.get("x", raw_packet.get("_hold_x", 0.0)),
            "y": raw_packet.get("y", raw_packet.get("_hold_y", 0.0)),
            "vx": raw_packet.get("vx", raw_packet.get("_hold_vx", 0.0)),
            "vy": raw_packet.get("vy", raw_packet.get("_hold_vy", 0.0)),
            "ts_utc": raw_packet.get("ts_utc", raw_packet.get("timestamp", "")),
        }

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        event = {
            "ts_utc": state.get("ts_utc", ""),
            "load_mw": _f(state.get("x"), 0.0),
            "renewables_mw": _f(state.get("y"), 0.0),
        }
        last_event = None
        if history:
            prev = history[-1]
            last_event = {
                "ts_utc": prev.get("ts_utc", ""),
                "load_mw": _f(prev.get("x"), 0.0),
                "renewables_mw": _f(prev.get("y"), 0.0),
            }
        w_t, flags = compute_reliability(
            event,
            last_event,
            expected_cadence_s=self._expected_cadence_s,
            reliability_cfg=self._cfg.get("reliability", {}),
            ftit_cfg=self._cfg.get("ftit", {}),
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
        x = _f(state.get("x"), 0.0)
        y = _f(state.get("y"), 0.0)
        vx = _f(state.get("vx"), 0.0)
        vy = _f(state.get("vy"), 0.0)
        w = max(0.05, float(reliability_w))
        margin = min(max(0.05, float(quantile) / (100.0 * w)), self._arena_size / 2.0)
        uncertainty = {
            "x_lower": x - margin,
            "x_upper": x + margin,
            "y_lower": y - margin,
            "y_upper": y + margin,
            "vx_lower": vx - margin,
            "vx_upper": vx + margin,
            "vy_lower": vy - margin,
            "vy_upper": vy + margin,
            "meta": {
                "inflation": margin,
                "w_t": w,
                "drift_flag": bool(drift_flag),
            },
        }
        return uncertainty, dict(uncertainty["meta"])

    def tighten_action_set(
        self,
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        *,
        cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
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
        unc = tightened_set.get("uncertainty", uncertainty)
        cstr = tightened_set.get("constraints", constraints)

        orig_ax = _f(candidate_action.get("ax"), 0.0)
        orig_ay = _f(candidate_action.get("ay"), 0.0)
        ax = orig_ax
        ay = orig_ay
        arena_min = _f(cstr.get("arena_min"), 0.0)
        arena_max = _f(cstr.get("arena_max"), self._arena_size)
        speed_limit = _f(cstr.get("max_speed"), self._speed_limit)
        obstacle_radius = _f(cstr.get("obstacle_radius"), self._obstacle_radius)
        dt = _f(cstr.get("dt_s"), self._dt_s)

        mag = math.hypot(ax, ay)
        reason = None
        if mag > speed_limit > 0:
            scale = speed_limit / mag
            ax *= scale
            ay *= scale
            reason = "speed_limit_clamp"

        x = _f(state.get("x"), 0.0)
        y = _f(state.get("y"), 0.0)
        x_hi = _f(unc.get("x_upper"), x)
        x_lo = _f(unc.get("x_lower"), x)
        y_hi = _f(unc.get("y_upper"), y)
        y_lo = _f(unc.get("y_lower"), y)

        next_x_hi = x_hi + ax * dt
        next_x_lo = x_lo + ax * dt
        next_y_hi = y_hi + ay * dt
        next_y_lo = y_lo + ay * dt

        if next_x_hi > arena_max:
            ax = min(ax, (arena_max - x_hi) / max(dt, 1e-9))
            reason = reason or "arena_bound_clamp"
        if next_x_lo < arena_min:
            ax = max(ax, (arena_min - x_lo) / max(dt, 1e-9))
            reason = reason or "arena_bound_clamp"
        if next_y_hi > arena_max:
            ay = min(ay, (arena_max - y_hi) / max(dt, 1e-9))
            reason = reason or "arena_bound_clamp"
        if next_y_lo < arena_min:
            ay = max(ay, (arena_min - y_lo) / max(dt, 1e-9))
            reason = reason or "arena_bound_clamp"

        next_x = x + ax * dt
        next_y = y + ay * dt
        obstacle_hit = False
        for cx, cy in self._obstacle_centres:
            dist = math.hypot(next_x - cx, next_y - cy)
            if dist < obstacle_radius:
                obstacle_hit = True
                away_x = next_x - cx
                away_y = next_y - cy
                away_mag = math.hypot(away_x, away_y)
                if away_mag > 1e-9:
                    ax = (away_x / away_mag) * min(speed_limit, obstacle_radius / max(dt, 1e-9))
                    ay = (away_y / away_mag) * min(speed_limit, obstacle_radius / max(dt, 1e-9))
                else:
                    ax = 0.0
                    ay = 0.0
                reason = "obstacle_avoidance_clamp"
                break

        repaired = abs(ax - orig_ax) > 1e-9 or abs(ay - orig_ay) > 1e-9
        meta = {
            "mode": "projection",
            "repaired": repaired,
            "original_ax": orig_ax,
            "original_ay": orig_ay,
            "obstacle_hit_predicted": obstacle_hit,
        }
        if repaired:
            meta["intervention_reason"] = reason or "navigation_clamp"
        return {"ax": float(ax), "ay": float(ay)}, meta

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
            model_hash=str(cfg.get("model_hash", "")),
            config_hash=str(cfg.get("config_hash", "")),
            prev_hash=prev_hash,
            dispatch_plan=dispatch_plan,
            intervened=intervened,
            intervention_reason=intervention_reason,
            reliability_w=reliability_w,
            drift_flag=drift_flag,
            inflation=inflation,
        )
