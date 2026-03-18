"""Navigation domain adapter for the ORIUS Universal Framework.

2D point-robot navigation in a bounded arena with circular obstacles.
This makes the navigation domain available through the same runtime
adapter interface as the other universal-framework domains.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from orius.dc3s.certificate import make_certificate
from orius.dc3s.domain_adapter import DomainAdapter
from orius.dc3s.quality import compute_reliability


def _f(x: Any, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


class NavigationDomainAdapter(DomainAdapter):
    """DomainAdapter for bounded 2D navigation."""

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        self._cfg = dict(cfg or {})
        nav = self._cfg.get("navigation", {})
        self._arena_size = _f(nav.get("arena_size"), 10.0)
        self._speed_limit = _f(nav.get("speed_limit"), 1.0)
        self._obstacle_radius = _f(nav.get("obstacle_radius"), 1.0)
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 1.0)
        raw_centres = nav.get("obstacle_centres", [(5.0, 5.0)])
        self._obstacle_centres = [
            (_f(cx, 5.0), _f(cy, 5.0))
            for cx, cy in raw_centres
        ]

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        """Parse raw navigation telemetry into a state dict."""
        out = {}
        for key in ("x", "y", "vx", "vy", "ts_utc"):
            val = raw_packet.get(key)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                out[key] = raw_packet.get(f"_hold_{key}", 0.0 if key != "ts_utc" else "")
            else:
                out[key] = val
        if "ts_utc" not in out and "timestamp" in raw_packet:
            out["ts_utc"] = str(raw_packet["timestamp"])
        if "load_mw" not in out:
            out["load_mw"] = math.hypot(_f(out.get("vx"), 0.0), _f(out.get("vy"), 0.0))
        if "renewables_mw" not in out:
            out["renewables_mw"] = math.hypot(_f(out.get("x"), 0.0), _f(out.get("y"), 0.0))
        return out

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        """Compute reliability using position and velocity as OQE signals."""
        event = {
            "ts_utc": state.get("ts_utc", ""),
            "load_mw": math.hypot(_f(state.get("vx"), 0.0), _f(state.get("vy"), 0.0)),
            "renewables_mw": math.hypot(_f(state.get("x"), 0.0), _f(state.get("y"), 0.0)),
        }
        last_event = None
        if history:
            prev = history[-1]
            last_event = {
                "ts_utc": prev.get("ts_utc", ""),
                "load_mw": math.hypot(_f(prev.get("vx"), 0.0), _f(prev.get("vy"), 0.0)),
                "renewables_mw": math.hypot(_f(prev.get("x"), 0.0), _f(prev.get("y"), 0.0)),
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
        """Build a simple box uncertainty set around position and velocity."""
        x = _f(state.get("x"), 0.0)
        y = _f(state.get("y"), 0.0)
        vx = _f(state.get("vx"), 0.0)
        vy = _f(state.get("vy"), 0.0)
        w = max(0.05, float(reliability_w))
        margin = max(0.1, float(quantile) / (w + 1e-9))
        margin = min(margin, self._arena_size / 2.0)
        uncertainty = {
            "x_lower": x - margin,
            "x_upper": x + margin,
            "y_lower": y - margin,
            "y_upper": y + margin,
            "vx_lower": vx - margin,
            "vx_upper": vx + margin,
            "vy_lower": vy - margin,
            "vy_upper": vy + margin,
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
        """Return the navigation-safe action set representation."""
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
        """Project a candidate 2D action back into the safe set."""
        ax = _f(candidate_action.get("ax"), 0.0)
        ay = _f(candidate_action.get("ay"), 0.0)
        dt = _f(constraints.get("dt"), self._expected_cadence_s)
        arena_min = _f(constraints.get("arena_min"), 0.0)
        arena_max = _f(constraints.get("arena_max"), self._arena_size)
        speed_limit = _f(constraints.get("speed_limit"), self._speed_limit)

        mag = math.hypot(ax, ay)
        if mag > speed_limit > 0.0:
            scale = speed_limit / mag
            ax *= scale
            ay *= scale

        x = _f(state.get("x"), 0.0)
        y = _f(state.get("y"), 0.0)
        next_x = x + ax * dt
        next_y = y + ay * dt
        reason = None

        next_x = max(arena_min, min(arena_max, next_x))
        next_y = max(arena_min, min(arena_max, next_y))
        if abs(next_x - (x + ax * dt)) > 1e-9 or abs(next_y - (y + ay * dt)) > 1e-9:
            reason = "arena_clamp"

        for cx, cy in self._obstacle_centres:
            dx = next_x - cx
            dy = next_y - cy
            dist = math.hypot(dx, dy)
            if dist < self._obstacle_radius:
                if dist <= 1e-9:
                    dx, dy = 1.0, 0.0
                    dist = 1.0
                scale = self._obstacle_radius / dist
                next_x = cx + dx * scale
                next_y = cy + dy * scale
                reason = reason or "obstacle_avoidance"

        safe_ax = (next_x - x) / max(dt, 1e-9)
        safe_ay = (next_y - y) / max(dt, 1e-9)
        repaired = abs(safe_ax - _f(candidate_action.get("ax"), 0.0)) > 1e-9 or abs(
            safe_ay - _f(candidate_action.get("ay"), 0.0)
        ) > 1e-9
        meta = {
            "mode": "projection",
            "repaired": repaired,
            "original_ax": _f(candidate_action.get("ax"), 0.0),
            "original_ay": _f(candidate_action.get("ay"), 0.0),
        }
        if repaired:
            meta["intervention_reason"] = reason or "navigation_projection"
        return {"ax": float(safe_ax), "ay": float(safe_ay)}, meta

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
        """Emit a standard ORIUS certificate for navigation."""
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
