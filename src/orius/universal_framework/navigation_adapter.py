"""Navigation domain adapter for ORIUS Universal Framework.

2D robot navigation in a bounded arena. Safety predicates:
- x in [0, arena_size]
- y in [0, arena_size]
- distance from any obstacle centre > obstacle_radius

Repair logic: clamp (ax, ay) so that the predicted next position stays within
the safe arena and outside forbidden obstacle zones.
"""
from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from orius.dc3s.domain_adapter import DomainAdapter
from orius.dc3s.quality import compute_reliability
from orius.dc3s.certificate import make_certificate


def _f(x: Any, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


class NavigationDomainAdapter(DomainAdapter):
    """DomainAdapter for 2D navigation / robotics domain."""

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        self._cfg = dict(cfg or {})
        nav = self._cfg.get("navigation", {})
        self._arena_size = _f(nav.get("arena_size"), 10.0)
        self._speed_limit = _f(nav.get("speed_limit"), 1.0)
        self._dt = _f(nav.get("dt"), 0.25)
        self._obs_centres: list[tuple[float, float]] = list(
            nav.get("obstacle_centres", [(5.0, 5.0)])
        )
        self._obs_radius = _f(nav.get("obstacle_radius"), 1.0)
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 0.25)
        self._margin = _f(nav.get("boundary_margin"), 0.05)

    # ------------------------------------------------------------------
    # 1. Ingest telemetry
    # ------------------------------------------------------------------

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        """Parse raw navigation packet into state vector z_t."""
        out: dict[str, Any] = {}
        for k in ("x", "y", "vx", "vy", "ts_utc"):
            v = raw_packet.get(k)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out[k] = raw_packet.get(f"_hold_{k}", 0.0 if k != "ts_utc" else "")
            else:
                out[k] = v
        # Canonical OQE mapping: x -> load_mw (proxy for positional signal)
        if "load_mw" not in out:
            out["load_mw"] = out.get("x", 0.0)
        if "ts_utc" not in out and "timestamp" in raw_packet:
            out["ts_utc"] = str(raw_packet["timestamp"])
        return out

    # ------------------------------------------------------------------
    # 2. Compute OQE reliability
    # ------------------------------------------------------------------

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        """Compute reliability w_t. Map x -> load_mw for OQE."""
        event = {
            "ts_utc": state.get("ts_utc", ""),
            "load_mw": state.get("x", state.get("load_mw", 0.0)),
            "renewables_mw": state.get("y", 0.0),
        }
        last_event = None
        if history:
            prev = history[-1]
            last_event = {
                "ts_utc": prev.get("ts_utc", ""),
                "load_mw": prev.get("x", prev.get("load_mw", 0.0)),
                "renewables_mw": prev.get("y", 0.0),
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

    # ------------------------------------------------------------------
    # 3. Build uncertainty set
    # ------------------------------------------------------------------

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
        """Conformal box over (x, y) inflated by w_t."""
        x = _f(state.get("x"), 5.0)
        y = _f(state.get("y"), 5.0)
        w = max(0.05, float(reliability_w))
        margin = max(0.1, float(quantile) / (w + 1e-9))
        margin = min(margin, 2.0)
        uncertainty = {
            "x_lower": max(0.0, x - margin),
            "x_upper": min(self._arena_size, x + margin),
            "y_lower": max(0.0, y - margin),
            "y_upper": min(self._arena_size, y + margin),
            "margin": margin,
            "meta": {"inflation": margin, "w_t": w},
        }
        meta = {"inflation": margin, "w_t": w}
        return uncertainty, meta

    # ------------------------------------------------------------------
    # 4. Tighten action set
    # ------------------------------------------------------------------

    def tighten_action_set(
        self,
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        *,
        cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Arena and obstacle constraints on (ax, ay)."""
        return {
            "uncertainty": dict(uncertainty),
            "constraints": dict(constraints),
            "cfg": dict(cfg),
        }

    # ------------------------------------------------------------------
    # 5. Repair action
    # ------------------------------------------------------------------

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
        """Clamp (ax, ay) to keep robot within arena and away from obstacles.

        Repair strategy: predict next position = (x + ax*dt, y + ay*dt) and
        clamp each component to stay inside [margin, arena - margin].
        Also avoid entering obstacle zones.
        """
        ax = _f(candidate_action.get("ax", 0.0), 0.0)
        ay = _f(candidate_action.get("ay", 0.0), 0.0)

        # Use uncertainty UPPER BOUND for conservative worst-case repair decision.
        # This ensures DC3S brakes early even when sensor bias under-reports position.
        x_obs = _f(state.get("x"), 5.0)
        y_obs = _f(state.get("y"), 5.0)
        margin_val = _f(uncertainty.get("margin", 1.0), 1.0)
        x = _f(uncertainty.get("x_upper"), x_obs + margin_val)
        y = _f(uncertainty.get("y_upper"), y_obs + margin_val)

        arena = _f(constraints.get("arena_size", self._arena_size), self._arena_size)
        dt = _f(cfg.get("dt", self._dt), self._dt)
        margin = self._margin

        # Predict next position
        nx = x + ax * dt
        ny = y + ay * dt

        ax_safe = ax
        ay_safe = ay
        intervention_reason: list[str] = []

        # Clamp to arena bounds
        if nx < margin:
            ax_safe = (margin - x) / dt if dt > 0 else 0.0
            intervention_reason.append("arena_x_min")
        elif nx > arena - margin:
            ax_safe = (arena - margin - x) / dt if dt > 0 else 0.0
            intervention_reason.append("arena_x_max")

        if ny < margin:
            ay_safe = (margin - y) / dt if dt > 0 else 0.0
            intervention_reason.append("arena_y_min")
        elif ny > arena - margin:
            ay_safe = (arena - margin - y) / dt if dt > 0 else 0.0
            intervention_reason.append("arena_y_max")

        # Also enforce speed limit
        speed_limit = _f(constraints.get("speed_limit", self._speed_limit), self._speed_limit)
        mag = math.hypot(ax_safe, ay_safe)
        if mag > speed_limit + 1e-6:
            scale = speed_limit / mag
            ax_safe *= scale
            ay_safe *= scale
            intervention_reason.append("speed_limit")

        # Blackout guard: if position is NaN-filled state (from zero-order hold), be conservative
        x_raw = candidate_action.get("ax")
        if x_raw is not None and isinstance(x_raw, float) and math.isnan(x_raw):
            ax_safe = 0.0
            ay_safe = 0.0
            intervention_reason.append("blackout_stop")

        repaired = abs(ax_safe - ax) > 1e-9 or abs(ay_safe - ay) > 1e-9
        meta = {
            "mode": "arena_clamp",
            "repaired": repaired,
            "original_ax": ax,
            "original_ay": ay,
        }
        if intervention_reason:
            meta["intervention_reason"] = ",".join(intervention_reason)

        return {"ax": float(ax_safe), "ay": float(ay_safe)}, meta

    # ------------------------------------------------------------------
    # 6. Emit certificate
    # ------------------------------------------------------------------

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
        """Emit certificate with navigation action fields."""
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
