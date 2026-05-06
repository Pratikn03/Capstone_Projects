"""Navigation domain adapter for the ORIUS universal runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from orius.dc3s.certificate import make_certificate
from orius.dc3s.domain_adapter import DomainAdapter
from orius.universal_framework.reliability_runtime import assess_domain_reliability
from orius.universal_framework.runtime_evidence import resolve_runtime_evidence


def _f(value: Any, default: float) -> float:
    try:
        v = float(value)
        if not math.isfinite(v):
            return float(default)
        return v
    except (TypeError, ValueError):
        return float(default)


class NavigationDomainAdapter(DomainAdapter):
    """DomainAdapter for bounded 2D navigation and guidance."""

    def __init__(self, cfg: Mapping[str, Any] | None = None):
        self._cfg = dict(cfg or {})
        self.domain_id = "navigation"
        nav_cfg = self._cfg.get("navigation", {})
        self._arena_size = _f(nav_cfg.get("arena_size"), 10.0)
        self._speed_limit = _f(nav_cfg.get("speed_limit"), 1.0)
        self._dt_s = _f(nav_cfg.get("dt_s", nav_cfg.get("dt")), 0.25)
        self._boundary_margin = _f(nav_cfg.get("boundary_margin"), 0.05)
        self._obstacle_radius = _f(nav_cfg.get("obstacle_radius"), 1.0)
        centres = nav_cfg.get("obstacle_centres", [(5.0, 5.0)])
        self._obstacle_centres = [
            (float(pair[0]), float(pair[1]))
            for pair in centres
            if isinstance(pair, Sequence) and len(pair) == 2
        ]
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 0.25)
        evidence = resolve_runtime_evidence(self.domain_id, self._cfg)
        self._runtime_surface = evidence.runtime_surface
        self._closure_tier = evidence.closure_tier
        self._maturity_tier = evidence.maturity_tier
        self._fallback_policy = evidence.fallback_policy
        self._exact_blocker = evidence.exact_blocker

    def capability_profile(self) -> Mapping[str, Any]:
        return {
            "safety_surface_type": "arena_obstacle_bounds",
            "repair_mode": "vector_projection",
            "fallback_mode": "hold_position",
            "supports_multi_agent_eval": False,
            "supports_certos_eval": True,
            "runtime_surface": self._runtime_surface,
            "closure_tier": self._closure_tier,
            "maturity_tier": self._maturity_tier,
            "fallback_policy": self._fallback_policy,
            "exact_blocker": self._exact_blocker,
        }

    def _in_safe_region(self, x: float, y: float) -> bool:
        lo = self._boundary_margin
        hi = self._arena_size - self._boundary_margin
        if not (lo <= x <= hi and lo <= y <= hi):
            return False
        return all(math.hypot(x - cx, y - cy) >= self._obstacle_radius for cx, cy in self._obstacle_centres)

    def true_constraint_violated(self, state: Mapping[str, Any]) -> bool | None:
        return not self._in_safe_region(_f(state.get("x"), 0.0), _f(state.get("y"), 0.0))

    def observed_constraint_satisfied(self, observed_state: Mapping[str, Any]) -> bool | None:
        return self._in_safe_region(
            _f(observed_state.get("x"), 0.0),
            _f(observed_state.get("y"), 0.0),
        )

    def constraint_margin(self, state: Mapping[str, Any]) -> float | None:
        x = _f(state.get("x"), 0.0)
        y = _f(state.get("y"), 0.0)
        lo = self._boundary_margin
        hi = self._arena_size - self._boundary_margin
        boundary_margin = min(x - lo, hi - x, y - lo, hi - y)
        obstacle_margin = min(
            (math.hypot(x - cx, y - cy) - self._obstacle_radius for cx, cy in self._obstacle_centres),
            default=boundary_margin,
        )
        return float(min(boundary_margin, obstacle_margin))

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
        w_t, flags = assess_domain_reliability(
            domain_id=self.domain_id,
            state=state,
            history=history,
            feature_sources={
                "x_position_m": "x",
                "y_position_m": "y",
                "x_velocity_mps": "vx",
                "y_velocity_mps": "vy",
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
        x = _f(state.get("x"), 0.0)
        y = _f(state.get("y"), 0.0)
        vx = _f(state.get("vx"), 0.0)
        vy = _f(state.get("vy"), 0.0)
        w = max(0.05, float(reliability_w))
        inflation = max(1.0, 1.0 / w)
        margin = min(max(0.05, (float(quantile) / 100.0) * inflation), self._arena_size / 2.0)
        uncertainty = {
            "x_lower": x - margin,
            "x_upper": x + margin,
            "y_lower": y - margin,
            "y_upper": y + margin,
            "vx_lower": vx - margin,
            "vx_upper": vx + margin,
            "vy_lower": vy - margin,
            "vy_upper": vy + margin,
            "margin": margin,
            "meta": {
                "inflation": inflation,
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
        unc = dict(uncertainty)
        cstr = dict(constraints)
        speed_limit = _f(cstr.get("max_speed", cstr.get("speed_limit")), self._speed_limit)
        arena_min = _f(cstr.get("arena_min"), self._boundary_margin)
        arena_max = _f(cstr.get("arena_max"), self._arena_size - self._boundary_margin)
        dt = _f(cstr.get("dt_s", cfg.get("dt", cfg.get("dt_s"))), self._dt_s)
        x_hi = _f(unc.get("x_upper"), 0.0)
        x_lo = _f(unc.get("x_lower"), 0.0)
        y_hi = _f(unc.get("y_upper"), 0.0)
        y_lo = _f(unc.get("y_lower"), 0.0)
        # Add a one-step speed buffer so the constraint is conservative w.r.t.
        # the true state potentially being at the outer edge of the OCSS.
        # Without this buffer, stale telemetry can cause the repaired action to
        # push the true state past the arena boundary even though the observed
        # upper-bound stays inside.
        speed_buffer = speed_limit * dt
        ax_lower = max(-speed_limit, (arena_min + speed_buffer - x_lo) / max(dt, 1e-9))
        ax_upper = min(speed_limit, (arena_max - speed_buffer - x_hi) / max(dt, 1e-9))
        ay_lower = max(-speed_limit, (arena_min + speed_buffer - y_lo) / max(dt, 1e-9))
        ay_upper = min(speed_limit, (arena_max - speed_buffer - y_hi) / max(dt, 1e-9))
        viable = ax_lower <= ax_upper + 1e-9 and ay_lower <= ay_upper + 1e-9
        if not viable:
            ax_lower = ax_upper = 0.0
            ay_lower = ay_upper = 0.0
            viable = True
        return {
            "uncertainty": unc,
            "constraints": cstr,
            "cfg": dict(cfg),
            "ax_lower": float(ax_lower),
            "ax_upper": float(ax_upper),
            "ay_lower": float(ay_lower),
            "ay_upper": float(ay_upper),
            "fallback_action": {"ax": 0.0, "ay": 0.0},
            "projection_surface": "arena_obstacle_projection",
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
        if "ax_lower" not in tightened_set or "ax_upper" not in tightened_set:
            tightened_set = self.tighten_action_set(
                uncertainty=tightened_set.get("uncertainty", uncertainty),
                constraints=tightened_set.get("constraints", constraints),
                cfg=cfg,
            )
        unc = tightened_set.get("uncertainty", uncertainty)
        cstr = tightened_set.get("constraints", constraints)

        orig_ax = _f(candidate_action.get("ax"), 0.0)
        orig_ay = _f(candidate_action.get("ay"), 0.0)
        ax = max(
            _f(tightened_set.get("ax_lower"), -self._speed_limit),
            min(_f(tightened_set.get("ax_upper"), self._speed_limit), orig_ax),
        )
        ay = max(
            _f(tightened_set.get("ay_lower"), -self._speed_limit),
            min(_f(tightened_set.get("ay_upper"), self._speed_limit), orig_ay),
        )

        speed_limit = _f(cstr.get("max_speed", cstr.get("speed_limit")), self._speed_limit)
        arena_min = _f(cstr.get("arena_min"), self._boundary_margin)
        arena_max = _f(cstr.get("arena_max"), self._arena_size - self._boundary_margin)
        obstacle_radius = _f(cstr.get("obstacle_radius"), self._obstacle_radius)
        dt = _f(cstr.get("dt_s", cfg.get("dt", cfg.get("dt_s"))), self._dt_s)

        reason_parts: list[str] = []

        mag = math.hypot(ax, ay)
        if mag > speed_limit > 0:
            scale = speed_limit / mag
            ax *= scale
            ay *= scale
            reason_parts.append("speed_limit")

        x_hi = _f(unc.get("x_upper"), _f(state.get("x"), 0.0))
        x_lo = _f(unc.get("x_lower"), _f(state.get("x"), 0.0))
        y_hi = _f(unc.get("y_upper"), _f(state.get("y"), 0.0))
        y_lo = _f(unc.get("y_lower"), _f(state.get("y"), 0.0))

        if x_hi + ax * dt > arena_max:
            ax = min(ax, (arena_max - x_hi) / max(dt, 1e-9))
            reason_parts.append("arena_x_max")
        if x_lo + ax * dt < arena_min:
            ax = max(ax, (arena_min - x_lo) / max(dt, 1e-9))
            reason_parts.append("arena_x_min")
        if y_hi + ay * dt > arena_max:
            ay = min(ay, (arena_max - y_hi) / max(dt, 1e-9))
            reason_parts.append("arena_y_max")
        if y_lo + ay * dt < arena_min:
            ay = max(ay, (arena_min - y_lo) / max(dt, 1e-9))
            reason_parts.append("arena_y_min")

        next_x = _f(state.get("x"), 0.0) + ax * dt
        next_y = _f(state.get("y"), 0.0) + ay * dt
        obstacle_hit = False
        for cx, cy in self._obstacle_centres:
            dist = math.hypot(next_x - cx, next_y - cy)
            if dist < obstacle_radius:
                obstacle_hit = True
                away_x = next_x - cx
                away_y = next_y - cy
                away_mag = math.hypot(away_x, away_y)
                if away_mag > 1e-9:
                    scale = min(speed_limit, obstacle_radius / max(dt, 1e-9)) / away_mag
                    ax = away_x * scale
                    ay = away_y * scale
                else:
                    ax = 0.0
                    ay = 0.0
                reason_parts.append("obstacle_avoidance")
                break

        repaired = abs(ax - orig_ax) > 1e-9 or abs(ay - orig_ay) > 1e-9
        meta = {
            "mode": "projection",
            "repaired": repaired,
            "original_ax": orig_ax,
            "original_ay": orig_ay,
            "obstacle_hit_predicted": obstacle_hit,
            "repair_surface": str(tightened_set.get("projection_surface", "arena_obstacle_projection")),
        }
        if repaired:
            meta["intervention_reason"] = ",".join(reason_parts) if reason_parts else "arena_bound_clamp"
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
            model_hash=str(cfg.get("model_hash", "")),
            config_hash=str(cfg.get("config_hash", "")),
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
