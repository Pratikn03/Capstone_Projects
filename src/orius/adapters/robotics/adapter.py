"""ROS 2 Mobile Robot Domain Adapter for ORIUS/DC3S.

Implements obstacle collision avoidance and stopping-distance viability
under LiDAR and localization telemetry degradation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from orius.dc3s.domain_adapter import DomainAdapter


class ROS2RoboticsAdapter(DomainAdapter):
    """Adapter for embodied mobile robot velocity command shielding."""

    def __init__(
        self,
        v_max: float = 1.5,
        w_max: float = 1.0,
        a_brake: float = 2.0,
        d_margin: float = 0.3,
    ) -> None:
        self.v_max = v_max
        self.w_max = w_max
        self.a_brake = a_brake
        self.d_margin = d_margin

    def capability_profile(self) -> Mapping[str, Any]:
        return {
            "safety_surface_type": "obstacle_stopping_viability",
            "repair_mode": "cbf_velocity_projection",
            "fallback_mode": "emergency_brake_hold",
            "supports_multi_agent_eval": True,
            "supports_certos_eval": True,
        }

    def true_constraint_violated(self, state: Mapping[str, Any]) -> bool | None:
        d_obs = float(state.get("obstacle_distance_m", 10.0))
        v = float(state.get("linear_velocity_mps", 0.0))
        stopping_dist = (v ** 2) / (2.0 * self.a_brake)
        return bool(d_obs < stopping_dist + self.d_margin - 1e-4)

    def observed_constraint_satisfied(self, observed_state: Mapping[str, Any]) -> bool | None:
        violated = self.true_constraint_violated(observed_state)
        return not violated if violated is not None else None

    def constraint_margin(self, state: Mapping[str, Any]) -> float | None:
        d_obs = float(state.get("obstacle_distance_m", 10.0))
        v = float(state.get("linear_velocity_mps", 0.0))
        stopping_dist = (v ** 2) / (2.0 * self.a_brake)
        return float(d_obs - (stopping_dist + self.d_margin))

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "obstacle_distance_m": float(raw_packet.get("scan_min_dist", 10.0)),
            "linear_velocity_mps": float(raw_packet.get("odom_v", 0.0)),
            "angular_velocity_radps": float(raw_packet.get("odom_w", 0.0)),
            "lidar_valid_points": int(raw_packet.get("valid_points", 360)),
            "is_dropped": bool(raw_packet.get("is_dropped", False)),
            "jitter_ms": float(raw_packet.get("jitter_ms", 2.0)),
        }

    def compute_oqe(
        self, state_history: Sequence[Mapping[str, Any]]
    ) -> tuple[float, Mapping[str, Any]]:
        if not state_history:
            return 1.0, {"flags": []}

        latest = state_history[-1]
        score = 1.0
        flags = []

        if latest.get("is_dropped"):
            flags.append("scan_dropped")
            score *= 0.3

        pts = int(latest.get("lidar_valid_points", 360))
        if pts < 100:
            flags.append("sparse_pointcloud")
            score *= max(0.4, pts / 100.0)

        jitter = float(latest.get("jitter_ms", 2.0))
        if jitter > 20.0:
            flags.append("high_ros2_jitter")
            score *= max(0.3, 1.0 - (jitter - 20.0) / 40.0)

        return float(max(0.0, min(1.0, score))), {"flags": flags}

    def build_uncertainty_set(
        self,
        state: Mapping[str, Any],
        oqe_score: float,
        conformal_quantile: float = 1.96,
    ) -> Mapping[str, Any]:
        d_obs = float(state.get("obstacle_distance_m", 10.0))
        inflation = 1.0 + (1.0 - oqe_score) * 2.5
        delta_d = 0.05 * conformal_quantile * inflation

        # Conservative lower bound on obstacle distance
        d_min_obs = max(0.0, d_obs - delta_d)
        return {
            "d_min_obs": d_min_obs,
            "d_obs_raw": d_obs,
            "inflation": inflation,
        }

    def tighten_action_set(
        self,
        uncertainty_set: Mapping[str, Any],
        constraints: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        d_min_obs = float(uncertainty_set["d_min_obs"])
        available_distance = max(0.0, d_min_obs - self.d_margin)

        # Viability: v_max_safe = sqrt(2 * a_brake * available_distance)
        v_safe_limit = min(self.v_max, math.sqrt(2.0 * self.a_brake * available_distance))

        return {
            "v_min": 0.0,
            "v_max": v_safe_limit,
            "w_min": -self.w_max,
            "w_max": self.w_max,
            "is_empty": bool(v_safe_limit <= 1e-4),
        }

    def repair_action(
        self,
        candidate_action: Mapping[str, Any],
        admissible_set: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        cand_v = float(candidate_action.get("v_cmd", 0.0))
        cand_w = float(candidate_action.get("w_cmd", 0.0))

        if admissible_set.get("is_empty"):
            # Emergency brake-hold
            return {"v_cmd": 0.0, "w_cmd": 0.0}, {"repaired": True, "fallback": True}

        v_max = float(admissible_set["v_max"])
        safe_v = max(0.0, min(v_max, cand_v))
        safe_w = max(-self.w_max, min(self.w_max, cand_w))
        intervened = not math.isclose(safe_v, cand_v, abs_tol=1e-4)

        return {"v_cmd": safe_v, "w_cmd": safe_w}, {"repaired": intervened, "fallback": False}

    def emit_certificate(self, record: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "domain": "robotics_ros2",
            "schema_version": "1.0",
            "certified": True,
            "ttl_steps": 1,
            "payload": dict(record),
        }
