"""ORIUS-Bench vehicle track — 1D longitudinal AV domain.

Wraps VehiclePlant. Safety: speed limit, headway. Fault injection on position/speed.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from orius.orius_bench.adapter import BenchmarkAdapter
from orius.vehicles.plant import VehiclePlant


class VehicleTrackAdapter(BenchmarkAdapter):
    """1D longitudinal vehicle benchmark track (AV domain)."""

    def __init__(
        self,
        speed_limit_mps: float = 30.0,
        dt_s: float = 0.25,
        min_headway_m: float = 5.0,
    ):
        self._speed_limit = speed_limit_mps
        self._dt = dt_s
        self._headway = min_headway_m
        self._plant: VehiclePlant | None = None
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        self._plant = VehiclePlant(
            dt_s=self._dt,
            speed_limit_mps=self._speed_limit,
            min_headway_m=self._headway,
        )
        self._plant.reset(position_m=0.0, speed_mps=5.0, lead_position_m=50.0)
        return self.true_state()

    def true_state(self) -> Mapping[str, Any]:
        assert self._plant is not None
        s = self._plant.state()
        return {
            "position_m": s["position_m"],
            "speed_mps": s["speed_mps"],
            "speed_limit_mps": s["speed_limit_mps"],
            "lead_position_m": s["lead_position_m"],
        }

    def observe(
        self,
        true_state: Mapping[str, Any],
        fault: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        obs = dict(true_state)
        if fault is None:
            return obs
        kind = fault.get("kind", "")
        if kind == "blackout":
            return {k: float("nan") for k in obs}
        if kind == "bias":
            mag = fault.get("magnitude", 0)
            obs["position_m"] = obs["position_m"] + mag
            if obs.get("lead_position_m") is not None:
                obs["lead_position_m"] = obs["lead_position_m"] + mag
        elif kind == "noise":
            sigma = fault.get("sigma", 2.0)
            assert self._rng is not None
            obs["position_m"] = obs["position_m"] + float(self._rng.normal(0, sigma))
        elif kind == "stuck_sensor":
            fv = fault.get("frozen_value", 10.0)
            obs["position_m"] = fv
        return obs

    def safe_action_set(
        self,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {"speed_max_mps": self._speed_limit}

    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        assert self._plant is not None
        a = float(action.get("acceleration_mps2", 0.0))
        self._plant.step(a)
        return dict(self.true_state())

    def compute_useful_work(self, trajectory: Sequence[Mapping[str, Any]]) -> float:
        total = 0.0
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            cur = trajectory[i]
            dx = cur.get("position_m", 0) - prev.get("position_m", 0)
            if not math.isnan(dx) and dx > 0:
                v = self.check_violation(cur)
                if not v["violated"]:
                    total += dx
        return total

    def check_violation(self, state: Mapping[str, Any]) -> dict[str, Any]:
        v = state.get("speed_mps", 0.0)
        v_limit = state.get("speed_limit_mps", self._speed_limit)
        x = state.get("position_m", 0.0)
        lead_x = state.get("lead_position_m")
        violated = False
        severity = 0.0
        if v > v_limit + 1e-9:
            violated = True
            severity = max(severity, v - v_limit)
        if lead_x is not None:
            d_min = self._headway + 2.0 * v
            gap = lead_x - x
            if gap < d_min - 1e-9:
                violated = True
                severity = max(severity, d_min - gap)
        return {"violated": violated, "severity": severity}

    @property
    def domain_name(self) -> str:
        return "vehicle"
