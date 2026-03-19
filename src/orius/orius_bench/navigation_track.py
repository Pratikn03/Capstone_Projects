"""ORIUS-Bench navigation track — toy 2D robot domain.

A point robot moves in a bounded 2D arena. Safety constraint: stay
inside the arena and outside forbidden zones (circular obstacles).
Sensor faults add bias/noise to position readings. The track
demonstrates that the ORIUS metrics generalise beyond batteries.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from orius.orius_bench.adapter import BenchmarkAdapter


class NavigationTrackAdapter(BenchmarkAdapter):
    """2D navigation benchmark track."""

    def __init__(
        self,
        arena_size: float = 10.0,
        speed_limit: float = 1.0,
        obstacle_centres: Sequence[tuple[float, float]] | None = None,
        obstacle_radius: float = 1.0,
        dt: float = 0.25,
    ):
        self._arena = arena_size
        self._speed = speed_limit
        self._obs_centres = list(obstacle_centres or [(5.0, 5.0)])
        self._obs_r = obstacle_radius
        self._dt = dt
        self._pos = np.zeros(2)
        self._vel = np.zeros(2)
        self._rng: np.random.Generator | None = None

    # -- BenchmarkAdapter ------------------------------------------------

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        # Start near x-boundary so an aggressive nominal action quickly causes violations
        self._pos = np.array([9.0, 1.0])
        self._vel = np.zeros(2)
        return self.true_state()

    def true_state(self) -> Mapping[str, Any]:
        return {
            "x": float(self._pos[0]),
            "y": float(self._pos[1]),
            "vx": float(self._vel[0]),
            "vy": float(self._vel[1]),
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
            obs["x"] = obs["x"] + mag
            obs["y"] = obs["y"] + mag
        elif kind == "noise":
            sigma = fault.get("sigma", 0.5)
            assert self._rng is not None
            obs["x"] = obs["x"] + float(self._rng.normal(0, sigma))
            obs["y"] = obs["y"] + float(self._rng.normal(0, sigma))
        elif kind == "stuck_sensor":
            fv = fault.get("frozen_value", 5.0)
            obs["x"] = fv
            obs["y"] = fv
        return obs

    def safe_action_set(
        self,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {
            "max_speed": self._speed,
            "arena_min": 0.0,
            "arena_max": self._arena,
        }

    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        ax = float(action.get("ax", 0))
        ay = float(action.get("ay", 0))
        # Clip acceleration
        mag = math.hypot(ax, ay)
        if mag > self._speed:
            scale = self._speed / mag
            ax *= scale
            ay *= scale

        self._vel = np.array([ax, ay])
        self._pos = self._pos + self._vel * self._dt
        # Physics — no clamping (like BatteryPlant)
        return dict(self.true_state())

    def compute_useful_work(self, trajectory: Sequence[Mapping[str, Any]]) -> float:
        """Useful work = total distance covered while safe."""
        total = 0.0
        for i in range(1, len(trajectory)):
            prev = trajectory[i - 1]
            cur = trajectory[i]
            dx = cur.get("x", 0) - prev.get("x", 0)
            dy = cur.get("y", 0) - prev.get("y", 0)
            if not (math.isnan(dx) or math.isnan(dy)):
                # Only count if current state is not violated
                v = self.check_violation(cur)
                if not v["violated"]:
                    total += math.hypot(dx, dy)
        return total

    def check_violation(self, state: Mapping[str, Any]) -> dict[str, Any]:
        x = state.get("x", 0)
        y = state.get("y", 0)
        severity = 0.0
        violated = False

        # Arena bounds
        for coord in [x, y]:
            if coord < 0:
                violated = True
                severity = max(severity, -coord)
            elif coord > self._arena:
                violated = True
                severity = max(severity, coord - self._arena)

        # Obstacles
        for cx, cy in self._obs_centres:
            dist = math.hypot(x - cx, y - cy)
            if dist < self._obs_r:
                violated = True
                severity = max(severity, self._obs_r - dist)

        return {"violated": violated, "severity": severity}

    @property
    def domain_name(self) -> str:
        return "navigation"
