"""ORIUS-Bench aerospace track — flight envelope domain.

Airspeed, altitude, bank. Safety: airspeed in [v_min, v_max].
Fault injection: bias, noise, stuck_sensor on airspeed_kt.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from orius.orius_bench.adapter import BenchmarkAdapter


class AerospaceTrackAdapter(BenchmarkAdapter):
    """Aerospace flight envelope benchmark track."""

    def __init__(
        self,
        v_min_kt: float = 60.0,
        v_max_kt: float = 350.0,
        dt: float = 0.25,
    ):
        self._v_min = v_min_kt
        self._v_max = v_max_kt
        self._dt = dt
        self._airspeed = 180.0
        self._altitude = 3000.0
        self._bank = 5.0
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        self._airspeed = 180.0
        self._altitude = 3000.0
        self._bank = 5.0
        return self.true_state()

    def true_state(self) -> Mapping[str, Any]:
        return {
            "airspeed_kt": float(self._airspeed),
            "altitude_m": float(self._altitude),
            "bank_angle_deg": float(self._bank),
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
        if kind == "bias" and "airspeed_kt" in obs:
            obs["airspeed_kt"] = obs["airspeed_kt"] + fault.get("magnitude", 0)
        elif kind == "noise" and "airspeed_kt" in obs:
            sigma = fault.get("sigma", 10.0)
            assert self._rng is not None
            obs["airspeed_kt"] = obs["airspeed_kt"] + float(self._rng.normal(0, sigma))
        elif kind == "stuck_sensor" and "airspeed_kt" in obs:
            obs["airspeed_kt"] = fault.get("frozen_value", 180.0)
        return obs

    def safe_action_set(
        self,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {"v_min_kt": self._v_min, "v_max_kt": self._v_max}

    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        throttle = float(action.get("throttle", 0.7))
        bank = float(action.get("bank_deg", 3.0))
        throttle = max(0.0, min(1.0, throttle))
        bank = max(-30.0, min(30.0, bank))
        self._airspeed = self._airspeed + (throttle - 0.5) * 20.0 * self._dt
        self._airspeed = max(0.0, min(400.0, self._airspeed))
        self._bank = bank
        self._altitude = self._altitude + 50.0 * (throttle - 0.4) * self._dt
        self._altitude = max(0.0, self._altitude)
        return dict(self.true_state())

    def compute_useful_work(self, trajectory: Sequence[Mapping[str, Any]]) -> float:
        total = 0.0
        for rec in trajectory:
            v = rec.get("airspeed_kt", 0)
            if not math.isnan(v) and self._v_min <= v <= self._v_max:
                total += v
        return total

    def check_violation(self, state: Mapping[str, Any]) -> dict[str, Any]:
        v = state.get("airspeed_kt", 180.0)
        violated = v < self._v_min or v > self._v_max
        severity = 0.0
        if v < self._v_min:
            severity = self._v_min - v
        elif v > self._v_max:
            severity = v - self._v_max
        return {"violated": violated, "severity": severity}

    @property
    def domain_name(self) -> str:
        return "aerospace"
