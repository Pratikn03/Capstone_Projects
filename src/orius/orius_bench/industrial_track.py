"""ORIUS-Bench industrial track — process control domain.

Temperature, pressure, power. Safety: temp in bounds, power below max.
Fault injection: bias, noise, stuck_sensor on primary state (temp_c).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from orius.orius_bench.adapter import BenchmarkAdapter


class IndustrialTrackAdapter(BenchmarkAdapter):
    """Industrial process control benchmark track."""

    def __init__(
        self,
        temp_min_c: float = 0.0,
        temp_max_c: float = 120.0,
        power_max_mw: float = 500.0,
        dt: float = 0.25,
    ):
        self._temp_min = temp_min_c
        self._temp_max = temp_max_c
        self._power_max = power_max_mw
        self._dt = dt
        self._temp = 85.0
        self._power = 450.0
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        # Start near upper temperature bound so over-limit setpoints quickly cause violations
        self._temp = 110.0
        self._power = 450.0
        return self.true_state()

    def true_state(self) -> Mapping[str, Any]:
        return {
            "temp_c": float(self._temp),
            "power_mw": float(self._power),
            "pressure_mbar": 1010.0,
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
        if kind == "bias" and "temp_c" in obs:
            obs["temp_c"] = obs["temp_c"] + fault.get("magnitude", 0)
        elif kind == "noise" and "temp_c" in obs:
            sigma = fault.get("sigma", 5.0)
            assert self._rng is not None
            obs["temp_c"] = obs["temp_c"] + float(self._rng.normal(0, sigma))
        elif kind == "stuck_sensor" and "temp_c" in obs:
            obs["temp_c"] = fault.get("frozen_value", 85.0)
        return obs

    def safe_action_set(
        self,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {"power_max_mw": self._power_max}

    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        setpoint = float(action.get("power_setpoint_mw", 450.0))
        # No internal clipping: violations manifest when setpoint exceeds power_max.
        # DC3S repair is the safety layer that prevents this from happening.
        self._power = setpoint
        self._temp = self._temp + 0.1 * (setpoint / 10.0 - self._temp) * self._dt
        self._temp = max(self._temp_min - 20, min(self._temp_max + 20, self._temp))
        return dict(self.true_state())

    def compute_useful_work(self, trajectory: Sequence[Mapping[str, Any]]) -> float:
        total = 0.0
        for rec in trajectory:
            p = rec.get("power_mw", 0)
            if not math.isnan(p):
                total += max(0.0, p)
        return total

    def check_violation(self, state: Mapping[str, Any]) -> dict[str, Any]:
        temp = state.get("temp_c", 85.0)
        power = state.get("power_mw", 0.0)
        violated = temp < self._temp_min or temp > self._temp_max or power > self._power_max
        severity = 0.0
        if temp < self._temp_min:
            severity = self._temp_min - temp
        elif temp > self._temp_max:
            severity = temp - self._temp_max
        elif power > self._power_max:
            severity = power - self._power_max
        return {"violated": violated, "severity": severity}

    @property
    def domain_name(self) -> str:
        return "industrial"
