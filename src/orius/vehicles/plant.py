"""1D longitudinal vehicle plant for ORIUS prototype.

Simple discrete-time integrator: position and speed along a lane.
Safety: speed limit and TTC-style lead-vehicle barrier.
"""
from __future__ import annotations

from typing import Any, Mapping


def _f(x: Any, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


class VehiclePlant:
    """1D longitudinal dynamics: x, v with acceleration command."""

    def __init__(
        self,
        dt_s: float = 0.25,
        speed_limit_mps: float = 30.0,
        speed_phys_max_mps: float = 50.0,
        accel_min_mps2: float = -5.0,
        accel_max_mps2: float = 3.0,
        min_headway_m: float = 5.0,
        headway_time_s: float = 2.0,
        ttc_min_s: float | None = None,
    ):
        self._dt = dt_s
        self._v_limit = speed_limit_mps
        self._v_phys = speed_phys_max_mps
        self._a_min = accel_min_mps2
        self._a_max = accel_max_mps2
        self._d_min = min_headway_m
        self._ttc_min_s = float(ttc_min_s if ttc_min_s is not None else headway_time_s)
        self._x = 0.0
        self._v = 0.0
        self._lead_x: float | None = None
        self._v_limit_t: float = speed_limit_mps

    def reset(
        self,
        position_m: float = 0.0,
        speed_mps: float = 0.0,
        lead_position_m: float | None = None,
        speed_limit_mps: float | None = None,
    ) -> Mapping[str, Any]:
        self._x = float(position_m)
        self._v = float(speed_mps)
        self._lead_x = float(lead_position_m) if lead_position_m is not None else None
        self._v_limit_t = float(speed_limit_mps) if speed_limit_mps is not None else self._v_limit
        return self.state()

    def state(self) -> Mapping[str, Any]:
        return {
            "position_m": self._x,
            "speed_mps": self._v,
            "speed_limit_mps": self._v_limit_t,
            "lead_position_m": self._lead_x,
        }

    def set_lead(self, lead_position_m: float | None) -> None:
        self._lead_x = lead_position_m

    def set_speed_limit(self, v_mps: float) -> None:
        self._v_limit_t = v_mps

    def step(self, acceleration_mps2: float) -> Mapping[str, Any]:
        a = max(self._a_min, min(self._a_max, float(acceleration_mps2)))
        self._v = max(0.0, min(self._v_phys, self._v + a * self._dt))
        self._x = self._x + self._v * self._dt
        return self.state()

    def check_violation(self) -> dict[str, Any]:
        violated = False
        severity = 0.0
        if self._v > self._v_limit_t + 1e-9:
            violated = True
            severity = max(severity, self._v - self._v_limit_t)
        if self._lead_x is not None:
            gap = self._lead_x - self._x
            gap_budget = gap - self._d_min
            if gap_budget <= 0.0:
                violated = True
                severity = max(severity, abs(gap_budget))
            else:
                ttc = gap_budget / max(self._v, 1e-9)
                if ttc < self._ttc_min_s - 1e-9:
                    violated = True
                    severity = max(severity, self._ttc_min_s - ttc)
        return {"violated": violated, "severity": severity}
