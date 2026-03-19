"""ORIUS-Bench healthcare track — vital signs domain.

HR, SpO2, respiratory rate. Safety: SpO2 >= min, HR in range.
Fault injection: bias, noise, stuck_sensor on spo2_pct.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from orius.orius_bench.adapter import BenchmarkAdapter


class HealthcareTrackAdapter(BenchmarkAdapter):
    """Healthcare vital signs benchmark track."""

    def __init__(
        self,
        spo2_min_pct: float = 90.0,
        hr_min_bpm: float = 40.0,
        hr_max_bpm: float = 120.0,
        dt: float = 0.25,
    ):
        self._spo2_min = spo2_min_pct
        self._hr_min = hr_min_bpm
        self._hr_max = hr_max_bpm
        self._dt = dt
        self._spo2 = 97.0
        self._hr = 72.0
        self._rr = 14.0
        self._rng: np.random.Generator | None = None

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        # Start in mild hypoxia: SpO2=85 % — clinician must intervene to restore above 90 %
        self._spo2 = 85.0
        self._hr = 72.0
        self._rr = 14.0
        return self.true_state()

    def true_state(self) -> Mapping[str, Any]:
        return {
            "spo2_pct": float(self._spo2),
            "hr_bpm": float(self._hr),
            "respiratory_rate": float(self._rr),
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
        if kind == "bias" and "spo2_pct" in obs:
            obs["spo2_pct"] = obs["spo2_pct"] + fault.get("magnitude", 0)
        elif kind == "noise" and "spo2_pct" in obs:
            sigma = fault.get("sigma", 3.0)
            assert self._rng is not None
            obs["spo2_pct"] = obs["spo2_pct"] + float(self._rng.normal(0, sigma))
        elif kind == "stuck_sensor" and "spo2_pct" in obs:
            obs["spo2_pct"] = fault.get("frozen_value", 95.0)
        return obs

    def safe_action_set(
        self,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {"spo2_min_pct": self._spo2_min}

    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        alert = float(action.get("alert_level", 0.2))
        alert = max(0.0, min(1.0, alert))
        # Dynamics: high alert -> spo2 improves (intervention), low -> natural drift
        self._spo2 = self._spo2 + (2.0 * alert - 0.1) * self._dt
        self._spo2 = max(70.0, min(100.0, self._spo2))
        self._hr = 72.0 + 5.0 * (alert - 0.5)
        self._hr = max(self._hr_min, min(self._hr_max, self._hr))
        return dict(self.true_state())

    def compute_useful_work(self, trajectory: Sequence[Mapping[str, Any]]) -> float:
        total = 0.0
        for rec in trajectory:
            spo2 = rec.get("spo2_pct", 0)
            if not math.isnan(spo2) and spo2 >= self._spo2_min:
                total += 1.0
        return total

    def check_violation(self, state: Mapping[str, Any]) -> dict[str, Any]:
        spo2 = state.get("spo2_pct", 97.0)
        hr = state.get("hr_bpm", 72.0)
        violated = spo2 < self._spo2_min or hr < self._hr_min or hr > self._hr_max
        severity = 0.0
        if spo2 < self._spo2_min:
            severity = self._spo2_min - spo2
        elif hr < self._hr_min:
            severity = self._hr_min - hr
        elif hr > self._hr_max:
            severity = hr - self._hr_max
        return {"violated": violated, "severity": severity}

    @property
    def domain_name(self) -> str:
        return "healthcare"
