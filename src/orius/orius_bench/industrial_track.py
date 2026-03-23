"""ORIUS-Bench industrial track — process control domain.

Temperature, pressure, power. Safety: temp in bounds, power below max.
Fault injection: bias, noise, stuck_sensor on primary state (temp_c).

Real-data mode
--------------
Pass ``dataset_path`` to load real UCI CCPP rows.  ``reset()`` selects a
near-limit operating point from the real data to initialise the synthetic
state (temp near 110 °C, power from real PE value).  All subsequent
``step()`` calls use synthetic dynamics so DC3S repair can demonstrate
improvement.  Real data provides realistic initial conditions only.
"""
from __future__ import annotations

import math
from pathlib import Path
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
        dataset_path: str | Path | None = None,
    ):
        self._temp_min = temp_min_c
        self._temp_max = temp_max_c
        self._power_max = power_max_mw
        self._dt = dt
        self._temp = 85.0
        self._power = 450.0
        self._rng: np.random.Generator | None = None
        # Real-data mode: rows used for initialization only
        self._real_rows: list[dict[str, float]] = []
        if dataset_path is not None:
            from orius.orius_bench.real_data_loader import load_ccpp_rows
            self._real_rows = load_ccpp_rows(Path(dataset_path))

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        if self._real_rows:
            # Select near-limit rows (high PE ≥ 480 MW) for realistic near-violation starts
            near_limit = [r for r in self._real_rows if r.get("PE", 0) >= 480.0]
            if not near_limit:
                near_limit = self._real_rows
            rng_idx = np.random.default_rng(seed)
            idx = int(rng_idx.integers(0, len(near_limit)))
            row = near_limit[idx]
            # Scale AT (ambient temp 1–38 °C) to synthetic near-limit temp (100–115 °C)
            at = float(row["AT"])
            at_norm = (at - 1.0) / (38.0 - 1.0)  # normalise to [0, 1]
            self._temp = 100.0 + at_norm * 15.0   # map to [100, 115] °C (near 120 limit)
            self._power = float(row["PE"])          # real power value (near-limit)
            return self.true_state()
        # Start near upper temperature bound so over-limit setpoints quickly cause violations
        self._temp = 110.0
        self._power = 450.0
        return self.true_state()

    @property
    def using_real_data(self) -> bool:
        return bool(self._real_rows)

    def true_state(self) -> Mapping[str, Any]:
        # Always return synthetic dynamics state
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
            if self._rng is None:
                raise RuntimeError("IndustrialTrackAdapter.reset() must be called before observe()")
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
        # Always use synthetic dynamics — real data only affects initialisation in reset()
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
