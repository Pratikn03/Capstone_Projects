"""ORIUS-Bench industrial track — process control domain.

Temperature, pressure, power. Safety: temp in bounds, power below max.
Fault injection: bias, noise, stuck_sensor on primary state (temp_c).

Real-data mode
--------------
Pass ``dataset_path`` to load the processed industrial ORIUS row.
``reset()`` selects a high-load operating point from that row and ``step()``
uses an action-conditioned surrogate anchored to the next real row rather than
an unrelated synthetic plant.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
        self._pressure = 1010.0
        self._rng: np.random.Generator | None = None
        self._episodes: list[list[dict[str, float]]] = []
        self._episode: list[dict[str, float]] = []
        self._episode_idx = 0
        if dataset_path is not None:
            from orius.orius_bench.real_data_loader import load_industrial_runtime_rows

            rows = load_industrial_runtime_rows(Path(dataset_path))
            grouped: dict[str, list[dict[str, float]]] = {}
            for row in rows:
                grouped.setdefault(str(row.get("sensor_id", "sensor-0")), []).append(dict(row))
            for episode in grouped.values():
                episode.sort(key=lambda row: int(row.get("step", 0)))
                self._episodes.append(episode)

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        if self._episodes:
            episode = self._episodes[int(self._rng.integers(0, len(self._episodes)))]
            near_limit = [idx for idx, row in enumerate(episode) if float(row.get("power_mw", 0.0)) >= 480.0]
            if not near_limit:
                peak_idx = int(np.argmax([float(row.get("power_mw", 0.0)) for row in episode]))
                near_limit = [peak_idx]
            self._episode = episode
            self._episode_idx = int(near_limit[int(self._rng.integers(0, len(near_limit)))])
            row = self._episode[self._episode_idx]
            self._temp = float(row["temp_c"])
            self._power = float(row["power_mw"])
            self._pressure = float(row["pressure_mbar"])
            return self.true_state()
        # Start near upper temperature bound so over-limit setpoints quickly cause violations
        self._temp = 110.0
        self._power = 450.0
        self._pressure = 1010.0
        return self.true_state()

    @property
    def using_real_data(self) -> bool:
        return bool(self._episodes)

    def true_state(self) -> Mapping[str, Any]:
        return {
            "temp_c": float(self._temp),
            "power_mw": float(self._power),
            "pressure_mbar": float(self._pressure),
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
        if self._episodes:
            setpoint = float(action.get("power_setpoint_mw", self._power))
            real_current = self._episode[self._episode_idx]
            real_next = self._episode[min(self._episode_idx + 1, len(self._episode) - 1)]
            delta_power = setpoint - float(real_current["power_mw"])
            self._power = float(setpoint)
            self._temp = float(
                self._temp + 0.18 * (float(real_next["temp_c"]) - self._temp) + 0.07 * delta_power
            )
            self._pressure = float(real_next["pressure_mbar"]) - 0.015 * delta_power
            self._episode_idx = min(self._episode_idx + 1, len(self._episode) - 1)
            return dict(self.true_state())
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
