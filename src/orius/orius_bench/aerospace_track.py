"""ORIUS-Bench aerospace track — flight envelope domain.

Airspeed, altitude, bank. Safety: airspeed in [v_min, v_max].
Fault injection: bias, noise, stuck_sensor on airspeed_kt.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from orius.orius_bench.adapter import BenchmarkAdapter


class AerospaceTrackAdapter(BenchmarkAdapter):
    """Aerospace flight envelope benchmark track."""

    def __init__(
        self,
        v_min_kt: float = 60.0,
        v_max_kt: float = 350.0,
        max_bank_deg: float = 30.0,
        dt: float = 0.25,
        dataset_path: str | Path | None = None,
    ):
        self._v_min = v_min_kt
        self._v_max = v_max_kt
        self._max_bank = max_bank_deg
        self._dt = dt
        self._airspeed = 55.0
        self._altitude = 3000.0
        self._bank = 5.0
        self._fuel = 80.0
        self._rng: np.random.Generator | None = None
        self._episodes: list[list[dict[str, float]]] = []
        self._episode: list[dict[str, float]] = []
        self._episode_idx = 0
        if dataset_path is not None:
            from orius.orius_bench.real_data_loader import load_aerospace_runtime_rows

            rows = load_aerospace_runtime_rows(Path(dataset_path))
            grouped: dict[str, list[dict[str, float]]] = {}
            for row in rows:
                grouped.setdefault(str(row.get("flight_id", "flight-0")), []).append(dict(row))
            for episode in grouped.values():
                episode.sort(key=lambda row: int(row.get("step", 0)))
                self._episodes.append(episode)

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        if self._episodes:
            episode = self._episodes[int(self._rng.integers(0, len(self._episodes)))]
            near_limit = [
                idx for idx, row in enumerate(episode)
                if abs(float(row.get("bank_angle_deg", 0.0))) >= self._max_bank * 0.75
                or float(row.get("airspeed_kt", self._v_min)) <= self._v_min + 15.0
                or float(row.get("airspeed_kt", self._v_max)) >= self._v_max - 15.0
            ]
            if not near_limit:
                near_limit = [0]
            self._episode = episode
            self._episode_idx = int(near_limit[int(self._rng.integers(0, len(near_limit)))])
            row = self._episode[self._episode_idx]
            self._airspeed = float(row["airspeed_kt"])
            self._altitude = float(row["altitude_m"])
            self._bank = float(row["bank_angle_deg"])
            self._fuel = float(row["fuel_remaining_pct"])
            return self.true_state()
        # Safe initial airspeed; bank angle is the primary safety violation axis.
        # Nominal controller proposes extreme bank_deg=90 → always violated.
        # DC3S repair clamps to ±max_bank_deg → zero violations.
        # Bank angle is NOT faulted by the fault engine, giving deterministic results.
        self._airspeed = 180.0
        self._altitude = 3000.0
        self._bank = 5.0
        self._fuel = 80.0
        return self.true_state()

    @property
    def using_real_data(self) -> bool:
        return bool(self._episodes)

    def true_state(self) -> Mapping[str, Any]:
        return {
            "airspeed_kt": float(self._airspeed),
            "altitude_m": float(self._altitude),
            "bank_angle_deg": float(self._bank),
            "fuel_remaining_pct": float(self._fuel),
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
            if self._rng is None:
                raise RuntimeError("AerospaceTrackAdapter.reset() must be called before observe()")
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
        if self._episodes:
            throttle = float(action.get("throttle", 0.7))
            bank = float(action.get("bank_deg", 3.0))
            throttle = max(0.0, min(1.0, throttle))
            real_next = self._episode[min(self._episode_idx + 1, len(self._episode) - 1)]
            self._airspeed = float(
                self._airspeed
                + 0.20 * (float(real_next["airspeed_kt"]) - self._airspeed)
                + 12.0 * (throttle - 0.5)
            )
            self._airspeed = max(0.0, min(600.0, self._airspeed))
            self._bank = float(bank)
            self._altitude = float(
                self._altitude
                + 0.15 * (float(real_next["altitude_m"]) - self._altitude)
                + 40.0 * (throttle - 0.4)
            )
            self._altitude = max(0.0, self._altitude)
            self._fuel = max(
                0.0,
                min(
                    100.0,
                    self._fuel
                    - 0.5 * throttle
                    + 0.10 * (float(real_next["fuel_remaining_pct"]) - self._fuel),
                ),
            )
            self._episode_idx = min(self._episode_idx + 1, len(self._episode) - 1)
            return dict(self.true_state())
        throttle = float(action.get("throttle", 0.7))
        bank = float(action.get("bank_deg", 3.0))
        throttle = max(0.0, min(1.0, throttle))
        # No bank clamping — let violations manifest; DC3S repair is the safety layer
        self._airspeed = self._airspeed + (throttle - 0.5) * 20.0 * self._dt
        self._airspeed = max(0.0, min(400.0, self._airspeed))
        self._bank = bank
        self._altitude = self._altitude + 50.0 * (throttle - 0.4) * self._dt
        self._altitude = max(0.0, self._altitude)
        self._fuel = max(0.0, self._fuel - throttle * 0.1 * self._dt)
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
        bank = abs(state.get("bank_angle_deg", 0.0))
        speed_violated = v < self._v_min or v > self._v_max
        bank_violated = bank > self._max_bank
        violated = speed_violated or bank_violated
        severity = 0.0
        if v < self._v_min:
            severity = self._v_min - v
        elif v > self._v_max:
            severity = v - self._v_max
        elif bank_violated:
            severity = bank - self._max_bank
        return {"violated": violated, "severity": severity}

    @property
    def domain_name(self) -> str:
        return "aerospace"
