"""ORIUS-Bench vehicle track — 1D longitudinal AV domain.

Uses a processed AV replay lane when available and falls back to the older
synthetic plant only in permissive support-tier mode. Safety: speed limit, TTC
entry barrier. Fault injection on position/speed.
"""
from __future__ import annotations

import math
from pathlib import Path
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
        ttc_min_s: float = 2.0,
        dataset_path: str | Path | None = None,
    ):
        self._speed_limit = speed_limit_mps
        self._dt = dt_s
        self._headway = min_headway_m
        self._ttc_min_s = ttc_min_s
        self._plant: VehiclePlant | None = None
        self._rng: np.random.Generator | None = None
        self._episodes: list[list[dict[str, float]]] = []
        self._episode: list[dict[str, float]] = []
        self._episode_idx = 0
        self._state: dict[str, float] = {}
        if dataset_path is not None:
            from orius.orius_bench.real_data_loader import load_vehicle_rows

            rows = load_vehicle_rows(Path(dataset_path))
            grouped: dict[str, list[dict[str, float]]] = {}
            for row in rows:
                grouped.setdefault(str(row.get("vehicle_id", "veh-0")), []).append(dict(row))
            for episode in grouped.values():
                episode.sort(key=lambda row: int(row.get("step", 0)))
                for idx, row in enumerate(episode):
                    if idx + 1 < len(episode):
                        next_row = episode[idx + 1]
                        lead_speed = (
                            float(next_row["lead_position_m"]) - float(row["lead_position_m"])
                        ) / max(self._dt, 1.0e-9)
                    else:
                        lead_speed = 0.0
                    row["lead_speed_mps"] = float(max(0.0, lead_speed))
                self._episodes.append(episode)

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        if self._episodes:
            candidates: list[tuple[int, int]] = []
            for ep_idx, episode in enumerate(self._episodes):
                for row_idx, row in enumerate(episode):
                    gap_budget = float(row["lead_position_m"]) - float(row["position_m"]) - self._headway
                    closing_speed = float(row["speed_mps"]) - float(row.get("lead_speed_mps", 0.0))
                    ttc = gap_budget / max(closing_speed, 1.0e-9) if gap_budget > 0.0 and closing_speed > 0.0 else float("inf")
                    if gap_budget <= 22.5 and closing_speed >= 1.0:
                        candidates.append((ep_idx, row_idx))
            if not candidates:
                scored_candidates: list[tuple[float, int, int]] = []
                for ep_idx, episode in enumerate(self._episodes):
                    for row_idx, row in enumerate(episode):
                        gap_budget = float(row["lead_position_m"]) - float(row["position_m"]) - self._headway
                        closing_speed = max(
                            0.0,
                            float(row["speed_mps"]) - float(row.get("lead_speed_mps", 0.0)),
                        )
                        score = gap_budget - 2.0 * closing_speed
                        scored_candidates.append((score, ep_idx, row_idx))
                scored_candidates.sort(key=lambda item: item[0])
                _, ep_idx, row_idx = scored_candidates[0]
                candidates = [(ep_idx, row_idx)]
            ep_idx, row_idx = candidates[int(self._rng.integers(0, len(candidates)))]
            self._episode = self._episodes[ep_idx]
            self._episode_idx = row_idx
            row = self._episode[self._episode_idx]
            gap = float(row["lead_position_m"]) - float(row["position_m"])
            stressed_gap = max(self._headway + 1.0, min(gap, gap - 4.0))
            stressed_lead_speed = max(
                0.0,
                min(float(row.get("lead_speed_mps", 0.0)), float(row["speed_mps"]) - 2.0),
            )
            self._state = {
                "position_m": float(row["position_m"]),
                "speed_mps": float(row["speed_mps"]),
                "speed_limit_mps": float(row["speed_limit_mps"]),
                "lead_position_m": float(row["position_m"]) + float(stressed_gap),
                "lead_speed_mps": float(stressed_lead_speed),
            }
            return self.true_state()
        self._plant = VehiclePlant(
            dt_s=self._dt,
            speed_limit_mps=self._speed_limit,
            min_headway_m=self._headway,
            ttc_min_s=self._ttc_min_s,
        )
        self._plant.reset(position_m=0.0, speed_mps=5.0, lead_position_m=50.0)
        return self.true_state()

    def true_state(self) -> Mapping[str, Any]:
        if self._episodes:
            return dict(self._state)
        if self._plant is None:
            raise RuntimeError("VehicleTrackAdapter.reset() must be called before true_state()")
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
            if self._rng is None:
                raise RuntimeError("VehicleTrackAdapter.reset() must be called before observe()")
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
        if self._episodes:
            current = dict(self._state)
            real_current = self._episode[self._episode_idx]
            real_next = self._episode[min(self._episode_idx + 1, len(self._episode) - 1)]
            a = float(action.get("acceleration_mps2", 0.0))
            lead_speed = float(current.get("lead_speed_mps", real_current.get("lead_speed_mps", 0.0)))
            speed_limit = float(real_current.get("speed_limit_mps", self._speed_limit))
            next_speed = max(
                0.0,
                float(current["speed_mps"])
                + a * self._dt
                + 0.10 * (float(real_next["speed_mps"]) - float(current["speed_mps"])),
            )
            next_pos = float(current["position_m"]) + next_speed * self._dt
            replay_lead_speed = float(real_current.get("lead_speed_mps", lead_speed))
            next_lead_speed = max(0.0, lead_speed + 0.20 * (replay_lead_speed - lead_speed))
            replay_lead = float(current["lead_position_m"]) + 0.10 * (
                float(real_next["lead_position_m"]) - float(current["lead_position_m"])
            )
            lead_progress = float(current["lead_position_m"]) + next_lead_speed * self._dt
            next_lead = max(min(replay_lead, lead_progress + 2.0), lead_progress, next_pos + 0.1)
            self._episode_idx = min(self._episode_idx + 1, len(self._episode) - 1)
            self._state = {
                "position_m": float(next_pos),
                "speed_mps": float(next_speed),
                "speed_limit_mps": float(speed_limit),
                "lead_position_m": float(next_lead),
                "lead_speed_mps": float(next_lead_speed),
            }
            return dict(self.true_state())
        if self._plant is None:
            raise RuntimeError("VehicleTrackAdapter.reset() must be called before step()")
        a = float(action.get("acceleration_mps2", 0.0))
        self._plant.step(a)
        return dict(self.true_state())

    @property
    def using_real_data(self) -> bool:
        return bool(self._episodes)

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
            gap = lead_x - x
            gap_budget = gap - self._headway
            if gap_budget <= 0.0:
                violated = True
                severity = max(severity, abs(gap_budget))
            else:
                ttc = gap_budget / max(v, 1e-9)
                if ttc < self._ttc_min_s - 1e-9:
                    violated = True
                    severity = max(severity, self._ttc_min_s - ttc)
        return {"violated": violated, "severity": severity}

    @property
    def domain_name(self) -> str:
        return "vehicle"
