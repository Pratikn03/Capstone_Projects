"""ORIUS-Bench battery track adapter.

Wraps the existing ``BatteryPlant`` simulation as a benchmark track,
providing domain-specific state, observation, and violation semantics.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from orius.cpsbench_iot.plant import BatteryPlant
from orius.orius_bench.adapter import BenchmarkAdapter


class BatteryTrackAdapter(BenchmarkAdapter):
    """Battery benchmark track backed by ``BatteryPlant``."""

    def __init__(
        self,
        capacity_mwh: float = 200.0,
        soc_min_frac: float = 0.1,
        soc_max_frac: float = 0.9,
        charge_eff: float = 0.92,
        discharge_eff: float = 0.95,
        dt_hours: float = 0.25,
        power_max_mw: float = 100.0,
    ):
        self._capacity = capacity_mwh
        self._soc_min_frac = soc_min_frac
        self._soc_max_frac = soc_max_frac
        self._charge_eff = charge_eff
        self._discharge_eff = discharge_eff
        self._dt = dt_hours
        self._power_max = power_max_mw
        self._plant: BatteryPlant | None = None
        self._rng: np.random.Generator | None = None

    # -- BenchmarkAdapter ------------------------------------------------

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        init_soc_mwh = 0.5 * self._capacity
        self._plant = BatteryPlant(
            soc_mwh=init_soc_mwh,
            min_soc_mwh=self._soc_min_frac * self._capacity,
            max_soc_mwh=self._soc_max_frac * self._capacity,
            charge_eff=self._charge_eff,
            discharge_eff=self._discharge_eff,
            dt_hours=self._dt,
        )
        return self.true_state()

    def true_state(self) -> Mapping[str, Any]:
        if self._plant is None:
            raise RuntimeError("BatteryTrackAdapter.reset() must be called before true_state()")
        return {
            "soc": self._plant.soc_mwh / self._capacity,
            "soc_mwh": self._plant.soc_mwh,
            "capacity_mwh": self._capacity,
            "power_max_mw": self._power_max,
        }

    def observe(
        self,
        true_state: Mapping[str, Any],
        fault: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        obs = dict(true_state)
        if fault is None:
            return obs
        # Apply fault effects
        kind = fault.get("kind", "")
        if kind == "blackout":
            return {k: float("nan") for k in obs}
        if kind == "bias" and "soc" in obs:
            obs["soc"] = obs["soc"] + fault.get("magnitude", 0)
        elif kind == "noise" and "soc" in obs:
            sigma = fault.get("sigma", 0.02)
            if self._rng is None:
                raise RuntimeError("BatteryTrackAdapter.reset() must be called before observe()")
            obs["soc"] = obs["soc"] + float(self._rng.normal(0, sigma))
        elif kind == "stuck_sensor" and "soc" in obs:
            obs["soc"] = fault.get("frozen_value", 0.5)
        return obs

    def safe_action_set(
        self,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        soc = state.get("soc", 0.5)
        return {
            "max_charge_mw": self._power_max if soc < self._soc_max_frac else 0.0,
            "max_discharge_mw": self._power_max if soc > self._soc_min_frac else 0.0,
        }

    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._plant is None:
            raise RuntimeError("BatteryTrackAdapter.reset() must be called before step()")
        charge = float(action.get("charge_mw", 0))
        discharge = float(action.get("discharge_mw", 0))
        self._plant.step(charge, discharge)
        return dict(self.true_state())

    def compute_useful_work(self, trajectory: Sequence[Mapping[str, Any]]) -> float:
        """Useful work = total net discharge energy delivered safely."""
        total = 0.0
        for rec in trajectory:
            d = rec.get("discharge_mw", 0)
            if not math.isnan(d):
                total += max(0.0, d)
        return total

    def check_violation(self, state: Mapping[str, Any]) -> dict[str, Any]:
        soc = state.get("soc", 0.5)
        violated = soc < self._soc_min_frac or soc > self._soc_max_frac
        severity = 0.0
        if soc < self._soc_min_frac:
            severity = self._soc_min_frac - soc
        elif soc > self._soc_max_frac:
            severity = soc - self._soc_max_frac
        return {"violated": violated, "severity": severity}

    @property
    def domain_name(self) -> str:
        return "battery"
