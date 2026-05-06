"""Minimal vehicle benchmark harness — CPSBench-like runner for 1D longitudinal.

Prototype extension. Outputs to reports/vehicles_prototype/ (isolated from battery).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .plant import VehiclePlant
from .vehicle_adapter import VehicleDomainAdapter


@dataclass
class VehicleStepResult:
    step: int
    true_state: dict[str, float]
    observed_state: dict[str, float]
    proposed_action: dict[str, float]
    safe_action: dict[str, float]
    violated: bool
    intervened: bool
    w_t: float


def run_vehicle_episode(
    adapter: VehicleDomainAdapter,
    plant: VehiclePlant,
    horizon: int,
    seed: int = 42,
    fault_inject: bool = False,
    fault_drop_rate: float = 0.0,
) -> list[VehicleStepResult]:
    """Run one vehicle episode through DC3S adapter."""
    rng = np.random.default_rng(seed)
    plant.reset(position_m=0.0, speed_mps=5.0, speed_limit_mps=30.0)
    cfg = adapter._cfg
    history: list[Mapping[str, Any]] = []
    results: list[VehicleStepResult] = []

    for t in range(horizon):
        true_state = plant.state()
        obs = dict(true_state)
        obs["ts_utc"] = f"2026-01-01T{t:02d}:00:00Z"
        obs["load_mw"] = obs["speed_mps"]

        if fault_inject and fault_drop_rate > 0 and rng.random() < fault_drop_rate:
            obs["speed_mps"] = float("nan")
            obs["position_m"] = obs.get("_hold_position_m", obs.get("position_m", 0.0))

        state = adapter.ingest_telemetry(obs)
        w_t, _ = adapter.compute_oqe(state, history[-1:] if history else None)
        quantile = 0.9
        uncertainty, _ = adapter.build_uncertainty_set(state, w_t, quantile, cfg=cfg, drift_flag=False)
        constraints = {
            "speed_limit_mps": true_state.get("speed_limit_mps", 30.0),
            "accel_min_mps2": -5.0,
            "accel_max_mps2": 3.0,
            "dt_s": 0.25,
        }
        tightened = adapter.tighten_action_set(uncertainty, constraints, cfg=cfg)
        candidate = {"acceleration_mps2": 2.0}
        safe_action, repair_meta = adapter.repair_action(
            candidate,
            tightened,
            state=state,
            uncertainty=uncertainty,
            constraints=constraints,
            cfg=cfg,
        )
        plant.step(safe_action["acceleration_mps2"])
        violation = plant.check_violation()
        history.append(state)

        results.append(
            VehicleStepResult(
                step=t,
                true_state=dict(plant.state()),
                observed_state=dict(obs),
                proposed_action=dict(candidate),
                safe_action=dict(safe_action),
                violated=violation["violated"],
                intervened=bool(repair_meta.get("repaired", False)),
                w_t=float(w_t),
            )
        )

    return results


def compute_vehicle_metrics(results: Sequence[VehicleStepResult]) -> dict[str, float]:
    """Minimal safety metrics for vehicle prototype."""
    n = len(results)
    if n == 0:
        return {"speed_limit_violations_pct": 0.0, "intervention_rate_pct": 0.0}
    violations = sum(1 for r in results if r.violated)
    interventions = sum(1 for r in results if r.intervened)
    return {
        "speed_limit_violations_pct": 100.0 * violations / n,
        "intervention_rate_pct": 100.0 * interventions / n,
        "n_steps": float(n),
    }
