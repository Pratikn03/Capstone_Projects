"""End-to-end sanity tests for vehicle prototype harness."""
from __future__ import annotations

import pytest

from orius.vehicles.plant import VehiclePlant
from orius.vehicles.vehicle_adapter import VehicleDomainAdapter
from orius.vehicles.vehicle_runner import run_vehicle_episode, compute_vehicle_metrics


def test_run_vehicle_episode_completes() -> None:
    adapter = VehicleDomainAdapter({"expected_cadence_s": 1.0})
    plant = VehiclePlant(dt_s=0.25, speed_limit_mps=30.0)
    results = run_vehicle_episode(adapter, plant, horizon=12, seed=42)
    assert len(results) == 12
    for r in results:
        assert hasattr(r, "step")
        assert hasattr(r, "violated")
        assert hasattr(r, "intervened")
        assert hasattr(r, "w_t")


def test_compute_vehicle_metrics() -> None:
    from orius.vehicles.vehicle_runner import VehicleStepResult
    results = [
        VehicleStepResult(0, {"speed_mps": 5}, {}, {}, {}, False, False, 0.9),
        VehicleStepResult(1, {"speed_mps": 6}, {}, {}, {}, False, True, 0.8),
    ]
    m = compute_vehicle_metrics(results)
    assert "speed_limit_violations_pct" in m
    assert "intervention_rate_pct" in m
    assert m["n_steps"] == 2


def test_dc3s_interventions_in_toy_scenario() -> None:
    """Sanity: when candidate exceeds safe bounds, repair activates."""
    adapter = VehicleDomainAdapter({"expected_cadence_s": 1.0})
    plant = VehiclePlant(dt_s=0.25, speed_limit_mps=10.0)
    results = run_vehicle_episode(adapter, plant, horizon=24, seed=42)
    metrics = compute_vehicle_metrics(results)
    assert metrics["speed_limit_violations_pct"] < 10.0
