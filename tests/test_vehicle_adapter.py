"""Unit tests for VehicleDomainAdapter (ORIUS vehicles prototype)."""
from __future__ import annotations

import math
import pytest

from orius.adapters.vehicle import VehicleDomainAdapter
from orius.vehicles.plant import VehiclePlant


class TestVehiclePlant:
    def test_reset_and_state(self) -> None:
        plant = VehiclePlant(dt_s=0.25, speed_limit_mps=30.0)
        s = plant.reset(position_m=10.0, speed_mps=5.0)
        assert s["position_m"] == 10.0
        assert s["speed_mps"] == 5.0

    def test_step_updates_state(self) -> None:
        plant = VehiclePlant(dt_s=0.25, speed_limit_mps=30.0)
        plant.reset(position_m=0.0, speed_mps=10.0)
        s = plant.step(2.0)
        assert s["speed_mps"] > 10.0
        assert s["position_m"] > 0.0

    def test_speed_limit_violation(self) -> None:
        plant = VehiclePlant(dt_s=1.0, speed_limit_mps=10.0)
        plant.reset(speed_mps=5.0)
        for _ in range(10):
            plant.step(10.0)
        v = plant.check_violation()
        assert v["violated"] is True


class TestVehicleDomainAdapterIngestTelemetry:
    def test_passthrough(self) -> None:
        adapter = VehicleDomainAdapter()
        packet = {"position_m": 1.0, "speed_mps": 5.0, "speed_limit_mps": 30.0, "ts_utc": "2026-01-01T00:00:00Z"}
        out = adapter.ingest_telemetry(packet)
        assert out["position_m"] == 1.0
        assert out["speed_mps"] == 5.0

    def test_nan_hold(self) -> None:
        adapter = VehicleDomainAdapter()
        packet = {"position_m": float("nan"), "speed_mps": 5.0, "_hold_position_m": 10.0}
        out = adapter.ingest_telemetry(packet)
        assert out["position_m"] == 10.0


class TestVehicleDomainAdapterComputeOqe:
    def test_returns_w_in_range(self) -> None:
        adapter = VehicleDomainAdapter()
        state = {"speed_mps": 5.0, "position_m": 0.0, "ts_utc": "2026-01-01T00:00:00Z"}
        w, flags = adapter.compute_oqe(state, None)
        assert 0.0 <= w <= 1.0
        assert isinstance(flags, dict)


class TestVehicleDomainAdapterBuildUncertaintySet:
    def test_returns_intervals(self) -> None:
        adapter = VehicleDomainAdapter()
        state = {"position_m": 10.0, "speed_mps": 5.0, "lead_position_m": 40.0}
        unc, meta = adapter.build_uncertainty_set(state, 0.9, 0.9, cfg={})
        assert "position_lower_m" in unc
        assert "position_upper_m" in unc
        assert "lead_position_lower_m" in unc
        assert "lead_position_upper_m" in unc
        assert unc["position_lower_m"] < unc["position_upper_m"]
        assert "inflation" in meta


class TestVehicleDomainAdapterRepairAction:
    def test_clips_acceleration(self) -> None:
        adapter = VehicleDomainAdapter()
        candidate = {"acceleration_mps2": 100.0}
        tightened = {"uncertainty": {"speed_lower_mps": 0, "speed_upper_mps": 30}, "constraints": {"accel_max_mps2": 3.0, "accel_min_mps2": -5.0, "speed_limit_mps": 30.0, "dt_s": 0.25}}
        state = {"speed_limit_mps": 30.0}
        safe, meta = adapter.repair_action(
            candidate, tightened, state=state, uncertainty={}, constraints=tightened["constraints"], cfg={}
        )
        assert safe["acceleration_mps2"] <= 3.0
        assert meta["repaired"] is True

    def test_passes_through_safe_action(self) -> None:
        adapter = VehicleDomainAdapter()
        candidate = {"acceleration_mps2": 1.0}
        tightened = {"uncertainty": {"speed_lower_mps": 0, "speed_upper_mps": 10}, "constraints": {"accel_max_mps2": 3.0, "accel_min_mps2": -5.0, "speed_limit_mps": 30.0, "dt_s": 0.25}}
        state = {"speed_limit_mps": 30.0}
        safe, meta = adapter.repair_action(
            candidate, tightened, state=state, uncertainty={}, constraints=tightened["constraints"], cfg={}
        )
        assert abs(safe["acceleration_mps2"] - 1.0) < 1e-6
        assert meta["repaired"] is False

    def test_ttc_clamp_brakes_when_gap_is_tight(self) -> None:
        adapter = VehicleDomainAdapter()
        candidate = {"acceleration_mps2": 2.0}
        tightened = {
            "uncertainty": {
                "position_upper_m": 40.0,
                "speed_lower_mps": 12.0,
                "speed_upper_mps": 12.0,
                "lead_position_lower_m": 65.0,
            },
            "constraints": {
                "accel_max_mps2": 3.0,
                "accel_min_mps2": -5.0,
                "speed_limit_mps": 30.0,
                "dt_s": 0.25,
                "min_headway_m": 5.0,
                "ttc_min_s": 2.0,
            },
        }
        state = {
            "position_m": 40.0,
            "speed_mps": 12.0,
            "speed_limit_mps": 30.0,
            "lead_position_m": 65.0,
        }
        safe, meta = adapter.repair_action(
            candidate,
            tightened,
            state=state,
            uncertainty=tightened["uncertainty"],
            constraints=tightened["constraints"],
            cfg={},
        )
        assert safe["acceleration_mps2"] < 0.0
        assert meta["repaired"] is True
        assert meta["intervention_reason"] == "ttc_clamp"
        assert meta["repair_surface"] == "ttc_predictive_barrier"

    def test_predictive_entry_barrier_triggers_when_one_step_safe_region_is_gone(self) -> None:
        adapter = VehicleDomainAdapter()
        candidate = {"acceleration_mps2": 2.0}
        tightened = {
            "uncertainty": {
                "position_upper_m": 44.0,
                "speed_lower_mps": 10.0,
                "speed_upper_mps": 10.0,
                "lead_position_lower_m": 50.0,
            },
            "constraints": {
                "accel_max_mps2": 3.0,
                "accel_min_mps2": -5.0,
                "speed_limit_mps": 30.0,
                "dt_s": 0.25,
                "min_headway_m": 5.0,
                "ttc_min_s": 2.0,
            },
        }
        state = {
            "position_m": 44.0,
            "speed_mps": 10.0,
            "speed_limit_mps": 30.0,
            "lead_position_m": 50.0,
        }
        safe, meta = adapter.repair_action(
            candidate,
            tightened,
            state=state,
            uncertainty=tightened["uncertainty"],
            constraints=tightened["constraints"],
            cfg={},
        )
        assert safe["acceleration_mps2"] == -5.0
        assert meta["repaired"] is True
        assert meta["intervention_reason"] == "headway_predictive_entry_barrier"
        assert meta["entry_barrier_triggered"] is True


class TestVehicleDomainAdapterEmitCertificate:
    def test_returns_certificate_dict(self) -> None:
        adapter = VehicleDomainAdapter()
        cert = adapter.emit_certificate(
            command_id="cmd-1",
            device_id="veh-0",
            zone_id="zone-0",
            controller="dc3s",
            proposed_action={"acceleration_mps2": 2.0},
            safe_action={"acceleration_mps2": 1.5},
            uncertainty={"meta": {"inflation": 1.0}},
            reliability={"w_t": 0.9},
            drift={"drift": False},
            cfg={},
        )
        assert "certificate_hash" in cert
        assert cert["proposed_action"]["acceleration_mps2"] == 2.0
        assert cert["safe_action"]["acceleration_mps2"] == 1.5
        assert cert["runtime_surface"] == "waymo_motion_replay_surrogate"
        assert cert["closure_tier"] == "defended_bounded_row"
        assert isinstance(cert["reliability_feature_basis"], dict)
