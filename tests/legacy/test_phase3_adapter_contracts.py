from __future__ import annotations

import pytest

from orius.adapters.aerospace import AerospaceDomainAdapter
from orius.adapters.healthcare import HealthcareDomainAdapter
from orius.adapters.industrial import IndustrialDomainAdapter
from orius.adapters.navigation import NavigationDomainAdapter
from orius.adapters.vehicle import VehicleDomainAdapter
from orius.dc3s.battery_adapter import BatteryDomainAdapter


@pytest.mark.parametrize(
    ("adapter", "state", "constraints", "quantile", "expected_bounds", "expected_fallback", "expected_closure_tier"),
    [
        (
            VehicleDomainAdapter(),
            {
                "position_m": 40.0,
                "speed_mps": 12.0,
                "lead_position_m": 65.0,
                "speed_limit_mps": 30.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            {
                "accel_min_mps2": -5.0,
                "accel_max_mps2": 3.0,
                "speed_limit_mps": 30.0,
                "dt_s": 0.25,
                "min_headway_m": 5.0,
                "ttc_min_s": 2.0,
            },
            0.9,
            ("acceleration_mps2_lower", "acceleration_mps2_upper"),
            {"acceleration_mps2": -5.0},
            "defended_bounded_row",
        ),
        (
            IndustrialDomainAdapter(),
            {
                "temp_c": 25.0,
                "pressure_mbar": 1010.0,
                "power_mw": 450.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            {"power_max_mw": 500.0},
            30.0,
            ("power_setpoint_lower_mw", "power_setpoint_upper_mw"),
            {"power_setpoint_mw": 440.0},
            "defended_bounded_row",
        ),
        (
            HealthcareDomainAdapter(),
            {
                "hr_bpm": 72.0,
                "spo2_pct": 89.0,
                "respiratory_rate": 14.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            {"spo2_min_pct": 90.0},
            5.0,
            ("alert_level_lower", "alert_level_upper"),
            {"alert_level": 1.0},
            "defended_bounded_row",
        ),
        (
            NavigationDomainAdapter(),
            {
                "x": 9.8,
                "y": 5.0,
                "vx": 0.0,
                "vy": 0.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            {"arena_size": 10.0, "speed_limit": 1.0, "dt_s": 0.25},
            1.0,
            ("ax_lower", "ax_upper", "ay_lower", "ay_upper"),
            {"ax": 0.0, "ay": 0.0},
            "shadow_synthetic",
        ),
        (
            AerospaceDomainAdapter(),
            {
                "altitude_m": 3000.0,
                "airspeed_kt": 180.0,
                "bank_angle_deg": 5.0,
                "fuel_remaining_pct": 8.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            {"v_min_kt": 60.0, "max_bank_deg": 30.0},
            5.0,
            ("throttle_lower", "throttle_upper", "bank_deg_lower", "bank_deg_upper"),
            {"throttle": 0.5, "bank_deg": 0.0},
            "experimental",
        ),
    ],
)
def test_universal_adapters_expose_explicit_tightened_bounds_and_fallbacks(
    adapter,
    state,
    constraints,
    quantile,
    expected_bounds,
    expected_fallback,
    expected_closure_tier,
) -> None:
    reliability_w, flags = adapter.compute_oqe(state, history=[dict(state)])
    uncertainty, _ = adapter.build_uncertainty_set(state, reliability_w, quantile, cfg={})
    tightened = adapter.tighten_action_set(uncertainty, constraints, cfg={})

    for bound_key in expected_bounds:
        assert bound_key in tightened
    assert tightened["fallback_action"] == expected_fallback
    assert tightened["projection_surface"]
    assert isinstance(tightened["viable"], bool)
    assert flags["closure_tier"] == expected_closure_tier


def test_battery_tightened_set_exposes_safe_hold_fallback() -> None:
    adapter = BatteryDomainAdapter()
    tightened = adapter.tighten_action_set(
        uncertainty={"meta": {"inflation": 1.0, "w_t": 0.8}},
        constraints={
            "max_power_mw": 50.0,
            "max_charge_mw": 40.0,
            "max_discharge_mw": 35.0,
        },
        cfg={},
    )

    assert tightened["charge_mw_lower"] == 0.0
    assert tightened["charge_mw_upper"] == 40.0
    assert tightened["discharge_mw_lower"] == 0.0
    assert tightened["discharge_mw_upper"] == 35.0
    assert tightened["fallback_action"] == {"charge_mw": 0.0, "discharge_mw": 0.0}


def test_battery_project_to_safe_set_accepts_mapping_state() -> None:
    adapter = BatteryDomainAdapter()
    safe, meta = adapter.project_to_safe_set(
        candidate_action={"charge_mw": 0.0, "discharge_mw": 20.0},
        uncertainty_set={"ftit_soc_min_mwh": 14.0},
        state={
            "current_soc_mwh": 15.0,
            "capacity_mwh": 100.0,
            "min_soc_mwh": 10.0,
            "max_soc_mwh": 90.0,
            "max_power_mw": 50.0,
        },
    )

    assert safe["discharge_mw"] == 1.0
    assert meta["repaired"] is True
