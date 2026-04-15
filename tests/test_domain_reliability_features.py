from __future__ import annotations

import pytest

from orius.adapters.aerospace import AerospaceDomainAdapter
from orius.adapters.healthcare import HealthcareDomainAdapter
from orius.adapters.industrial import IndustrialDomainAdapter
from orius.adapters.navigation import NavigationDomainAdapter
from orius.adapters.vehicle import VehicleDomainAdapter


@pytest.mark.parametrize(
    ("adapter", "state", "expected_signals", "expected_sources", "expected_closure_tier", "expected_runtime_surface"),
    [
        (
            VehicleDomainAdapter(),
            {
                "position_m": 10.0,
                "speed_mps": 8.0,
                "speed_limit_mps": 13.4,
                "lead_position_m": 35.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            ["ego_speed_mps", "lead_gap_m", "speed_limit_mps"],
            ["speed_mps", "derived:_lead_gap", "speed_limit_mps"],
            "defended_bounded_row",
            "waymo_motion_replay_surrogate",
        ),
        (
            IndustrialDomainAdapter(),
            {
                "temp_c": 20.0,
                "pressure_mbar": 1010.0,
                "power_mw": 450.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            ["power_output_mw", "temperature_c", "pressure_mbar"],
            ["power_mw", "temp_c", "pressure_mbar"],
            "defended_bounded_row",
            "uci_ccpp_processed_replay_surrogate",
        ),
        (
            HealthcareDomainAdapter(),
            {
                "hr_bpm": 72.0,
                "spo2_pct": 97.0,
                "respiratory_rate": 14.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            ["heart_rate_bpm", "spo2_pct", "respiratory_rate_bpm"],
            ["hr_bpm", "spo2_pct", "respiratory_rate"],
            "defended_bounded_row",
            "bidmc_processed_replay_surrogate",
        ),
        (
            NavigationDomainAdapter(),
            {
                "x": 1.0,
                "y": 2.0,
                "vx": 0.2,
                "vy": 0.1,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            ["x_position_m", "y_position_m", "x_velocity_mps", "y_velocity_mps"],
            ["x", "y", "vx", "vy"],
            "shadow_synthetic",
            "kitti_runtime_shadow_support",
        ),
        (
            AerospaceDomainAdapter(),
            {
                "altitude_m": 3000.0,
                "airspeed_kt": 180.0,
                "bank_angle_deg": 5.0,
                "fuel_remaining_pct": 80.0,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            ["altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct"],
            ["altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct"],
            "experimental",
            "public_adsb_runtime_support_lane",
        ),
    ],
)
def test_non_battery_adapters_record_domain_native_reliability_feature_basis(
    adapter,
    state,
    expected_signals,
    expected_sources,
    expected_closure_tier,
    expected_runtime_surface,
) -> None:
    reliability_w, flags = adapter.compute_oqe(state, history=[dict(state)])

    assert 0.0 <= reliability_w <= 1.0
    assert flags["runtime_surface"] == expected_runtime_surface
    assert flags["closure_tier"] == expected_closure_tier

    basis = flags["reliability_feature_basis"]
    assert basis["domain_native"] is True
    assert basis["signal_names"] == expected_signals
    assert basis["source_fields"] == expected_sources
    assert "load_mw" not in basis["source_fields"]
    assert "renewables_mw" not in basis["source_fields"]
