"""Tests for ORIUS Universal Framework."""
from __future__ import annotations

import pytest

from orius.universal_framework import run_universal_step, get_adapter, list_domains


def test_list_domains() -> None:
    domains = list_domains()
    assert "energy" in domains
    assert "av" in domains
    assert "industrial" in domains
    assert "healthcare" in domains
    assert "surgical_robotics" in domains
    assert "aerospace" in domains


def test_get_adapter_industrial() -> None:
    adapter = get_adapter("industrial", {})
    assert adapter is not None


def test_run_universal_step_industrial() -> None:
    adapter = get_adapter("industrial", {})
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "temp_c": 25.0,
            "pressure_mbar": 1010.0,
            "power_mw": 450.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        history=None,
        candidate_action={"power_setpoint_mw": 480.0},
        constraints={"power_max_mw": 500.0},
        quantile=30.0,
    )
    assert "certificate" in result
    assert "safe_action" in result
    assert "reliability_w" in result
    assert 0.0 <= result["reliability_w"] <= 1.0
    safe = result["safe_action"]
    assert "power_setpoint_mw" in safe
    assert 0.0 <= safe["power_setpoint_mw"] <= 500.0


def test_run_universal_step_healthcare() -> None:
    adapter = get_adapter("healthcare", {})
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "hr_bpm": 72.0,
            "spo2_pct": 97.0,
            "respiratory_rate": 14.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        history=None,
        candidate_action={"alert_level": 0.3},
        constraints={"spo2_min_pct": 90.0},
        quantile=5.0,
    )
    assert "certificate" in result
    assert "safe_action" in result
    assert "reliability_w" in result


def test_run_universal_step_aerospace() -> None:
    adapter = get_adapter("aerospace", {})
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "altitude_m": 3000.0,
            "airspeed_kt": 180.0,
            "bank_angle_deg": 5.0,
            "fuel_remaining_pct": 65.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        history=None,
        candidate_action={"throttle": 0.7, "bank_deg": 3.0},
        constraints={"v_min_kt": 60.0, "v_max_kt": 350.0},
        quantile=5.0,
    )
    assert "certificate" in result
    assert "safe_action" in result
    assert "reliability_w" in result
    safe = result["safe_action"]
    assert "throttle" in safe
    assert "bank_deg" in safe


def test_run_universal_step_vehicle_repairs_tight_headway() -> None:
    adapter = get_adapter("av", {})
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "position_m": 44.0,
            "speed_mps": 10.0,
            "speed_limit_mps": 30.0,
            "lead_position_m": 50.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        history=None,
        candidate_action={"acceleration_mps2": 2.0},
        constraints={
            "speed_limit_mps": 30.0,
            "accel_min_mps2": -5.0,
            "accel_max_mps2": 3.0,
            "dt_s": 0.25,
            "min_headway_m": 5.0,
            "headway_time_s": 2.0,
        },
        quantile=0.9,
    )
    assert result["repair_meta"]["repaired"] is True
    assert result["repair_meta"]["intervention_reason"] == "headway_clamp"
    assert result["safe_action"]["acceleration_mps2"] < 0.0


def test_get_adapter_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown domain"):
        get_adapter("unknown_domain", {})
