"""Tests for ORIUS Universal Framework."""

from __future__ import annotations

import pytest

from orius.universal_framework import get_adapter, list_domains, run_universal_step


def test_list_domains() -> None:
    domains = list_domains()
    assert "energy" in domains
    assert "av" in domains
    assert "healthcare" in domains
    assert "navigation" not in domains
    assert "industrial" not in domains
    assert "aerospace" not in domains
    assert domains == ["av", "energy", "healthcare"]


def test_get_adapter_energy() -> None:
    adapter = get_adapter("energy", {})
    assert adapter is not None


def test_run_universal_step_energy() -> None:
    adapter = get_adapter("energy", {})
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "current_soc_mwh": 500.0,
            "yhat_load": 100.0,
            "charge_mw": 0.0,
            "discharge_mw": 0.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        history=None,
        candidate_action={"charge_mw": 0.0, "discharge_mw": 25.0},
        constraints={"min_soc_mwh": 0.0, "max_soc_mwh": 1000.0, "max_power_mw": 250.0},
        quantile=5.0,
    )
    assert "certificate" in result
    assert "safe_action" in result
    assert "reliability_w" in result
    assert 0.0 <= result["reliability_w"] <= 1.0
    safe = result["safe_action"]
    assert "charge_mw" in safe
    assert "discharge_mw" in safe


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


def test_run_universal_step_vehicle_repairs_tight_ttc_case() -> None:
    adapter = get_adapter("av", {})
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "position_m": 40.0,
            "speed_mps": 12.0,
            "speed_limit_mps": 30.0,
            "lead_position_m": 75.0,
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
            "ttc_min_s": 2.0,
            "lead_speed_mps": 0.0,
        },
        quantile=1.1,
    )
    assert result["repair_meta"]["repaired"] is True
    assert result["repair_meta"]["intervention_reason"] == "ttc_clamp"
    assert result["safe_action"]["acceleration_mps2"] < 0.0


def test_get_adapter_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown domain"):
        get_adapter("unknown_domain", {})
