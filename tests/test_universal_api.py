"""API tests for the ORIUS universal framework endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient

from services.api.main import app


def test_universal_domains_endpoint_exposes_framework_metadata() -> None:
    client = TestClient(app)

    response = client.get("/universal/domains")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["domains"] == sorted(payload["domains"])
    assert "energy" in payload["domains"]
    assert "av" in payload["domains"]
    assert "navigation" in payload["domains"]
    assert payload["pipeline_stages"] == [
        "Detect",
        "Calibrate",
        "Constrain",
        "Shield",
        "Certify",
    ]
    assert "domain_state" in payload["tables"]
    assert "fault_taxonomy" in payload["tables"]


def test_universal_step_industrial_round_trip() -> None:
    client = TestClient(app)
    request = {
        "domain_id": "INDUSTRIAL",
        "raw_telemetry": {
            "temp_c": 25.0,
            "pressure_mbar": 1010.0,
            "power_mw": 450.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        "candidate_action": {"power_setpoint_mw": 520.0},
        "constraints": {"power_max_mw": 500.0},
        "quantile": 30.0,
        "residual": 2.5,
    }

    response = client.post("/universal/step", json=request)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["domain_id"] == "industrial"
    assert 0.0 <= payload["safe_action"]["power_setpoint_mw"] <= 500.0
    assert payload["safe_action"]["power_setpoint_mw"] < request["candidate_action"]["power_setpoint_mw"]
    assert 0.0 <= payload["reliability_w"] <= 1.0
    assert payload["pipeline_stages"] == [
        "Detect",
        "Calibrate",
        "Constrain",
        "Shield",
        "Certify",
    ]
    assert payload["drift_flag"] is False
    assert "certificate_hash" in payload["certificate"]
    assert payload["repair_meta"]["repaired"] is True
    assert payload["state"]["power_mw"] == 450.0


def test_universal_step_unknown_domain_returns_404() -> None:
    client = TestClient(app)

    response = client.post(
        "/universal/step",
        json={
            "domain_id": "unknown",
            "raw_telemetry": {"ts_utc": "2026-01-01T00:00:00Z"},
            "candidate_action": {},
            "constraints": {},
        },
    )

    assert response.status_code == 404
    assert "Unknown domain" in response.text


def test_universal_step_navigation_round_trip() -> None:
    client = TestClient(app)

    response = client.post(
        "/universal/step",
        json={
            "domain_id": "navigation",
            "raw_telemetry": {
                "x": 4.8,
                "y": 4.8,
                "vx": 0.1,
                "vy": 0.1,
                "ts_utc": "2026-01-01T00:00:00Z",
            },
            "candidate_action": {"ax": 1.0, "ay": 1.0},
            "constraints": {"arena_min": 0.0, "arena_max": 10.0, "speed_limit": 1.0, "dt": 1.0},
            "quantile": 10.0,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["domain_id"] == "navigation"
    assert payload["repair_meta"]["repaired"] is True
    assert "certificate_hash" in payload["certificate"]
