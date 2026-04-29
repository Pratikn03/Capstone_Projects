"""API tests for the ORIUS universal framework endpoints."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from services.api.config import get_api_keys
from services.api.main import app
from services.api.security import API_KEY_NAME


UNIVERSAL_HEADERS = {API_KEY_NAME: "universal-test-key"}


@pytest.fixture(autouse=True)
def _universal_auth(monkeypatch):
    monkeypatch.setenv("ORIUS_API_KEYS", json.dumps({"universal-test-key": ["read", "write"]}))
    monkeypatch.delenv("ORIUS_AUTH_DISABLED_FOR_TESTS", raising=False)
    get_api_keys.cache_clear()
    yield
    get_api_keys.cache_clear()


def test_universal_domains_endpoint_exposes_framework_metadata() -> None:
    client = TestClient(app)

    response = client.get("/universal/domains", headers=UNIVERSAL_HEADERS)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["domains"] == sorted(payload["domains"])
    assert "energy" in payload["domains"]
    assert "av" in payload["domains"]
    assert payload["pipeline_stages"] == [
        "Detect",
        "Calibrate",
        "Constrain",
        "Shield",
        "Certify",
    ]
    assert "domain_state" in payload["tables"]
    assert "fault_taxonomy" in payload["tables"]


def test_universal_step_vehicle_round_trip() -> None:
    client = TestClient(app)
    request = {
        "domain_id": "AV",
        "raw_telemetry": {
            "position_m": 40.0,
            "speed_mps": 12.0,
            "speed_limit_mps": 30.0,
            "lead_position_m": 75.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        "candidate_action": {"acceleration_mps2": 4.0},
        "constraints": {"speed_limit_mps": 30.0, "min_headway_m": 5.0, "ttc_min_s": 2.0},
        "quantile": 2.0,
        "residual": 2.5,
    }

    response = client.post("/universal/step", json=request, headers=UNIVERSAL_HEADERS)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["domain_id"] == "av"
    assert payload["safe_action"]["acceleration_mps2"] <= request["candidate_action"]["acceleration_mps2"]
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
    assert payload["state"]["position_m"] == 40.0
    assert set(payload["theorem_contracts"]) == {"T3a", "T11"}
    assert payload["theorem_contracts"]["T3a"]["status"] == "runtime_linked"
    assert payload["theorem_contracts"]["T11"]["forward_only"] is True


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
        headers=UNIVERSAL_HEADERS,
    )

    assert response.status_code == 404
    assert "Unknown domain" in response.text
