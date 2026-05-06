"""Tests for the ORIUS /step FastAPI endpoint.

Phase 4 of ORIUS gap-closing plan.
Tests health check, the canonical three supported domains, and error handling.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from orius.api.app import app
from services.api.config import get_api_keys
from services.api.security import API_KEY_NAME

client = TestClient(app, raise_server_exceptions=False)
HEADERS = {API_KEY_NAME: "orius-api-test-key"}


@pytest.fixture(autouse=True)
def _auth(monkeypatch):
    monkeypatch.setenv("ORIUS_API_KEYS", json.dumps({"orius-api-test-key": ["read", "write"]}))
    monkeypatch.delenv("ORIUS_AUTH_DISABLED_FOR_TESTS", raising=False)
    get_api_keys.cache_clear()
    yield
    get_api_keys.cache_clear()


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "orius" in data["service"].lower()


class TestStepEndpointBasic:
    def _step(
        self, domain: str, state: dict, candidate_action: dict | None = None, constraints: dict | None = None
    ) -> dict:
        payload = {
            "domain": domain,
            "state": state,
            "candidate_action": candidate_action or {},
            "constraints": constraints or {},
        }
        resp = client.post("/step", json=payload, headers=HEADERS)
        assert resp.status_code == 200, f"domain={domain}: {resp.text}"
        return resp.json()

    def test_healthcare_domain(self):
        data = self._step(
            domain="healthcare",
            state={"spo2_pct": 87.0, "hr_bpm": 72.0, "respiratory_rate": 14.0},
            candidate_action={"alert_level": 0.2},
            constraints={"spo2_min_pct": 90.0},
        )
        assert "safe_action" in data

    def test_vehicle_domain(self):
        data = self._step(
            domain="vehicle",
            state={
                "position_m": 40.0,
                "speed_mps": 12.0,
                "speed_limit_mps": 30.0,
                "lead_position_m": 75.0,
            },
            candidate_action={"acceleration_mps2": 4.0},
            constraints={"speed_limit_mps": 30.0, "min_headway_m": 5.0, "ttc_min_s": 2.0},
        )
        assert "safe_action" in data

    def test_battery_domain(self):
        data = self._step(
            domain="battery",
            state={
                "current_soc_mwh": 5000.0,
                "capacity_mwh": 10000.0,
                "min_soc_mwh": 1000.0,
                "max_soc_mwh": 9000.0,
                "yhat_load": 1000.0,
            },
            candidate_action={"charge_mw": 100.0, "discharge_mw": 0.0},
            constraints={"max_power_mw": 50.0},
        )
        assert "safe_action" in data
        assert isinstance(data["uncertainty_set"]["lower"], list)
        assert isinstance(data["uncertainty_set"]["upper"], list)

    def test_response_has_certificate(self):
        data = self._step(
            domain="healthcare",
            state={"spo2_pct": 92.0, "hr_bpm": 70.0, "respiratory_rate": 14.0},
        )
        assert "certificate" in data

    def test_response_has_uncertainty_set(self):
        data = self._step(
            domain="healthcare",
            state={"spo2_pct": 92.0, "hr_bpm": 70.0, "respiratory_rate": 14.0},
        )
        assert "uncertainty_set" in data

    def test_repaired_flag_type(self):
        data = self._step(
            domain="healthcare",
            state={"spo2_pct": 92.0, "hr_bpm": 70.0, "respiratory_rate": 14.0},
        )
        assert isinstance(data["repaired"], bool)

    def test_domain_case_insensitive(self):
        """Domain name should be case-insensitive."""
        for name in ("HEALTHCARE", "Healthcare", "healthcare"):
            resp = client.post(
                "/step",
                json={
                    "domain": name,
                    "state": {"spo2_pct": 92.0, "hr_bpm": 70.0, "respiratory_rate": 14.0},
                    "candidate_action": {},
                    "constraints": {},
                },
                headers=HEADERS,
            )
            assert resp.status_code == 200, f"Failed for domain='{name}'"


class TestStepEndpointErrors:
    def test_unsupported_domain_returns_400(self):
        resp = client.post(
            "/step",
            json={
                "domain": "quantum_teleporter",
                "state": {},
                "candidate_action": {},
            },
            headers=HEADERS,
        )
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["detail"].lower()

    def test_missing_domain_field_returns_422(self):
        resp = client.post("/step", json={"state": {}, "candidate_action": {}}, headers=HEADERS)
        assert resp.status_code == 422

    def test_quantile_out_of_range_returns_422(self):
        resp = client.post(
            "/step",
            json={
                "domain": "healthcare",
                "state": {"spo2_pct": 92.0, "hr_bpm": 70.0, "respiratory_rate": 14.0},
                "candidate_action": {},
                "quantile": 150.0,  # out of range [1, 99]
            },
            headers=HEADERS,
        )
        assert resp.status_code == 422
