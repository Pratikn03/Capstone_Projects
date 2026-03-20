"""Tests for the ORIUS /step FastAPI endpoint.

Phase 4 of ORIUS gap-closing plan.
Tests health check, all 6 supported domains, error handling.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orius.api.app import app

client = TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "orius" in data["service"].lower()


class TestStepEndpointBasic:
    def _step(self, domain: str, state: dict, candidate_action: dict | None = None,
              constraints: dict | None = None) -> dict:
        payload = {
            "domain": domain,
            "state": state,
            "candidate_action": candidate_action or {},
            "constraints": constraints or {},
        }
        resp = client.post("/step", json=payload)
        assert resp.status_code == 200, f"domain={domain}: {resp.text}"
        return resp.json()

    def test_industrial_domain(self):
        data = self._step(
            domain="industrial",
            state={"power_mw": 490.0, "temp_c": 112.0, "pressure_mbar": 1010.0},
            candidate_action={"power_setpoint_mw": 520.0},
            constraints={"power_max_mw": 500.0},
        )
        assert "safe_action" in data
        assert "reliability_w" in data
        assert "repaired" in data
        assert 0.0 <= data["reliability_w"] <= 1.0

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
            state={"speed_ms": 25.0, "lateral_error_m": 0.5},
            candidate_action={"brake_force": 0.3},
            constraints={},
        )
        assert "safe_action" in data

    def test_aerospace_domain(self):
        data = self._step(
            domain="aerospace",
            state={"altitude_m": 10000.0, "airspeed_kts": 250.0},
            candidate_action={"thrust_reduction": 0.0},
            constraints={},
        )
        assert "safe_action" in data

    def test_navigation_domain(self):
        data = self._step(
            domain="navigation",
            state={"heading_deg": 45.0, "speed_ms": 10.0},
            candidate_action={"speed_reduction": 0.0},
            constraints={},
        )
        assert "safe_action" in data

    def test_response_has_certificate(self):
        data = self._step(
            domain="industrial",
            state={"power_mw": 450.0, "temp_c": 100.0, "pressure_mbar": 1010.0},
        )
        assert "certificate" in data

    def test_response_has_uncertainty_set(self):
        data = self._step(
            domain="industrial",
            state={"power_mw": 450.0, "temp_c": 100.0},
        )
        assert "uncertainty_set" in data

    def test_repaired_flag_type(self):
        data = self._step(
            domain="industrial",
            state={"power_mw": 450.0, "temp_c": 100.0},
        )
        assert isinstance(data["repaired"], bool)

    def test_domain_case_insensitive(self):
        """Domain name should be case-insensitive."""
        for name in ("INDUSTRIAL", "Industrial", "industrial"):
            resp = client.post("/step", json={
                "domain": name,
                "state": {"power_mw": 450.0},
                "candidate_action": {},
                "constraints": {},
            })
            assert resp.status_code == 200, f"Failed for domain='{name}'"


class TestStepEndpointErrors:
    def test_unsupported_domain_returns_400(self):
        resp = client.post("/step", json={
            "domain": "quantum_teleporter",
            "state": {},
            "candidate_action": {},
        })
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["detail"].lower()

    def test_missing_domain_field_returns_422(self):
        resp = client.post("/step", json={"state": {}, "candidate_action": {}})
        assert resp.status_code == 422

    def test_quantile_out_of_range_returns_422(self):
        resp = client.post("/step", json={
            "domain": "industrial",
            "state": {"power_mw": 450.0},
            "candidate_action": {},
            "quantile": 150.0,  # out of range [1, 99]
        })
        assert resp.status_code == 422
