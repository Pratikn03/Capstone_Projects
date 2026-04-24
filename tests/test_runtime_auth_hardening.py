"""Regression tests for fail-closed runtime route authentication."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from services.api.config import get_api_keys, is_auth_disabled_for_tests
from services.api.main import app
from services.api.security import API_KEY_NAME


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    monkeypatch.delenv("ORIUS_AUTH_DISABLED_FOR_TESTS", raising=False)
    monkeypatch.setenv(
        "ORIUS_API_KEYS",
        json.dumps(
            {
                "read-only": ["read"],
                "write-key": ["read", "write"],
            }
        ),
    )
    get_api_keys.cache_clear()
    yield
    get_api_keys.cache_clear()


def _dc3s_payload(enqueue_iot: bool = False) -> dict:
    return {
        "device_id": "auth-device",
        "zone_id": "DE",
        "current_soc_mwh": 1.0,
        "telemetry_event": {
            "ts_utc": "2026-02-22T00:00:00+00:00",
            "load_mw": 52.0,
            "renewables_mw": 12.0,
        },
        "controller": "deterministic",
        "horizon": 24,
        "enqueue_iot": enqueue_iot,
    }


def test_dc3s_step_requires_write_scope() -> None:
    client = TestClient(app)

    missing = client.post("/dc3s/step", json=_dc3s_payload())
    read_only = client.post(
        "/dc3s/step",
        json=_dc3s_payload(),
        headers={API_KEY_NAME: "read-only"},
    )
    enqueue_read_only = client.post(
        "/dc3s/step",
        json=_dc3s_payload(enqueue_iot=True),
        headers={API_KEY_NAME: "read-only"},
    )

    assert missing.status_code == 403
    assert read_only.status_code == 401
    assert enqueue_read_only.status_code == 401


def test_universal_and_optimize_mutations_require_write_scope() -> None:
    client = TestClient(app)

    universal = client.post(
        "/universal/step",
        json={
            "domain_id": "AV",
            "raw_telemetry": {"position_m": 40.0, "speed_mps": 12.0},
            "candidate_action": {"acceleration_mps2": 0.0},
            "constraints": {},
        },
        headers={API_KEY_NAME: "read-only"},
    )
    optimize = client.post(
        "/optimize",
        json={"forecast_load_mw": [1.0], "forecast_renewables_mw": [0.0]},
        headers={API_KEY_NAME: "read-only"},
    )

    assert universal.status_code == 401
    assert optimize.status_code == 401


def test_monitor_and_prometheus_metrics_require_read_scope() -> None:
    client = TestClient(app)

    assert client.get("/monitor/research-metrics").status_code == 403
    assert client.get("/metrics").status_code == 403


def test_explicit_test_auth_bypass(monkeypatch) -> None:
    monkeypatch.setenv("ORIUS_AUTH_DISABLED_FOR_TESTS", "1")
    monkeypatch.setenv("ORIUS_ENV", "test")
    get_api_keys.cache_clear()
    client = TestClient(app)

    # Dependency bypass should allow the request to reach the handler.
    # Unknown domain proves auth was bypassed without requiring a full runtime path.
    response = client.post(
        "/universal/step",
        json={
            "domain_id": "unknown",
            "raw_telemetry": {},
            "candidate_action": {},
            "constraints": {},
        },
    )

    assert response.status_code == 404


def test_auth_bypass_flag_requires_test_context(monkeypatch) -> None:
    monkeypatch.setenv("ORIUS_AUTH_DISABLED_FOR_TESTS", "1")
    monkeypatch.delenv("ORIUS_ENV", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    assert is_auth_disabled_for_tests() is False
