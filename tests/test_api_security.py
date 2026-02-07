"""Integration tests for API security and islanding behavior."""
import json
import os

from fastapi.testclient import TestClient

from services.api.config import get_api_keys
from services.api.main import app, watchdog
from services.api.security import API_KEY_NAME


def _set_api_keys_env() -> None:
    keys = {
        "admin-secret-123": ["read", "write", "admin"],
        "operator-secret-456": ["read", "write"],
        "analyst-secret-789": ["read"],
    }
    os.environ["GRIDPULSE_API_KEYS"] = json.dumps(keys)
    get_api_keys.cache_clear()


def test_heartbeat_requires_write_scope():
    _set_api_keys_env()
    with TestClient(app) as client:
        resp = client.post("/system/heartbeat", headers={API_KEY_NAME: "analyst-secret-789"})
        assert resp.status_code == 401
        assert "Missing required scope" in resp.json()["detail"]

        resp = client.post("/system/heartbeat", headers={API_KEY_NAME: "operator-secret-456"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "heartbeat_received"


def test_dispatch_rejected_when_islanded():
    _set_api_keys_env()
    with TestClient(app) as client:
        watchdog.trigger_island_mode()
        try:
            payload = {"charge_mw": 0.0, "discharge_mw": 0.0, "current_soc_mwh": 5.0}
            resp = client.post(
                "/control/dispatch",
                headers={API_KEY_NAME: "operator-secret-456"},
                json=payload,
            )
            assert resp.status_code == 503
            assert resp.json()["detail"] == "System is ISLANDED. Remote control rejected."
        finally:
            watchdog.is_islanded = False
