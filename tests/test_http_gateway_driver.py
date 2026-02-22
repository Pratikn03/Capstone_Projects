"""Unit tests for HTTP gateway driver retry and normalization behavior."""
from __future__ import annotations

import requests

from iot.edge_agent.drivers.http_gateway import HTTPGatewayDriver


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def request(self, method, url, json=None, headers=None, timeout=None):
        self.calls.append((method, url, json, headers, timeout))
        if not self._responses:
            raise RuntimeError("No fake responses left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        return None


def test_fetch_telemetry_normalizes_gateway_fields():
    session = _FakeSession(
        [
            _FakeResponse(
                status_code=200,
                payload={
                    "timestamp": "2026-02-22T00:00:00+00:00",
                    "load": 100.0,
                    "renewables": 25.5,
                    "soc_mwh": 3.0,
                },
            )
        ]
    )
    driver = HTTPGatewayDriver(base_url="http://gw.local", session=session)
    out = driver.fetch_telemetry()
    assert out["ts_utc"] == "2026-02-22T00:00:00+00:00"
    assert out["load_mw"] == 100.0
    assert out["renewables_mw"] == 25.5
    assert out["soc_mwh"] == 3.0


def test_apply_command_retries_after_5xx():
    session = _FakeSession(
        [
            _FakeResponse(status_code=500, payload={"error": "temporary"}),
            _FakeResponse(status_code=200, payload={"accepted": True, "violation": False}),
        ]
    )
    driver = HTTPGatewayDriver(base_url="http://gw.local", retries=2, session=session)
    out = driver.apply_command(charge_mw=1.0, discharge_mw=0.0)
    assert out["accepted"] is True
    assert len(session.calls) == 2


def test_gateway_request_raises_after_retry_exhaustion():
    session = _FakeSession([requests.Timeout("timeout-1"), requests.Timeout("timeout-2")])
    driver = HTTPGatewayDriver(base_url="http://gw.local", retries=1, session=session)
    try:
        driver.fetch_telemetry()
    except RuntimeError as exc:
        assert "failed after retries" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for exhausted retries")
