"""Regression tests for deployment-readiness gates."""

from __future__ import annotations

import json
import os

from scripts.validate_production_readiness import validate
from services.api.config import get_api_keys


def _clear_auth_cache() -> None:
    get_api_keys.cache_clear()


def test_strict_readiness_fails_without_production_secrets(monkeypatch):
    monkeypatch.delenv("ORIUS_API_KEYS", raising=False)
    monkeypatch.delenv("ORIUS_CERTIFICATE_SIGNING_KEY", raising=False)
    monkeypatch.setenv("ORIUS_ENV", "production")
    _clear_auth_cache()

    findings, _warnings = validate(strict=True)

    assert any("ORIUS_API_KEYS" in finding for finding in findings)
    assert any("ORIUS_CERTIFICATE_SIGNING_KEY" in finding for finding in findings)


def test_strict_readiness_accepts_auth_and_signing_config(monkeypatch):
    monkeypatch.setenv("ORIUS_ENV", "production")
    monkeypatch.setenv("ORIUS_API_KEYS", json.dumps({"prod-key": ["read", "write", "admin"]}))
    monkeypatch.setenv("ORIUS_CERTIFICATE_SIGNING_KEY", "x" * 40)
    _clear_auth_cache()

    findings, _warnings = validate(strict=True)

    assert findings == []


def test_auth_bypass_flag_is_not_effective_in_production(monkeypatch):
    monkeypatch.setenv("ORIUS_ENV", "production")
    monkeypatch.setenv("ORIUS_AUTH_DISABLED_FOR_TESTS", "1")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("ORIUS_API_KEYS", json.dumps({"prod-key": ["read", "write", "admin"]}))
    monkeypatch.setenv("ORIUS_CERTIFICATE_SIGNING_KEY", "x" * 40)
    _clear_auth_cache()

    findings, _warnings = validate(strict=True)

    assert "auth bypass is active outside a test environment" not in findings
    assert os.getenv("ORIUS_AUTH_DISABLED_FOR_TESTS") == "1"
