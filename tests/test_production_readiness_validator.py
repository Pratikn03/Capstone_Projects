"""Regression tests for deployment-readiness gates."""

from __future__ import annotations

import json
import os
import sys

from orius.security.policy import get_certificate_keys, get_device_keys, secret_backend_is_managed
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


def test_strict_readiness_requires_profile_flags_and_device_keys(monkeypatch):
    monkeypatch.setenv("ORIUS_ENV", "production")
    monkeypatch.setenv("ORIUS_API_KEYS", json.dumps({"prod-key": ["read", "write", "admin"]}))
    monkeypatch.setenv("ORIUS_CERTIFICATE_KEYS", json.dumps({"cert-2026-01": "x" * 40}))
    monkeypatch.setenv("ORIUS_CERTIFICATE_ACTIVE_KEY_ID", "cert-2026-01")
    monkeypatch.delenv("ORIUS_REQUIRE_CERT_SIGNATURE", raising=False)
    monkeypatch.delenv("ORIUS_REQUIRE_DEVICE_SIGNATURE", raising=False)
    monkeypatch.delenv("ORIUS_DEVICE_KEYS", raising=False)
    monkeypatch.delenv("ORIUS_REQUIRE_ARTIFACT_MANIFEST", raising=False)
    monkeypatch.delenv("ORIUS_REQUIRE_MODEL_HASH", raising=False)
    _clear_auth_cache()

    findings, _warnings = validate(strict=True)

    assert any("ORIUS_REQUIRE_CERT_SIGNATURE=1" in finding for finding in findings)
    assert any("ORIUS_REQUIRE_DEVICE_SIGNATURE=1" in finding for finding in findings)
    assert any("configured ORIUS_DEVICE_KEYS" in finding for finding in findings)
    assert any("ORIUS_REQUIRE_ARTIFACT_MANIFEST=1" in finding for finding in findings)
    assert any("ORIUS_REQUIRE_MODEL_HASH=1" in finding for finding in findings)


def test_strict_readiness_requires_active_certificate_key_id_for_rotation(monkeypatch):
    monkeypatch.setenv("ORIUS_ENV", "production")
    monkeypatch.setenv("ORIUS_API_KEYS", json.dumps({"prod-key": ["read", "write", "admin"]}))
    monkeypatch.setenv("ORIUS_CERTIFICATE_SIGNING_KEY", "x" * 40)
    monkeypatch.delenv("ORIUS_CERTIFICATE_ACTIVE_KEY_ID", raising=False)
    monkeypatch.delenv("ORIUS_CERTIFICATE_KEY_ID", raising=False)
    monkeypatch.setenv("ORIUS_REQUIRE_CERT_SIGNATURE", "1")
    monkeypatch.setenv("ORIUS_REQUIRE_DEVICE_SIGNATURE", "1")
    monkeypatch.setenv("ORIUS_DEVICE_KEYS", json.dumps({"device-1": {"key-1": "y" * 40}}))
    monkeypatch.setenv("ORIUS_REQUIRE_ARTIFACT_MANIFEST", "1")
    monkeypatch.setenv("ORIUS_REQUIRE_MODEL_HASH", "1")
    _clear_auth_cache()

    findings, _warnings = validate(strict=True)

    assert any("ORIUS_CERTIFICATE_ACTIVE_KEY_ID" in finding for finding in findings)


def test_strict_readiness_accepts_auth_signing_and_device_identity_config(monkeypatch):
    monkeypatch.setenv("ORIUS_ENV", "production")
    monkeypatch.setenv("ORIUS_API_KEYS", json.dumps({"prod-key": ["read", "write", "admin"]}))
    monkeypatch.setenv("ORIUS_CERTIFICATE_KEYS", json.dumps({"cert-2026-01": "x" * 40}))
    monkeypatch.setenv("ORIUS_CERTIFICATE_ACTIVE_KEY_ID", "cert-2026-01")
    monkeypatch.setenv("ORIUS_DEVICE_KEYS", json.dumps({"device-1": {"key-1": "y" * 40}}))
    monkeypatch.setenv("ORIUS_REQUIRE_CERT_SIGNATURE", "1")
    monkeypatch.setenv("ORIUS_REQUIRE_DEVICE_SIGNATURE", "1")
    monkeypatch.setenv("ORIUS_REQUIRE_ARTIFACT_MANIFEST", "1")
    monkeypatch.setenv("ORIUS_REQUIRE_MODEL_HASH", "1")
    _clear_auth_cache()

    findings, _warnings = validate(strict=True)

    assert findings == []


def test_deployment_grade_requires_managed_secret_backend(monkeypatch):
    monkeypatch.setenv("ORIUS_ENV", "production")
    monkeypatch.setenv("ORIUS_API_KEYS", json.dumps({"prod-key": ["read", "write", "admin"]}))
    monkeypatch.setenv("ORIUS_CERTIFICATE_SIGNING_KEY", "x" * 40)
    monkeypatch.setenv("ORIUS_DEVICE_KEYS", json.dumps({"device-1": {"key-1": "y" * 40}}))
    monkeypatch.setenv("ORIUS_REQUIRE_CERT_SIGNATURE", "1")
    monkeypatch.setenv("ORIUS_REQUIRE_DEVICE_SIGNATURE", "1")
    monkeypatch.setenv("ORIUS_REQUIRE_ARTIFACT_MANIFEST", "1")
    monkeypatch.setenv("ORIUS_REQUIRE_MTLS", "1")
    monkeypatch.delenv("ORIUS_SECRET_BACKEND", raising=False)
    monkeypatch.delenv("ORIUS_SECRETS_COMMAND", raising=False)
    _clear_auth_cache()

    findings, _warnings = validate(deployment_grade=True)

    assert any("managed secret backend" in finding for finding in findings)
    assert any("explicit --release-manifest" in finding for finding in findings)


def test_deployment_grade_rejects_managed_label_with_env_secret_material(monkeypatch):
    monkeypatch.setenv("ORIUS_ENV", "production")
    monkeypatch.setenv("ORIUS_SECRET_BACKEND", "aws_kms")
    monkeypatch.setenv("ORIUS_API_KEYS", json.dumps({"prod-key": ["read", "write", "admin"]}))
    monkeypatch.setenv("ORIUS_CERTIFICATE_KEYS", json.dumps({"cert-1": "x" * 40}))
    monkeypatch.setenv("ORIUS_CERTIFICATE_ACTIVE_KEY_ID", "cert-1")
    monkeypatch.setenv("ORIUS_DEVICE_KEYS", json.dumps({"device-1": {"key-1": "y" * 40}}))
    monkeypatch.setenv("ORIUS_REQUIRE_CERT_SIGNATURE", "1")
    monkeypatch.setenv("ORIUS_REQUIRE_DEVICE_SIGNATURE", "1")
    monkeypatch.setenv("ORIUS_REQUIRE_ARTIFACT_MANIFEST", "1")
    monkeypatch.setenv("ORIUS_REQUIRE_MTLS", "1")
    _clear_auth_cache()

    findings, _warnings = validate(deployment_grade=True)

    assert any("implemented managed secret backend" in finding for finding in findings)
    assert any("must not source certificate/device secrets from env/local files" in finding for finding in findings)


def test_external_command_secret_backend_supplies_rotation_keys(monkeypatch, tmp_path):
    command = tmp_path / "emit_secrets.py"
    command.write_text(
        "import json\n"
        "print(json.dumps({"
        "'certificate_keys': {'cert-1': 'x' * 40}, "
        "'device_keys': {'device-1': {'key-1': 'y' * 40}}"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ORIUS_SECRET_BACKEND", "external_command")
    monkeypatch.setenv("ORIUS_SECRETS_COMMAND", f"{sys.executable} {command}")
    monkeypatch.setenv("ORIUS_CERTIFICATE_ACTIVE_KEY_ID", "cert-1")
    monkeypatch.delenv("ORIUS_CERTIFICATE_KEYS", raising=False)
    monkeypatch.delenv("ORIUS_DEVICE_KEYS", raising=False)

    assert secret_backend_is_managed()
    assert get_certificate_keys()["cert-1"] == "x" * 40
    assert get_device_keys()["device-1"]["key-1"] == "y" * 40


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
