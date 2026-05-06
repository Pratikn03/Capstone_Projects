"""Tests for DC3S route certificate signing helper."""

from __future__ import annotations

import pytest

from orius.dc3s.certificate import make_certificate, verify_certificate
from services.api.routers.dc3s import _sign_certificate_if_configured


def _cert() -> dict:
    return make_certificate(
        command_id="route-signing-test",
        device_id="device",
        zone_id="DE",
        controller="test",
        proposed_action={"charge_mw": 0.0},
        safe_action={"charge_mw": 0.0},
        uncertainty={},
        reliability={},
        drift={},
        model_hash="m",
        config_hash="c",
    )


def test_dc3s_route_signs_when_key_is_configured(monkeypatch) -> None:
    monkeypatch.setenv("ORIUS_ENV", "dev")
    monkeypatch.setenv("ORIUS_CERTIFICATE_SIGNING_KEY", "test-secret-with-enough-length-123")

    signed = _sign_certificate_if_configured(_cert())

    assert signed["signature"]
    verification = verify_certificate(
        signed,
        require_signature=True,
        signature_secret="test-secret-with-enough-length-123",
    )
    assert verification["valid"] is True


def test_dc3s_route_requires_signing_key_in_production(monkeypatch) -> None:
    monkeypatch.setenv("ORIUS_ENV", "production")
    monkeypatch.delenv("ORIUS_CERTIFICATE_SIGNING_KEY", raising=False)
    monkeypatch.delenv("ORIUS_CERTIFICATE_KEYS", raising=False)

    with pytest.raises(Exception, match="certificate signing required"):
        _sign_certificate_if_configured(_cert())
