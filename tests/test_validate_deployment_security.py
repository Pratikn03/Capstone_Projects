"""Deployment-security umbrella validator tests."""

from __future__ import annotations

from scripts.validate_deployment_security import validate


def test_deployment_security_validator_passes() -> None:
    assert validate() == []
