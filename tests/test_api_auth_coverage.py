"""Tests for API auth coverage validation."""
from __future__ import annotations

from scripts.validate_api_auth_coverage import validate


def test_api_auth_coverage_has_no_unprotected_non_health_routes() -> None:
    assert validate() == []
