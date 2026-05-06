"""CertOS belief engine: state/uncertainty representation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_belief(state: Mapping[str, Any], uncertainty: Mapping[str, Any]) -> dict[str, Any]:
    """Return combined belief (state + uncertainty) for CertOS."""
    return {
        "state": dict(state),
        "uncertainty": dict(uncertainty),
    }
