"""Deprecated placeholder kept for backwards compatibility only."""
from __future__ import annotations

from typing import Any


class RealDeviceDriverStub:
    """Legacy interface-compatible placeholder.

    Prefer ``ModbusTCPDriver`` for the protocol-ready non-hardware integration
    path; this stub remains only to avoid breaking older imports.
    """

    def apply_command(self, *, charge_mw: float, discharge_mw: float) -> dict[str, Any]:
        raise NotImplementedError(
            "Real hardware driver stub is deprecated. Use ModbusTCPDriver or HTTPGatewayDriver instead."
        )
