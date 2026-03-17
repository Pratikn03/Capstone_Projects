"""CertOS recovery manager: regain valid certificate after expiry."""
from __future__ import annotations

from typing import Any, Callable


class RecoveryManager:
    """Manages recovery from certificate expiry/revocation."""

    def __init__(
        self,
        on_recover: Callable[[], dict[str, Any]] | None = None,
    ):
        self._on_recover = on_recover or (lambda: {"charge_mw": 0.0, "discharge_mw": 0.0})
        self._recovery_steps = 0

    def attempt_recovery(self) -> dict[str, Any]:
        """Attempt to recover a valid certificate."""
        self._recovery_steps += 1
        return self._on_recover()

    def reset(self) -> None:
        self._recovery_steps = 0
