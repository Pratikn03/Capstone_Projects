"""CertOS certificate engine: six lifecycle operations.

ISSUE, VALIDATE, EXPIRE, RENEW, REVOKE, FALLBACK.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


class LifecycleOp(str, Enum):
    ISSUE = "ISSUE"
    VALIDATE = "VALIDATE"
    EXPIRE = "EXPIRE"
    RENEW = "RENEW"
    REVOKE = "REVOKE"
    FALLBACK = "FALLBACK"


class CertificateEngine:
    """Manages certificate lifecycle: no action without certificate/fallback."""

    def __init__(self):
        self._current: Mapping[str, Any] | None = None
        self._fallback_active = False

    def issue(
        self,
        proposed_action: Mapping[str, Any],
        safe_action: Mapping[str, Any],
        validity_horizon: int = 1,
    ) -> Mapping[str, Any]:
        """ISSUE: Create a new certificate."""
        cert = {
            "op": LifecycleOp.ISSUE.value,
            "proposed_action": dict(proposed_action),
            "safe_action": dict(safe_action),
            "validity_horizon": validity_horizon,
            "status": "valid",
        }
        self._current = cert
        self._fallback_active = False
        return cert

    def validate(self, cert: Mapping[str, Any]) -> bool:
        """VALIDATE: Check certificate is still valid."""
        return cert.get("status") == "valid" and cert.get("validity_horizon", 0) > 0

    def expire(self) -> Mapping[str, Any] | None:
        """EXPIRE: Mark certificate as expired."""
        if self._current is None:
            return None
        self._current = dict(self._current)
        self._current["status"] = "expired"
        self._current["op"] = LifecycleOp.EXPIRE.value
        return self._current

    def renew(
        self,
        proposed_action: Mapping[str, Any],
        safe_action: Mapping[str, Any],
        validity_horizon: int = 1,
    ) -> Mapping[str, Any]:
        """RENEW: Issue a new certificate (replacement)."""
        return self.issue(proposed_action, safe_action, validity_horizon)

    def revoke(self) -> Mapping[str, Any] | None:
        """REVOKE: Invalidate the certificate."""
        if self._current is None:
            return None
        self._current = dict(self._current)
        self._current["status"] = "revoked"
        self._current["op"] = LifecycleOp.REVOKE.value
        return self._current

    def fallback(self, fallback_action: Mapping[str, Any]) -> Mapping[str, Any]:
        """FALLBACK: Use fallback action when certificate invalid."""
        self._fallback_active = True
        return {
            "op": LifecycleOp.FALLBACK.value,
            "action": dict(fallback_action),
            "reason": "certificate_invalid",
        }

    def get_safe_action(self) -> Mapping[str, Any] | None:
        """Return the current safe action (certificate or fallback)."""
        if self._fallback_active:
            return {"charge_mw": 0.0, "discharge_mw": 0.0}
        if self._current and self.validate(self._current):
            return self._current.get("safe_action")
        return None

    def require_action(self) -> Mapping[str, Any]:
        """Enforce invariant: no action without certificate/fallback."""
        action = self.get_safe_action()
        if action is not None:
            return action
        return self.fallback({"charge_mw": 0.0, "discharge_mw": 0.0})["action"]
