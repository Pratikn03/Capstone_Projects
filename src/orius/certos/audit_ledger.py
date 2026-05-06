"""CertOS audit ledger: auditable intervention log."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class AuditLedger:
    """Append-only log of certificate lifecycle events."""

    def __init__(self):
        self._entries: list[dict[str, Any]] = []

    def append(
        self,
        op: str,
        step: int,
        cert_hash: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        entry = {
            "op": op,
            "step": step,
            "cert_hash": cert_hash,
            "meta": dict(meta or {}),
        }
        self._entries.append(entry)

    def entries(self) -> Sequence[dict[str, Any]]:
        return list(self._entries)

    def intervention_count(self) -> int:
        return sum(1 for e in self._entries if e.get("op") in ("REVOKE", "FALLBACK", "EXPIRE"))
