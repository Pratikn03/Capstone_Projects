"""CertOS runtime orchestrator (Paper 6).

High-level loop that combines all nine CertOS engines into a single
step-by-step runtime. Enforces the three CertOS invariants:

    INV-1: No dispatch without a valid or fallback certificate
    INV-2: Certificate hash chain is unbroken
    INV-3: Fallback triggers iff H_t ≤ 0
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from orius.certos.audit_ledger import AuditLedger
from orius.certos.certificate_engine import CertificateEngine, LifecycleOp
from orius.certos.recovery_manager import RecoveryManager


@dataclass
class CertOSConfig:
    """Configuration for the CertOS runtime."""

    soc_min_mwh: float = 20.0
    soc_max_mwh: float = 180.0
    capacity_mwh: float = 200.0
    sigma_d: float = 5.0
    degraded_threshold: int = 4
    fallback_horizon: int = 12
    alpha: float = 0.10


@dataclass
class CertOSState:
    """Complete snapshot of the CertOS runtime at one step."""

    step: int
    validity_horizon: int
    status: str  # "valid", "degraded", "expired", "fallback"
    safe_action: dict[str, float]
    certificate: dict[str, Any]
    fallback_active: bool
    audit_count: int
    lifecycle_op: str = ""
    validation_passed: bool = False
    hash_chain_ok: bool = True
    cert_hash: str | None = None
    prev_hash: str | None = None


def _to_json_safe(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    return obj


def _cert_hash(cert: Mapping[str, Any]) -> str:
    """Deterministic hash of a certificate dict."""
    safe = _to_json_safe(cert)
    payload = json.dumps(safe, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class CertOSRuntime:
    """Paper 6 runtime: orchestrates the full CertOS lifecycle."""

    def __init__(
        self,
        config: CertOSConfig | None = None,
        on_recover: Callable[[], dict[str, Any]] | None = None,
    ):
        self.cfg = config or CertOSConfig()
        self._cert_engine = CertificateEngine()
        self._audit = AuditLedger()
        self._recovery = RecoveryManager(on_recover=on_recover)
        self._step = 0
        self._prev_hash: str | None = None
        self._last_validity: int = 0

    # ── Public API ────────────────────────────────────────────────────

    def issue(
        self,
        proposed_action: Mapping[str, float],
        safe_action: Mapping[str, float],
        validity_horizon: int,
    ) -> CertOSState:
        """ISSUE: create a certificate and advance step."""
        cert = self._issue_chained_certificate(
            LifecycleOp.ISSUE,
            proposed_action,
            safe_action,
            validity_horizon,
        )
        self._last_validity = validity_horizon
        self._audit.append(
            LifecycleOp.ISSUE.value,
            self._step,
            cert["cert_hash"],
            meta={"prev_hash": cert.get("prev_hash"), "validation_passed": False},
        )
        state = self._snapshot(
            "valid",
            safe_action,
            cert,
            lifecycle_op=LifecycleOp.ISSUE.value,
            validation_passed=False,
        )
        self._step += 1
        return state

    def validate_and_step(
        self,
        observed_soc_mwh: float,
        proposed_action: Mapping[str, float],
        safe_action: Mapping[str, float],
        validity_horizon: int,
    ) -> CertOSState:
        """Main per-step entry: validate current, decide lifecycle op."""
        del observed_soc_mwh

        validation_passed, validation_meta = self._validate_current(validity_horizon)
        if validity_horizon <= 0:
            return self._handle_expired(safe_action, validation_passed, validation_meta)
        if validity_horizon <= self.cfg.degraded_threshold:
            return self._handle_degraded(
                proposed_action,
                safe_action,
                validity_horizon,
                validation_passed,
                validation_meta,
            )
        return self._handle_valid(
            proposed_action,
            safe_action,
            validity_horizon,
            validation_passed,
            validation_meta,
        )

    def revoke(self) -> CertOSState:
        """REVOKE: force invalidate current certificate."""
        cert = self._cert_engine.revoke()
        revoked_cert: dict[str, Any] = {}
        if cert is not None:
            revoked_cert = dict(cert)
            revoked_cert["prev_hash"] = self._prev_hash
            revoked_cert["cert_hash"] = _cert_hash(revoked_cert)
            self._prev_hash = revoked_cert["cert_hash"]
            self._audit.append(
                LifecycleOp.REVOKE.value,
                self._step,
                revoked_cert["cert_hash"],
                meta={"prev_hash": revoked_cert.get("prev_hash")},
            )

        fb_action = {"charge_mw": 0.0, "discharge_mw": 0.0}
        self._cert_engine.fallback(fb_action)
        self._audit.append(
            LifecycleOp.FALLBACK.value,
            self._step,
            meta={"reason": "revoked_certificate", "revoked_cert_hash": revoked_cert.get("cert_hash")},
        )
        self._last_validity = 0
        return self._snapshot(
            "fallback",
            fb_action,
            revoked_cert,
            fallback=True,
            lifecycle_op=LifecycleOp.FALLBACK.value,
            validation_passed=False,
        )

    def recover(self) -> dict[str, Any]:
        """Attempt to recover from fallback state."""
        result = self._recovery.attempt_recovery()
        if result.get("recovered"):
            self._audit.append("RECOVER", self._step, meta=result)
        return result

    @property
    def audit_log(self) -> Sequence[dict[str, Any]]:
        """Step-level public audit view for compatibility with existing tests."""
        return self._coalesce_audit_entries(self._audit.entries())

    @property
    def raw_audit_log(self) -> Sequence[dict[str, Any]]:
        """Uncollapsed lifecycle stream used for artifact generation."""
        return self._audit.entries()

    @property
    def intervention_count(self) -> int:
        return sum(
            1
            for entry in self.audit_log
            if entry.get("op") in (LifecycleOp.FALLBACK.value, LifecycleOp.REVOKE.value, LifecycleOp.EXPIRE.value)
        )

    @property
    def step(self) -> int:
        return self._step

    # ── Invariant checks ──────────────────────────────────────────────

    def check_invariants(self, state: CertOSState) -> list[str]:
        """Return list of violated invariant names (empty = all OK)."""
        violations = []

        action = state.safe_action
        if action.get("discharge_mw", 0) > 0 or action.get("charge_mw", 0) > 0:
            if state.status not in ("valid", "degraded", "fallback"):
                violations.append("INV-1")

        if not self._verify_hash_chain():
            violations.append("INV-2")

        if state.validity_horizon <= 0 and not state.fallback_active:
            violations.append("INV-3")
        if state.validity_horizon > 0 and state.fallback_active:
            violations.append("INV-3")

        return violations

    # ── Private helpers ───────────────────────────────────────────────

    def _handle_valid(
        self,
        proposed: Mapping[str, float],
        safe: Mapping[str, float],
        h_t: int,
        validation_passed: bool,
        validation_meta: Mapping[str, Any],
    ) -> CertOSState:
        cert = self._issue_chained_certificate(LifecycleOp.ISSUE, proposed, safe, h_t)
        self._last_validity = h_t
        self._audit.append(
            LifecycleOp.ISSUE.value,
            self._step,
            cert["cert_hash"],
            meta={
                "prev_hash": cert.get("prev_hash"),
                "validation_passed": validation_passed,
                **dict(validation_meta),
            },
        )
        state = self._snapshot(
            "valid",
            safe,
            cert,
            lifecycle_op=LifecycleOp.ISSUE.value,
            validation_passed=validation_passed,
        )
        self._step += 1
        return state

    def _handle_degraded(
        self,
        proposed: Mapping[str, float],
        safe: Mapping[str, float],
        h_t: int,
        validation_passed: bool,
        validation_meta: Mapping[str, Any],
    ) -> CertOSState:
        cert = self._issue_chained_certificate(LifecycleOp.RENEW, proposed, safe, h_t)
        self._last_validity = h_t
        self._audit.append(
            LifecycleOp.RENEW.value,
            self._step,
            cert["cert_hash"],
            meta={
                "prev_hash": cert.get("prev_hash"),
                "validation_passed": validation_passed,
                **dict(validation_meta),
            },
        )
        state = self._snapshot(
            "degraded",
            safe,
            cert,
            lifecycle_op=LifecycleOp.RENEW.value,
            validation_passed=validation_passed,
        )
        self._step += 1
        return state

    def _handle_expired(
        self,
        safe: Mapping[str, float],
        validation_passed: bool,
        validation_meta: Mapping[str, Any],
    ) -> CertOSState:
        expired_cert = self._cert_engine.expire()
        cert_for_state: dict[str, Any] = {"status": "expired"}
        if expired_cert is not None:
            cert_for_state = dict(expired_cert)
            cert_for_state["prev_hash"] = self._prev_hash
            cert_for_state["cert_hash"] = _cert_hash(cert_for_state)
            self._prev_hash = cert_for_state["cert_hash"]
            self._audit.append(
                LifecycleOp.EXPIRE.value,
                self._step,
                cert_for_state["cert_hash"],
                meta={
                    "prev_hash": cert_for_state.get("prev_hash"),
                    "validation_passed": validation_passed,
                    **dict(validation_meta),
                },
            )

        fb_action = {"charge_mw": 0.0, "discharge_mw": 0.0}
        self._cert_engine.fallback(fb_action)
        self._audit.append(
            LifecycleOp.FALLBACK.value,
            self._step,
            meta={
                "reason": "expired_certificate",
                "expired_cert_hash": cert_for_state.get("cert_hash"),
                "validation_passed": validation_passed,
            },
        )
        self._last_validity = 0
        state = self._snapshot(
            "fallback",
            fb_action,
            cert_for_state,
            fallback=True,
            lifecycle_op=LifecycleOp.FALLBACK.value,
            validation_passed=validation_passed,
        )
        self._step += 1
        return state

    def _snapshot(
        self,
        status: str,
        action: Mapping[str, float],
        cert: Mapping[str, Any],
        fallback: bool = False,
        lifecycle_op: str = "",
        validation_passed: bool = False,
    ) -> CertOSState:
        return CertOSState(
            step=self._step,
            validity_horizon=self._last_validity if not fallback else 0,
            status=status,
            safe_action=dict(action),
            certificate=dict(cert),
            fallback_active=fallback,
            audit_count=len(self._audit.entries()),
            lifecycle_op=lifecycle_op,
            validation_passed=validation_passed,
            hash_chain_ok=self._verify_hash_chain(),
            cert_hash=cert.get("cert_hash"),
            prev_hash=cert.get("prev_hash"),
        )

    def _issue_chained_certificate(
        self,
        op: LifecycleOp,
        proposed: Mapping[str, float],
        safe: Mapping[str, float],
        validity_horizon: int,
    ) -> dict[str, Any]:
        if op is LifecycleOp.ISSUE:
            cert = dict(self._cert_engine.issue(proposed, safe, validity_horizon))
        elif op is LifecycleOp.RENEW:
            cert = dict(self._cert_engine.renew(proposed, safe, validity_horizon))
        else:
            raise ValueError(f"Unsupported certificate op for chaining: {op}")

        cert["prev_hash"] = self._prev_hash
        cert["cert_hash"] = _cert_hash(cert)
        self._cert_engine.replace_current(cert)
        self._prev_hash = cert["cert_hash"]
        return cert

    def _validate_current(self, validity_horizon: int) -> tuple[bool, dict[str, Any]]:
        current = self._cert_engine.current_certificate
        if current is None:
            meta = {"reason": "no_active_certificate", "validity_horizon": validity_horizon, "result": False}
            self._audit.append(LifecycleOp.VALIDATE.value, self._step, meta=meta)
            return False, meta

        passed = self._cert_engine.validate(current, current_horizon=validity_horizon)
        meta = {
            "reason": "current_certificate_checked",
            "validity_horizon": validity_horizon,
            "current_status": current.get("status"),
            "current_cert_hash": current.get("cert_hash"),
            "result": passed,
        }
        self._audit.append(
            LifecycleOp.VALIDATE.value,
            self._step,
            current.get("cert_hash"),
            meta=meta,
        )
        return passed, meta

    def _verify_hash_chain(self) -> bool:
        last_hash: str | None = None
        for entry in self._audit.entries():
            if entry.get("op") not in {
                LifecycleOp.ISSUE.value,
                LifecycleOp.RENEW.value,
                LifecycleOp.EXPIRE.value,
                LifecycleOp.REVOKE.value,
            }:
                continue

            cert_hash = entry.get("cert_hash")
            if cert_hash is None:
                return False

            prev_hash = entry.get("meta", {}).get("prev_hash")
            if prev_hash != last_hash:
                return False

            last_hash = cert_hash
        return True

    def _coalesce_audit_entries(
        self,
        entries: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        collapsed: list[dict[str, Any]] = []
        grouped: dict[int, list[Mapping[str, Any]]] = {}
        for entry in entries:
            grouped.setdefault(int(entry["step"]), []).append(entry)

        for step in sorted(grouped):
            step_entries = grouped[step]
            primary = dict(step_entries[-1])
            meta = dict(primary.get("meta") or {})
            meta["ops_for_step"] = [entry["op"] for entry in step_entries]
            primary["meta"] = meta
            collapsed.append(primary)
        return collapsed
