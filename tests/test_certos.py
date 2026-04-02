"""CertOS runtime tests (Paper 6).

Covers:
- Certificate lifecycle: ISSUE, VALIDATE, EXPIRE, RENEW, REVOKE, FALLBACK
- Three invariants: INV-1, INV-2, INV-3
- Audit ledger completeness
- Recovery manager
- Runtime orchestrator end-to-end
"""
from __future__ import annotations

import pytest

from orius.certos.audit_ledger import AuditLedger
from orius.certos.certificate_engine import CertificateEngine, LifecycleOp
from orius.certos.recovery_manager import RecoveryManager
from orius.certos.runtime import CertOSConfig, CertOSRuntime, CertOSState, DomainGovernancePolicy


class DiscreteAlertPolicy(DomainGovernancePolicy):
    def is_actuation(self, action):
        return bool(action.get("alert_level", 0.0))

    def fallback_action(self, constraints=None, state=None):
        return {"alert_level": 0.0}


# ── Certificate Engine ────────────────────────────────────────────────

class TestCertificateEngine:
    def test_issue_creates_valid_cert(self):
        ce = CertificateEngine()
        cert = ce.issue({"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        assert cert["status"] == "valid"
        assert cert["op"] == "ISSUE"
        assert ce.validate(cert)

    def test_expire_marks_expired(self):
        ce = CertificateEngine()
        ce.issue({"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        cert = ce.expire()
        assert cert["status"] == "expired"
        assert not ce.validate(cert)

    def test_renew_creates_fresh_cert(self):
        ce = CertificateEngine()
        ce.issue({"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        cert = ce.renew({"discharge_mw": 40}, {"discharge_mw": 38}, 8)
        assert cert["status"] == "valid"
        assert ce.validate(cert)

    def test_revoke_invalidates(self):
        ce = CertificateEngine()
        ce.issue({"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        cert = ce.revoke()
        assert cert["status"] == "revoked"

    def test_fallback_returns_action(self):
        ce = CertificateEngine()
        fb = ce.fallback({"charge_mw": 0, "discharge_mw": 0})
        assert fb["op"] == "FALLBACK"

    def test_require_action_fallback_without_cert(self):
        """require_action without a cert auto-engages fallback."""
        ce = CertificateEngine()
        action = ce.require_action()
        assert action["charge_mw"] == 0.0
        assert action["discharge_mw"] == 0.0

    def test_get_safe_action_after_issue(self):
        ce = CertificateEngine()
        ce.issue({"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        sa = ce.get_safe_action()
        assert sa is not None
        assert sa["discharge_mw"] == 45


# ── Audit Ledger ──────────────────────────────────────────────────────

class TestAuditLedger:
    def test_append_and_entries(self):
        al = AuditLedger()
        al.append("ISSUE", step=0, cert_hash="abc")
        al.append("FALLBACK", step=5)
        entries = al.entries()
        assert len(entries) == 2
        assert entries[0]["op"] == "ISSUE"
        assert entries[1]["op"] == "FALLBACK"

    def test_intervention_count(self):
        al = AuditLedger()
        al.append("ISSUE", step=0)
        al.append("ISSUE", step=1)
        al.append("FALLBACK", step=2)
        al.append("REVOKE", step=3)
        al.append("EXPIRE", step=4)
        assert al.intervention_count() == 3  # FALLBACK + REVOKE + EXPIRE

    def test_empty_ledger(self):
        al = AuditLedger()
        assert len(al.entries()) == 0
        assert al.intervention_count() == 0


# ── Recovery Manager ──────────────────────────────────────────────────

class TestRecoveryManager:
    def test_recovery_with_callback(self):
        rm = RecoveryManager(on_recover=lambda: {"recovered": True})
        result = rm.attempt_recovery()
        assert result["recovered"]

    def test_recovery_without_callback(self):
        rm = RecoveryManager()
        result = rm.attempt_recovery()
        assert not result.get("recovered", False)


# ── Runtime Orchestrator ──────────────────────────────────────────────

class TestCertOSRuntime:
    def test_issue_produces_valid_state(self):
        rt = CertOSRuntime()
        state = rt.issue(
            {"discharge_mw": 50}, {"discharge_mw": 45}, validity_horizon=10
        )
        assert state.status == "valid"
        assert state.validity_horizon == 10
        assert not state.fallback_active

    def test_validate_and_step_valid(self):
        rt = CertOSRuntime()
        state = rt.validate_and_step(
            observed_soc_mwh=100.0,
            proposed_action={"discharge_mw": 50},
            safe_action={"discharge_mw": 45},
            validity_horizon=10,
        )
        assert state.status == "valid"
        assert rt.step == 1

    def test_validate_and_step_degraded(self):
        rt = CertOSRuntime(config=CertOSConfig(degraded_threshold=5))
        state = rt.validate_and_step(
            observed_soc_mwh=100.0,
            proposed_action={"discharge_mw": 50},
            safe_action={"discharge_mw": 45},
            validity_horizon=3,
        )
        assert state.status == "degraded"

    def test_validate_and_step_expired_fallback(self):
        rt = CertOSRuntime()
        state = rt.validate_and_step(
            observed_soc_mwh=100.0,
            proposed_action={"discharge_mw": 50},
            safe_action={"discharge_mw": 45},
            validity_horizon=0,
        )
        assert state.status == "fallback"
        assert state.fallback_active

    def test_revoke_triggers_fallback(self):
        rt = CertOSRuntime()
        rt.issue({"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        state = rt.revoke()
        assert state.status == "fallback"

    def test_audit_log_records_ops(self):
        rt = CertOSRuntime()
        rt.issue({"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 0)
        assert len(rt.audit_log) >= 2
        ops = [e["op"] for e in rt.audit_log]
        assert "ISSUE" in ops
        assert "FALLBACK" in ops

    def test_step_counter_advances(self):
        rt = CertOSRuntime()
        assert rt.step == 0
        rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        assert rt.step == 1
        rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        assert rt.step == 2

    def test_validate_and_step_supports_discrete_policy(self):
        rt = CertOSRuntime(config=CertOSConfig(governance_policy=DiscreteAlertPolicy()))
        state = rt.validate_and_step(
            None,
            {"alert_level": 1.0},
            {"alert_level": 1.0},
            2,
            observed_state={"alert_level": 1.0},
        )
        assert state.status == "degraded"
        assert state.safe_action["alert_level"] == 1.0


# ── Invariant Enforcement ─────────────────────────────────────────────

class TestInvariants:
    def test_inv1_no_action_without_cert(self):
        """INV-1: expired cert with nonzero action → violation."""
        rt = CertOSRuntime()
        # Manually build a state that violates INV-1
        bad_state = CertOSState(
            step=0, validity_horizon=0, status="expired",
            safe_action={"discharge_mw": 50}, certificate={},
            fallback_active=False, audit_count=0,
        )
        violations = rt.check_invariants(bad_state)
        assert "INV-1" in violations

    def test_inv3_fallback_iff_expired(self):
        """INV-3: H_t > 0 but fallback active → violation."""
        rt = CertOSRuntime()
        bad_state = CertOSState(
            step=0, validity_horizon=10, status="valid",
            safe_action={"discharge_mw": 50}, certificate={},
            fallback_active=True, audit_count=0,
        )
        violations = rt.check_invariants(bad_state)
        assert "INV-3" in violations

    def test_valid_state_no_invariant_violations(self):
        rt = CertOSRuntime()
        state = rt.issue({"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        violations = rt.check_invariants(state)
        assert violations == []

    def test_fallback_state_no_violations(self):
        rt = CertOSRuntime()
        state = rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 0)
        violations = rt.check_invariants(state)
        assert violations == []

    def test_inv2_detects_hash_chain_tamper(self):
        rt = CertOSRuntime()
        rt.issue({"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 6)
        tampered = rt.raw_audit_log[-1]
        tampered["meta"]["prev_hash"] = "tampered"
        violations = rt.check_invariants(
            CertOSState(
                step=rt.step,
                validity_horizon=6,
                status="valid",
                safe_action={"discharge_mw": 45},
                certificate={},
                fallback_active=False,
                audit_count=len(rt.audit_log),
            )
        )
        assert "INV-2" in violations


# ── Multi-Step Lifecycle ──────────────────────────────────────────────

class TestCertOSMultiStep:
    def test_full_lifecycle_valid_degrade_expire_recover(self):
        """Simulate cert going valid → degraded → expired → recovered."""
        rt = CertOSRuntime(config=CertOSConfig(degraded_threshold=5))

        # Valid
        s = rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 20)
        assert s.status == "valid"

        # Still valid
        s = rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 8)
        assert s.status == "valid"

        # Degraded
        s = rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 3)
        assert s.status == "degraded"

        # Expired → fallback
        s = rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 0)
        assert s.status == "fallback"
        assert s.fallback_active

        # Recover → new valid cert
        s = rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 15)
        assert s.status == "valid"
        assert not s.fallback_active

        # All invariants should hold for final state
        assert rt.check_invariants(s) == []

    def test_lifecycle_ops_in_audit(self):
        """Audit log should record every lifecycle operation."""
        rt = CertOSRuntime(config=CertOSConfig(degraded_threshold=5))
        rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 10)
        rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 3)
        rt.validate_and_step(100.0, {"discharge_mw": 50}, {"discharge_mw": 45}, 0)
        log = rt.audit_log
        assert len(log) >= 3
        assert rt.intervention_count >= 1


# ── CertOS Module Surface (Step 6.1) ────────────────────────────────────────


class TestCertOSModuleSurface:
    """Verify all nine CertOS modules are importable and callable."""

    def test_belief_engine(self):
        from orius.certos.belief_engine import get_belief
        b = get_belief({"soc": 0.5}, {"w": 0.9})
        assert "state" in b and "uncertainty" in b

    def test_reliability_engine(self):
        from orius.certos.reliability_engine import compute_reliability
        w, flags = compute_reliability({"soc_mwh": 50}, None, expected_cadence_s=3600.0)
        assert 0 <= w <= 1

    def test_shift_engine(self):
        from orius.certos.shift_engine import build_uncertainty_set
        lower, upper, meta = build_uncertainty_set(50.0, 5.0, 0.9)
        assert len(lower) > 0 and len(upper) > 0

    def test_reachability_engine(self):
        from orius.certos.reachability_engine import compute_validity_horizon
        constraints = {"min_soc_mwh": 10, "max_soc_mwh": 90}
        r = compute_validity_horizon(40, 60, {"discharge_mw": 10}, constraints, 5.0, 100)
        assert "tau_t" in r

    def test_safe_action_filter(self):
        from orius.certos.safe_action_filter import filter_action
        safe, meta = filter_action(
            {"discharge_mw": 50},
            {"soc_mwh": 50},
            {"lower": [40], "upper": [60]},
            {"capacity_mwh": 200},
            {},
        )
        assert "discharge_mw" in safe or "charge_mw" in safe

    def test_graceful_planner(self):
        from orius.certos.graceful_planner import plan_fallback
        actions = plan_fallback(
            {"discharge_mw": 20}, 5, 50.0,
            {"min_soc_mwh": 10, "max_soc_mwh": 90},
        )
        assert len(actions) > 0
