"""
Tests for the supporting five-invariant reference harness in
`orius/universal/contract.py`.

Each test corresponds to one of the harness invariants. The active T11 theorem
surface is narrower and lives in `orius.universal_theory`.

Test structure:
  - TestCompliantAdapter      : a minimal adapter that passes all five checks
  - TestInvariant{1..5}       : each invariant tested via a deliberately-broken adapter
  - TestContractVerifier       : edge cases (empty sets, scalar vs. array actions)
"""
import numpy as np
import pytest

from orius.universal.contract import (
    ContractVerifier,
    ContractViolation,
    TightenedSet,
    RepairResult,
    UniversalAdapterProtocol,
)


# ── Minimal compliant adapter ──────────────────────────────────────────────────

class MinimalCompliantAdapter:
    """A minimal adapter that satisfies all five supporting harness invariants.

    Uses a 1-D box action space in [-1, 1]. The tightened set shrinks
    symmetrically with decreasing w_t, and the repair is L∞ projection.
    """

    def observe(self, raw: dict) -> tuple[np.ndarray, float]:
        z_t = np.asarray(raw.get("state", [0.0]), dtype=float)
        # Reliability is given directly in the raw packet for testing purposes.
        w_t = float(raw.get("w_t", 0.8))
        w_t = float(np.clip(w_t, 0.0, 1.0))
        return z_t, w_t

    def uncertainty_set(self, z_t: np.ndarray, w_t: float, q_t: float) -> TightenedSet:
        if w_t <= 0.0:
            # Supporting invariant 5: at zero reliability, set is empty.
            return TightenedSet(lower=np.zeros(1), upper=np.zeros(1), is_empty=True)

        # Inflation: as w_t decreases, the set shrinks.
        # At w_t = 1: inflation = q_t, set = [-1 + q_t, 1 - q_t] ≈ conformal set.
        # At w_t → 0: inflation → ∞, set collapses to empty.
        inflation = q_t / (w_t + 1e-9)
        lower = np.array([-1.0 + inflation])
        upper = np.array([ 1.0 - inflation])
        is_empty = bool(np.any(upper < lower))
        if is_empty:
            lower = np.zeros(1)
            upper = np.zeros(1)
        return TightenedSet(lower=lower, upper=upper, is_empty=is_empty)

    def repair(self, candidate: np.ndarray, safe_set: TightenedSet) -> RepairResult:
        candidate = np.asarray(candidate, dtype=float)
        if safe_set.is_empty:
            repaired = self.fallback()
        else:
            # L∞ projection: satisfies supporting invariant 1 by construction.
            repaired = np.clip(candidate, safe_set.lower, safe_set.upper)
        dist = float(np.linalg.norm(repaired - candidate))
        return RepairResult(action=repaired, was_repaired=dist > 1e-9, repair_distance=dist)

    def fallback(self) -> np.ndarray:
        return np.zeros(1)  # zero action is always safe


# ── Test: compliant adapter passes all checks ──────────────────────────────────

class TestCompliantAdapter:
    def test_passes_all_five_invariants(self):
        adapter = MinimalCompliantAdapter()
        verifier = ContractVerifier(adapter, alpha=0.05, epsilon=1e-6)
        z_t = np.array([0.3])
        q_t = 0.1
        # Should not raise
        verifier.check(z_t, q_t)

    def test_protocol_isinstance_check(self):
        adapter = MinimalCompliantAdapter()
        assert isinstance(adapter, UniversalAdapterProtocol)

    def test_verifier_with_multiple_test_points(self):
        adapter = MinimalCompliantAdapter()
        verifier = ContractVerifier(adapter, alpha=0.05)
        for z_val in [0.0, 0.5, -0.5, 0.99]:
            verifier.check(np.array([z_val]), q_t=0.05)


# ── Broken adapters for invariant violation tests ──────────────────────────────

class InvariantBroken_3_ReliabilityRange:
    """Returns w_t > 1 — violates Invariant 3."""
    def observe(self, raw):
        return np.array([0.0]), 1.5  # w_t outside [0, 1]
    def uncertainty_set(self, z_t, w_t, q_t):
        return TightenedSet(np.array([-0.9]), np.array([0.9]), False)
    def repair(self, candidate, safe_set):
        r = np.clip(np.asarray(candidate), safe_set.lower, safe_set.upper)
        return RepairResult(r, False, 0.0)
    def fallback(self):
        return np.zeros(1)


class InvariantBroken_4_CalibrationConsistency:
    """Returns a constant-width set regardless of w_t — violates Invariant 4.

    The correct behaviour is that width decreases as w_t decreases.
    This adapter returns the same width at w=1 and w=0.5, so the width
    change is 0 instead of the expected ≈ 2*q_t, triggering the check.
    """
    def observe(self, raw):
        return np.array([0.0]), 0.8
    def uncertainty_set(self, z_t, w_t, q_t):
        if w_t <= 0.0:
            return TightenedSet(np.zeros(1), np.zeros(1), True)
        # Bug: constant width regardless of w_t → width change from w=1 to w=0.5 is 0.
        # The correct change should be ≈ 2*q_t.
        constant_half = 0.5
        return TightenedSet(np.array([-constant_half]), np.array([constant_half]), False)
    def repair(self, candidate, safe_set):
        r = np.clip(np.asarray(candidate), safe_set.lower, safe_set.upper)
        return RepairResult(r, False, 0.0)
    def fallback(self):
        return np.zeros(1)


class InvariantBroken_5_ZeroReliabilityEmpty:
    """At w_t=0, returns a non-empty set — violates Invariant 5.

    Passes Invariant 4 (correct width change at w=1 vs w=0.5)
    but fails Invariant 5 (non-empty set at w=0).
    """
    def observe(self, raw):
        return np.array([0.0]), 0.8
    def uncertainty_set(self, z_t, w_t, q_t):
        # Correctly implement the inflation rule so Invariant 4 passes.
        # Bug: never returns is_empty=True, even at w_t=0.
        margin = q_t / (w_t + 1e-9) if w_t > 0 else q_t / 1e-9
        margin = min(margin, 0.9)  # clamp so set doesn't go empty by accident
        return TightenedSet(np.array([-1.0 + margin]), np.array([1.0 - margin]), False)
    def repair(self, candidate, safe_set):
        r = np.clip(np.asarray(candidate), safe_set.lower, safe_set.upper)
        return RepairResult(r, False, 0.0)
    def fallback(self):
        return np.zeros(1)


class InvariantBroken_2_TightenedSubset:
    """At w_t=0.5, the set expands below the nominal lower bound — violates Invariant 2.

    Passes Invariant 4 (width changes correctly, monotone in the right direction)
    but fails Invariant 2 (tightened lower bound dips below nominal lower bound).
    The set is shifted downward, maintaining correct width but breaking subset property.
    """
    def observe(self, raw):
        return np.array([0.0]), 0.8
    def uncertainty_set(self, z_t, w_t, q_t):
        if w_t <= 0.0:
            return TightenedSet(np.zeros(1), np.zeros(1), True)
        margin = q_t / (w_t + 1e-9)
        correct_lower = np.array([-1.0 + margin])
        correct_upper = np.array([ 1.0 - margin])
        if w_t >= 1.0:
            # Nominal at w=1: correct box, no shift
            return TightenedSet(correct_lower, correct_upper, False)
        # Bug at w<1: shift the entire interval downward by 0.5.
        # This keeps width correct (so Inv 4 passes) but lower < nominal_lower (Inv 2 fails).
        shift = np.array([0.5])
        return TightenedSet(correct_lower - shift, correct_upper - shift, False)
    def repair(self, candidate, safe_set):
        r = np.clip(np.asarray(candidate), safe_set.lower, safe_set.upper)
        return RepairResult(r, False, 0.0)
    def fallback(self):
        return np.zeros(1)


class InvariantBroken_1_RepairMembership:
    """repair() returns an action outside the tightened set — violates Invariant 1."""
    def observe(self, raw):
        return np.array([0.0]), 0.8
    def uncertainty_set(self, z_t, w_t, q_t):
        if w_t <= 0.0:
            return TightenedSet(np.zeros(1), np.zeros(1), True)
        inflation = q_t / (w_t + 1e-9)
        return TightenedSet(np.array([-1.0 + inflation]), np.array([1.0 - inflation]), False)
    def repair(self, candidate, safe_set):
        # Bug: does not clip — returns the raw candidate regardless.
        candidate = np.asarray(candidate, dtype=float)
        dist = 0.0
        return RepairResult(action=candidate, was_repaired=False, repair_distance=dist)
    def fallback(self):
        return np.zeros(1)


# ── Invariant violation tests ──────────────────────────────────────────────────

class TestInvariant3:
    def test_reliability_out_of_range_raises(self):
        verifier = ContractVerifier(InvariantBroken_3_ReliabilityRange(), alpha=0.05)
        with pytest.raises(ContractViolation, match="Invariant 3"):
            verifier.check(np.array([0.0]), q_t=0.1)


class TestInvariant4:
    def test_calibration_inconsistency_raises(self):
        verifier = ContractVerifier(InvariantBroken_4_CalibrationConsistency(), alpha=0.05)
        with pytest.raises(ContractViolation, match="Invariant 4"):
            verifier.check(np.array([0.0]), q_t=0.1)


class TestInvariant5:
    def test_zero_reliability_nonempty_raises(self):
        verifier = ContractVerifier(InvariantBroken_5_ZeroReliabilityEmpty(), alpha=0.05)
        with pytest.raises(ContractViolation, match="Invariant 5"):
            verifier.check(np.array([0.0]), q_t=0.1)


class TestInvariant2:
    def test_tightened_exceeds_nominal_raises(self):
        verifier = ContractVerifier(InvariantBroken_2_TightenedSubset(), alpha=0.05)
        with pytest.raises(ContractViolation, match="Invariant 2"):
            verifier.check(np.array([0.0]), q_t=0.1)


class TestInvariant1:
    def test_repair_outside_safe_set_raises(self):
        verifier = ContractVerifier(InvariantBroken_1_RepairMembership(), alpha=0.05)
        with pytest.raises(ContractViolation, match="Invariant 1"):
            verifier.check(np.array([0.0]), q_t=0.1)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestContractVerifierEdgeCases:
    def test_raises_if_adapter_missing_methods(self):
        """Non-adapter object should fail isinstance check in ContractVerifier."""
        class NotAnAdapter:
            pass
        with pytest.raises(TypeError, match="UniversalAdapterProtocol"):
            ContractVerifier(NotAnAdapter(), alpha=0.05)  # type: ignore

    def test_tolerance_parameter_does_not_suppress_large_violations(self):
        """A large invariant violation (10x) should not be masked by the default tol."""
        verifier = ContractVerifier(InvariantBroken_1_RepairMembership(), alpha=0.05)
        with pytest.raises(ContractViolation):
            verifier.check(np.array([0.0]), q_t=0.1, tol=1e-6)
