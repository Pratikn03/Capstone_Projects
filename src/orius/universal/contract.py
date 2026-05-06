"""
Reference adapter contract for the ORIUS universality program.

This module defines the formal contract that any domain adapter must satisfy
to instantiate the simplified theorem miniatures used in the thesis
unification chapter. The canonical production runtime contract now lives in
``orius.universal_theory``; this module remains as a compact, machine-checkable
reference harness for pedagogical adapters such as CBF and Robust MPC
reductions.

Background — why this exists
-----------------------------
The original DomainAdapter (src/orius/dc3s/domain_adapter.py) used
``Mapping[str, Any]`` for all arguments and return types. That design is
correct for flexibility but provides zero static or runtime safety guarantees:
nothing prevents a buggy adapter from returning a repaired action *outside*
the tightened set, violating the core invariant that T2's proof relies on.

This module restates the adapter interface with:
  1. Typed dataclasses (TightenedSet, RepairResult) whose fields encode the
     algebraic structure that T2 and T11 require.
  2. A ContractVerifier that runs the five formal invariant checks at
     initialization or at arbitrary test points, giving a machine-checkable
     certificate that the adapter satisfies the universality contract.

A domain adapter that passes ``ContractVerifier.check()`` is compatible with
the simplified adapter contract used in the universality chapter's reference
reductions. The defended production claim remains battery-anchored and tiered.

Supporting Mini-Harness for Active T11
--------------------------------------
The active manuscript's T11 surface is a four-obligation one-step transfer
theorem plus a separate episode-aggregation corollary that requires an
explicit per-step risk budget. The five checks below are a compact supporting
reference harness for theorem miniatures; passing them is evidence that an
adapter has the right shape, not a standalone proof that the full active T11
surface or the episode-level T3 envelope has been discharged.

The Five Invariants (supporting reference harness)
--------------------------------------------------
  Inv 1: repair(a, A_t) ∈ A_t            — repair stays in tightened set
  Inv 2: A_t ⊆ A_nominal                  — tightened ⊆ nominal
  Inv 3: w_t ∈ [0, 1]                    — reliability is a bounded runtime score
  Inv 4: uncertainty_set(w=1) = conf_set  — calibration consistency at full reliability
  Inv 5: uncertainty_set(w=0).is_empty    — total failure → only fallback admissible
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# ── Typed value objects ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SafetyBound:
    """Parameters that characterise the DC3S safety guarantee.

    These are the two scalar parameters that appear in the supporting
    risk-envelope algebra (α) and in the inflation rule denominator (ε). Holding them in
    a frozen dataclass prevents accidental mutation during verification.

    Attributes:
        alpha   : target miscoverage rate, e.g. 0.05 for 95% coverage.
                  The active episode-level bound still needs a separate
                  predictable per-step risk-budget argument.
        epsilon : numerical stability floor added to the reliability
                  denominator: inflation = q / (w + ε). Prevents
                  division by zero when w_t → 0.
    """

    alpha: float
    epsilon: float


@dataclass(frozen=True)
class TightenedSet:
    """Output of Stage 3 (Constrain / FTIT): the reliability-aware safe action set A_t.

    Geometrically this is an axis-aligned box in action space. More complex
    geometries (polytopes, ellipsoids) can be represented by projecting to
    their bounding box for purposes of the invariant checks.

    T11 Invariant 2: lower and upper define a set that is a *subset* of the
    nominal constraint set (tightening is monotone in w_t).
    T11 Invariant 1: any output of repair() must satisfy
        lower ≤ result.action ≤ upper  (component-wise).

    Attributes:
        lower    : lower bound per action dimension, shape (action_dim,).
        upper    : upper bound per action dimension, shape (action_dim,).
        is_empty : True iff the tightened set is empty, meaning no safe
                   action exists and the fallback must be used. This
                   satisfies supporting invariant 5 at w_t = 0.
    """

    lower: np.ndarray
    upper: np.ndarray
    is_empty: bool


@dataclass(frozen=True)
class RepairResult:
    """Output of Stage 4 (Shield): the projected safe action.

    T11 Invariant 1 requires action ∈ tightened_set. ContractVerifier
    enforces this at runtime; the type system cannot enforce set membership
    directly, so the verifier is the machine-checkable certificate.

    Attributes:
        action          : the repaired action, shape (action_dim,).
        was_repaired    : True iff action ≠ candidate (shield fired).
        repair_distance : L2 distance from candidate to action. Used
                          for performance accounting (repair rate metric).
    """

    action: np.ndarray
    was_repaired: bool
    repair_distance: float


# ── Universal adapter Protocol ─────────────────────────────────────────────────


@runtime_checkable
class UniversalAdapterProtocol(Protocol):
    """Structural typing contract for ORIUS domain adapters.

    Python's ``@runtime_checkable Protocol`` provides two levels of checking:
      - Static: mypy / pyright verify method signatures at type-check time.
      - Runtime: ``isinstance(adapter, UniversalAdapterProtocol)`` checks
        that the four methods exist (but not their return types or invariants).

    Full runtime invariant verification for this supporting five-invariant
    mini-harness is provided
    by ContractVerifier, which calls each method and checks the outputs.

    Implementation note: the four methods are intentionally minimal. Adapters
    may have additional methods for telemetry parsing, certificate emission, etc.
    Only these four are required for the supporting harness; the active T11
    theorem still needs the four manuscript obligations to be discharged.
    """

    def observe(self, raw: dict) -> tuple[np.ndarray, float]:
        """Parse raw telemetry into (observed_state z_t, reliability_score w_t).

        This combines DC3S Stage 1 (Detect / OQE) with telemetry ingestion.
        Separating these two concerns is optional — what matters for T11 is
        that the returned w_t satisfies Invariant 3.

        Args:
            raw: domain-specific telemetry packet (sensor readings, timestamps, etc.)

        Returns:
            z_t : observed state vector, shape (state_dim,). May differ from
                  the true state x_t by up to g(w_t) (Assumption A2).
            w_t : reliability score in [0, 1].  ← T11 Invariant 3
                  w_t = 1.0 means observation is as good as the calibration set.
                  w_t = 0.0 means the observation is completely unreliable.
        """
        ...

    def uncertainty_set(self, z_t: np.ndarray, w_t: float, q_t: float) -> TightenedSet:
        """Compute the reliability-adjusted tightened safe action set A_t.

        This implements DC3S Stages 2 (Calibrate) and 3 (Constrain) jointly.
        The returned set must satisfy:

            T11 Invariant 4: uncertainty_set(z_t, w=1.0, q_t) corresponds to
                the conformal prediction set of width 2·q_t. At full reliability,
                no extra inflation is applied.

            T11 Invariant 5: uncertainty_set(z_t, w=0.0, q_t).is_empty == True.
                At zero reliability, the observation is useless — the only safe
                action is the domain's fallback (Assumption A3/A8).

            T11 Invariant 2: the returned set is a subset of the nominal
                constraint set at every w_t ∈ [0, 1].

        Args:
            z_t : observed state vector from observe().
            w_t : reliability score from observe(), in [0, 1].
            q_t : conformal quantile for the current calibration window.
                  Width of the base conformal prediction interval is 2·q_t.

        Returns:
            TightenedSet with lower/upper bounds and is_empty flag.
        """
        ...

    def repair(self, candidate: np.ndarray, safe_set: TightenedSet) -> RepairResult:
        """Project the candidate action into safe_set (DC3S Stage 4: Shield).

        Must satisfy T11 Invariant 1:
            result.action ∈ safe_set  (i.e., safe_set.lower ≤ result.action ≤ safe_set.upper)

        When safe_set.is_empty, the adapter must return the fallback action.
        ContractVerifier tests this invariant by passing a deliberately
        out-of-bounds candidate and checking the returned action.

        Args:
            candidate : proposed action from the upstream controller, shape (action_dim,).
            safe_set  : TightenedSet from uncertainty_set(), the feasible region.

        Returns:
            RepairResult with the projected action and repair metadata.
        """
        ...

    def fallback(self) -> np.ndarray:
        """Return the always-safe fallback action (Assumptions A3 and A8).

        The fallback is the action taken when safe_set.is_empty (total
        observation failure) or when the shield cannot find a feasible repair.
        Examples: zero dispatch for battery, zero velocity for AV, zero
        infusion rate for ICU pump.

        Must satisfy the domain-specific precondition:
            fallback() ∈ nominal_constraints
        This precondition is domain-specific and checked by the domain expert,
        not by ContractVerifier (which is domain-agnostic).

        Returns:
            Safe fallback action, shape (action_dim,).
        """
        ...


# ── Contract verifier ──────────────────────────────────────────────────────────


class ContractVerifier:
    """Runtime checker for the supporting five-invariant reference harness.

    Passing these checks shows that an adapter fits the compact theorem
    miniature used by the unification chapter. It does not by itself prove
    that the active manuscript's four T11 obligations hold end to end.

    Usage
    -----
    ::

        from orius.universal.contract import ContractVerifier

        adapter = MyDomainAdapter(...)
        verifier = ContractVerifier(adapter, alpha=0.05, epsilon=1e-6)

        # Check at a specific test point:
        z_t = np.array([0.5, 0.3])   # example observed state
        verifier.check(z_t, q_t=0.1) # raises ContractViolation if any invariant fails

        # If no exception is raised, the adapter satisfies the supporting harness.

    Implementation notes
    --------------------
    - Each of the five private _check_* methods corresponds to exactly one
      supporting mini-harness invariant. The mapping is documented in each method.
    - The ``tol`` parameter handles floating-point rounding; it is not a
      relaxation of the formal invariant.
    - ContractVerifier is stateless between calls to check() — it does not
      accumulate state across multiple test points.
    """

    def __init__(
        self,
        adapter: UniversalAdapterProtocol,
        alpha: float,
        epsilon: float = 1e-6,
    ) -> None:
        """
        Args:
            adapter : the domain adapter to verify.
            alpha   : miscoverage rate used in SafetyBound (e.g. 0.05).
            epsilon : stability floor for inflation denominator (e.g. 1e-6).
        """
        if not isinstance(adapter, UniversalAdapterProtocol):
            raise TypeError(
                f"{type(adapter).__name__} does not implement UniversalAdapterProtocol. "
                "Ensure observe(), uncertainty_set(), repair(), and fallback() are defined."
            )
        self.adapter = adapter
        self.bound = SafetyBound(alpha=alpha, epsilon=epsilon)

    def check(self, z_t: np.ndarray, q_t: float, tol: float = 1e-6) -> None:
        """Run all five supporting mini-harness checks at the given test point.

        Invariants are checked in dependency order:
          3 (range) → 4 (calibration) → 5 (zero-rel fallback) → 2 (subset) → 1 (repair)

        Args:
            z_t : observed state vector to test at.
            q_t : conformal quantile to test with (e.g. 0.1 MWh).
            tol : absolute tolerance for floating-point comparisons.

        Raises:
            ContractViolation: describes which invariant failed and why.
        """
        self._check_invariant_3_reliability_range(z_t)
        self._check_invariant_4_calibration_consistency(z_t, q_t, tol)
        self._check_invariant_5_zero_reliability_empty(z_t, q_t)
        self._check_invariant_2_tightened_subset(z_t, q_t, tol)
        self._check_invariant_1_repair_membership(z_t, q_t, tol)

    # ── Private invariant checks ───────────────────────────────────────────────

    def _check_invariant_3_reliability_range(self, z_t: np.ndarray) -> None:
        """Supporting invariant 3: reliability score w_t must be in [0, 1].

        Theoretical requirement: w_t is a bounded runtime reliability score
        that parameterizes tightening and any later risk-envelope algebra.
        Values outside [0, 1] break that interpretation.
        """
        _, w_t = self.adapter.observe({"state": z_t.tolist()})
        if not (0.0 <= float(w_t) <= 1.0):
            raise ContractViolation(
                f"Invariant 3 (reliability range) FAILED: "
                f"observe() returned w_t = {w_t:.6f}, expected w_t ∈ [0, 1]. "
                "Check that your OQE scoring is properly normalised."
            )

    def _check_invariant_4_calibration_consistency(self, z_t: np.ndarray, q_t: float, tol: float) -> None:
        """Supporting invariant 4: the width shrinkage from w_t=1 to w_t=0.5 must be ≈ 2·q_t.

        Theoretical requirement: the inflation rule m_t = q_t / (w_t + ε) gives:
            m_t(w=1)   ≈ q_t       → each side tightened by q_t
            m_t(w=0.5) ≈ 2·q_t    → each side tightened by 2·q_t
        So width(w=1) − width(w=0.5) ≈ 2·(2·q_t − q_t) · 1_dim = 2·q_t per dim.

        We verify this *relative* change (domain-agnostic: does not require knowing
        the nominal action space bounds). This catches adapters that use a flat
        inflation independent of w_t (e.g. always 10·q_t regardless of w).
        """
        set_w1 = self.adapter.uncertainty_set(z_t, w_t=1.0, q_t=q_t)
        if set_w1.is_empty:
            raise ContractViolation(
                "Invariant 4 (calibration consistency) FAILED: "
                "uncertainty_set(w=1) is empty. At full reliability, at least "
                "the nominal conformal set should be non-empty."
            )
        set_w05 = self.adapter.uncertainty_set(z_t, w_t=0.5, q_t=q_t)

        width_w1 = float(np.max(set_w1.upper - set_w1.lower)) if not set_w1.is_empty else 0.0
        width_w05 = float(np.max(set_w05.upper - set_w05.lower)) if not set_w05.is_empty else 0.0

        # Width must decrease as reliability drops (monotonicity sub-check).
        if width_w05 > width_w1 + tol:
            raise ContractViolation(
                f"Invariant 4 (calibration consistency) FAILED: "
                f"set width at w=0.5 ({width_w05:.5f}) > width at w=1 ({width_w1:.5f}). "
                "Lower reliability must produce a narrower (more conservative) action set."
            )

        # The width reduction from w=1 to w=0.5 should ≈ 2·q_t
        # (each side gains one more q_t of margin when w halves).
        expected_shrinkage = 2.0 * q_t
        actual_shrinkage = width_w1 - width_w05
        rel_tol = 0.15  # 15% tolerance: accounts for ε stabiliser and box approximations
        if abs(actual_shrinkage - expected_shrinkage) > tol + rel_tol * expected_shrinkage:
            raise ContractViolation(
                f"Invariant 4 (calibration consistency) FAILED: "
                f"width shrinkage from w=1→w=0.5 is {actual_shrinkage:.5f}, "
                f"expected ≈ 2·q_t = {expected_shrinkage:.5f} "
                f"(relative error {abs(actual_shrinkage - expected_shrinkage) / (expected_shrinkage + 1e-12):.1%}). "
                "Check that your inflation rule uses m_t = q_t / (w_t + ε)."
            )

    def _check_invariant_5_zero_reliability_empty(self, z_t: np.ndarray, q_t: float) -> None:
        """Supporting invariant 5: at w_t = 0 (total observation failure), the tightened
        set must be empty (is_empty = True).

        Theoretical requirement: when the observation is completely unreliable,
        no action can be certified as safe — only the fallback is admissible.
        This corresponds to the limit m_t → ∞ as w_t → 0, which shrinks the
        tightened set to the empty set (or the single fallback action).
        """
        zero_rel_set = self.adapter.uncertainty_set(z_t, w_t=0.0, q_t=q_t)
        if not zero_rel_set.is_empty:
            raise ContractViolation(
                "Invariant 5 (zero-reliability fallback) FAILED: "
                "uncertainty_set(w=0) is not empty. At zero reliability, the "
                "inflation m_t = q_t / (0 + ε) is very large, and the tightened "
                "set should collapse to empty. Only the fallback action is safe."
            )

    def _check_invariant_2_tightened_subset(self, z_t: np.ndarray, q_t: float, tol: float) -> None:
        """Supporting invariant 2: the tightened set A_t must be a subset of the nominal
        set A_nominal (= uncertainty_set at w_t = 1, the widest permissible set).

        Theoretical requirement: tightening is monotone in w_t — lower reliability
        produces a *smaller* (more conservative) action set, never a larger one.
        This is supporting evidence for the active T2/T11 proof pattern.
        """
        nominal = self.adapter.uncertainty_set(z_t, w_t=1.0, q_t=q_t)
        tightened = self.adapter.uncertainty_set(z_t, w_t=0.5, q_t=q_t)
        if tightened.is_empty:
            return  # empty set ⊆ everything — invariant trivially satisfied
        if np.any(tightened.lower < nominal.lower - tol):
            raise ContractViolation(
                f"Invariant 2 (subset) FAILED: tightened lower bound "
                f"{tightened.lower} < nominal lower {nominal.lower} by more than tol={tol}. "
                "Tightening must not expand the action set below the nominal floor."
            )
        if np.any(tightened.upper > nominal.upper + tol):
            raise ContractViolation(
                f"Invariant 2 (subset) FAILED: tightened upper bound "
                f"{tightened.upper} > nominal upper {nominal.upper} by more than tol={tol}. "
                "Tightening must not expand the action set above the nominal ceiling."
            )

    def _check_invariant_1_repair_membership(self, z_t: np.ndarray, q_t: float, tol: float) -> None:
        """Supporting invariant 1: the repaired action must lie in the tightened set.

        This is the core supporting invariant: it mirrors the repair-membership
        obligation used in the active one-step transfer proof.

        We test it by constructing a candidate action that is *deliberately*
        outside the tightened set (above the upper bound) and checking that
        repair() returns an action that is back inside.
        """
        safe_set = self.adapter.uncertainty_set(z_t, w_t=0.5, q_t=q_t)
        if safe_set.is_empty:
            # Cannot test repair against an empty set; fallback is the only action.
            # Check that fallback() exists and returns something.
            fb = self.adapter.fallback()
            if fb is None or (hasattr(fb, "__len__") and len(fb) == 0):
                raise ContractViolation(
                    "Invariant 1 (repair membership) FAILED: safe_set is empty "
                    "and fallback() returned None or empty array."
                )
            return

        # Construct a candidate that is above the upper bound in every dimension.
        outside_candidate = safe_set.upper + 1.0

        result = self.adapter.repair(outside_candidate, safe_set)
        if result is None or result.action is None:
            raise ContractViolation("Invariant 1 (repair membership) FAILED: repair() returned None.")
        if np.any(result.action < safe_set.lower - tol):
            raise ContractViolation(
                f"Invariant 1 (repair membership) FAILED: repaired action "
                f"{result.action} < lower bound {safe_set.lower} "
                f"(violation = {safe_set.lower - result.action})."
            )
        if np.any(result.action > safe_set.upper + tol):
            raise ContractViolation(
                f"Invariant 1 (repair membership) FAILED: repaired action "
                f"{result.action} > upper bound {safe_set.upper} "
                f"(violation = {result.action - safe_set.upper})."
            )


# ── Exception ──────────────────────────────────────────────────────────────────


class ContractViolation(Exception):
    """Raised when a DomainAdapter fails one of the five supporting invariant checks.

    The exception message describes which invariant failed, the observed values,
    and a diagnostic hint. Catching this exception programmatically is intentional
    in test suites; in production the invariant failure indicates a bug in the
    adapter implementation that must be fixed before deployment.
    """

    pass
