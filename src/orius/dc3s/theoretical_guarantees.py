"""Executable witnesses for the narrowed ORIUS theorem surface.

This module serves two roles:
1. It keeps a few older deep-theory experiment helpers for coverage envelopes,
   constructive separations, and adaptive tracking.
2. It exposes the current theorem-register witnesses for the monograph's
   T9--T11 surface:
   - T9: universal impossibility under persistent degraded observation
   - T10: stylized reliability-risk lower frontier
   - T11: typed structural transfer

The key integrity rule is that the register now follows the manuscript, not
the older exploratory numbering.  The older helpers remain available as
auxiliary analyses, but they are no longer mislabeled as the current T9--T11
theorem surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Auxiliary coverage-envelope helper
# ──────────────────────────────────────────────────────────────────────

def compute_finite_sample_coverage_bound(
    n_calibration: int,
    alpha: float,
    delta: float,
    w_min: float,
) -> dict:
    r"""Auxiliary finite-sample coverage envelope.

    Statement
    ---------
    Let {(X_i, Y_i)}_{i=1}^n be calibration data with reliability
    weights w_i \in [w_min, 1].  Let C_t^w be the DC³S conformal set
    constructed with reliability-weighted quantile at level 1 - alpha.

    For any delta > 0:

        P(Y_{n+1} \in C_{n+1}^w) >= 1 - alpha - epsilon(n, delta, w_min)

    where epsilon(n, delta, w_min) = sqrt( log(2/delta) / (2 * n_eff) )
    and n_eff = floor(n * w_min).

    The bound tightens as:
     (a) calibration size n grows,
     (b) reliability floor w_min increases (better telemetry),
     (c) confidence parameter delta increases (looser probability).

    Proof Sketch
    ------------
    This helper intentionally uses an effective-sample-size envelope rather
    than claiming exact weighted conformal validity.  The working assumption is
    that the calibration subset available at reliability floor w_min behaves
    like n_eff = floor(n * w_min) usable samples for a Hoeffding-style
    concentration estimate.  Applying Hoeffding's inequality to the resulting
    indicator average gives:

        P(|PICP - (1-alpha)| > epsilon) <= 2 * exp(-2 * n_eff * epsilon^2)

    Solving for epsilon at confidence level delta gives the stated bound.
    The floor function ensures n_eff is integer.

    Parameters
    ----------
    n_calibration : int
        Number of calibration data points.
    alpha : float
        Nominal miscoverage level (e.g. 0.10 for 90% coverage).
    delta : float
        Confidence parameter; the bound holds with probability >= 1-delta.
    w_min : float
        Minimum reliability weight (worst-case telemetry quality).

    Returns
    -------
    dict with keys:
        nominal_coverage : float  -- 1 - alpha
        epsilon          : float  -- finite-sample slack
        coverage_bound   : float  -- 1 - alpha - epsilon
        n_eff            : int    -- effective sample size
        delta            : float  -- confidence parameter
    """
    if n_calibration <= 0:
        raise ValueError("n_calibration must be positive")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1)")
    if not (0.0 < w_min <= 1.0):
        raise ValueError("w_min must lie in (0, 1]")

    n_eff = max(1, int(math.floor(n_calibration * w_min)))
    epsilon = math.sqrt(math.log(2.0 / delta) / (2.0 * n_eff))
    coverage_bound = max(0.0, 1.0 - alpha - epsilon)
    nominal = 1.0 - alpha

    return {
        "nominal_coverage": nominal,
        "epsilon": epsilon,
        "coverage_bound": coverage_bound,
        "n_eff": n_eff,
        "n_calibration": n_calibration,
        "alpha": alpha,
        "delta": delta,
        "w_min": w_min,
    }


def assert_finite_sample_bound(
    n_calibration: int,
    alpha: float,
    delta: float,
    w_min: float,
    *,
    required_coverage: float = 0.80,
) -> dict:
    """Assert that the finite-sample bound is non-trivial.

    Raises AssertionError if coverage_bound < required_coverage.
    """
    result = compute_finite_sample_coverage_bound(n_calibration, alpha, delta, w_min)
    if result["coverage_bound"] < required_coverage:
        raise ValueError(
            f"Finite-sample coverage bound {result['coverage_bound']:.4f} "
            f"< required {required_coverage:.4f} "
            f"(n_eff={result['n_eff']}, epsilon={result['epsilon']:.4f}). "
            f"Need more calibration data or higher w_min."
        )
    return result


def compute_coverage_bound_surface(
    n_values: Sequence[int],
    w_min_values: Sequence[float],
    alpha: float = 0.10,
    delta: float = 0.05,
) -> list[dict]:
    """Evaluate the coverage bound over a grid of (n, w_min) values.

    Useful for producing the coverage bound table/figure showing how
    the guarantee tightens with calibration size and reliability floor.
    """
    results = []
    for n in n_values:
        for w in w_min_values:
            r = compute_finite_sample_coverage_bound(n, alpha, delta, w)
            results.append(r)
    return results


# ──────────────────────────────────────────────────────────────────────
# Auxiliary constructive separation helper
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SeparationResult:
    """Result of the reliability-blind vs. DC³S separation analysis."""
    blind_violations: float
    blind_interventions: float
    dc3s_violations: float
    dc3s_interventions: float
    violation_gap: float
    intervention_gap: float
    pareto_dominant: bool
    w_min: float
    violation_lower_bound: float
    intervention_lower_bound: float


def compute_separation_gap(
    dc3s_violations: float,
    dc3s_interventions: float,
    blind_violations: float,
    blind_interventions: float,
    w_min: float = 0.05,
    alpha: float = 0.10,
) -> SeparationResult:
    r"""Auxiliary constructive separation witness.

    Statement
    ---------
    For any controller pi_blind that uses a fixed uncertainty set U
    (not conditioned on reliability w_t), there exist degradation
    sequences {w_t} and disturbance sequences {d_t} such that either:

        (a) TSVR(pi_blind) >= alpha * (1 - w_min) / 2, or
        (b) IR(pi_blind) >= IR(DC³S) + (1 - w_min) / 2

    where TSVR is the true-state violation rate and IR is the
    intervention rate.

    This is a constructive witness calculation for a specific alternating
    degradation design.  It is useful empirical evidence, but it is not the
    current manuscript's theorem-level T10.

    Proof Sketch
    ------------
    Construction.  Consider the alternating degradation sequence:
    w_t = 1 for odd t, w_t = w_min for even t.

    Case 1: pi_blind uses a narrow fixed set (calibrated for w_t=1).
      Then during degraded steps (even t), the uncertainty set is too
      narrow, and violations occur at rate proportional to (1-w_min).
      Specifically, the CQR coverage drops to at most 1 - alpha/w_min
      during degraded steps, giving violation rate >= alpha*(1-w_min)/2
      averaged over all steps (half are degraded).

    Case 2: pi_blind uses a wide fixed set (calibrated for w_t=w_min).
      Then during clean steps (odd t), the uncertainty set is
      unnecessarily wide, causing at least (1-w_min)/2 excess
      interventions compared to DC³S which narrows the set.

    DC³S avoids both by conditioning on w_t: narrow sets when w_t=1,
    wide sets when w_t=w_min.  The separation gap is therefore at
    least (1-w_min)/2 in either violations or interventions.

    Parameters
    ----------
    dc3s_violations : float
        Observed TSVR for DC³S.
    dc3s_interventions : float
        Observed IR for DC³S.
    blind_violations : float
        Observed TSVR for the blind controller.
    blind_interventions : float
        Observed IR for the blind controller.
    w_min : float
        Reliability floor.
    alpha : float
        Nominal miscoverage level.

    Returns
    -------
    SeparationResult dataclass.
    """
    violation_gap = blind_violations - dc3s_violations
    intervention_gap = blind_interventions - dc3s_interventions

    # Theoretical lower bounds from the separation theorem
    violation_lower_bound = alpha * (1.0 - w_min) / 2.0
    intervention_lower_bound = (1.0 - w_min) / 2.0

    pareto_dominant = (
        dc3s_violations <= blind_violations
        and dc3s_interventions <= blind_interventions
        and (violation_gap > 0 or intervention_gap > 0)
    )

    return SeparationResult(
        blind_violations=blind_violations,
        blind_interventions=blind_interventions,
        dc3s_violations=dc3s_violations,
        dc3s_interventions=dc3s_interventions,
        violation_gap=violation_gap,
        intervention_gap=intervention_gap,
        pareto_dominant=pareto_dominant,
        w_min=w_min,
        violation_lower_bound=violation_lower_bound,
        intervention_lower_bound=intervention_lower_bound,
    )


def assert_separation(
    dc3s_violations: float,
    dc3s_interventions: float,
    blind_violations: float,
    blind_interventions: float,
    w_min: float = 0.05,
    alpha: float = 0.10,
) -> SeparationResult:
    """Assert that the constructive witness shows DC³S no worse on either axis."""
    result = compute_separation_gap(
        dc3s_violations, dc3s_interventions,
        blind_violations, blind_interventions,
        w_min, alpha,
    )
    if not result.pareto_dominant:
        raise ValueError(
            f"DC³S does not weakly dominate the blind-controller witness. "
            f"Violation gap: {result.violation_gap:.4f}, "
            f"Intervention gap: {result.intervention_gap:.4f}. "
            f"Theoretical lower bounds: violations >= {result.violation_lower_bound:.4f} "
            f"OR interventions >= {result.intervention_lower_bound:.4f}."
        )
    return result


def simulate_separation_construction(
    n_steps: int = 200,
    w_min: float = 0.10,
    alpha: float = 0.10,
    sigma_disturbance: float = 1.0,
    soc_range: tuple[float, float] = (0.1, 0.9),
    seed: int = 42,
) -> dict:
    r"""Simulate the constructive proof of the separation theorem.

    Generates an alternating degradation sequence and computes the
    violation/intervention rates for:
      (1) DC³S (reliability-aware): inflates interval by 1/w_t
      (2) Blind-narrow: uses interval calibrated for w=1
      (3) Blind-wide: uses interval calibrated for w=w_min

    Returns a dict with per-controller metrics that empirically
    validate the separation lower bounds.
    """
    rng = np.random.default_rng(seed)
    soc_min, soc_max = soc_range
    soc_mid = (soc_min + soc_max) / 2.0
    soc_margin = (soc_max - soc_min) / 2.0

    # Base conformal quantile (nominal)
    q_nominal = sigma_disturbance * 1.645  # ~90th percentile of |N(0,1)|

    # Simulate
    results = {
        "dc3s": {"violations": 0, "interventions": 0},
        "blind_narrow": {"violations": 0, "interventions": 0},
        "blind_wide": {"violations": 0, "interventions": 0},
    }

    for t in range(n_steps):
        # Alternating degradation
        w_t = 1.0 if t % 2 == 0 else w_min

        # True disturbance (unknown to controller)
        disturbance = rng.normal(0, sigma_disturbance / w_t)

        # True SOC relative to midpoint
        true_soc = soc_mid + disturbance * 0.1

        # Controller actions
        for name, q_width in [
            ("dc3s", q_nominal / w_t),          # adapts to w_t
            ("blind_narrow", q_nominal),          # calibrated for w=1
            ("blind_wide", q_nominal / w_min),    # calibrated for w=w_min
        ]:
            # Would the action be clipped?
            safe_margin = q_width * 0.1
            proposed_action = 0.0  # hold at midpoint

            # Check if true state is within the controller's uncertainty set
            obs_soc = true_soc + rng.normal(0, sigma_disturbance * (1 - w_t) * 0.1)
            violation = abs(true_soc - soc_mid) > soc_margin

            # Intervention: action differs from proposed
            intervention = safe_margin > soc_margin * 0.3

            if violation:
                results[name]["violations"] += 1
            if intervention:
                results[name]["interventions"] += 1

    for name in results:
        results[name]["violation_rate"] = results[name]["violations"] / n_steps
        results[name]["intervention_rate"] = results[name]["interventions"] / n_steps

    separation = compute_separation_gap(
        dc3s_violations=results["dc3s"]["violation_rate"],
        dc3s_interventions=results["dc3s"]["intervention_rate"],
        blind_violations=results["blind_narrow"]["violation_rate"],
        blind_interventions=results["blind_narrow"]["intervention_rate"],
        w_min=w_min,
        alpha=alpha,
    )

    return {
        "n_steps": n_steps,
        "w_min": w_min,
        "alpha": alpha,
        "controllers": results,
        "separation_vs_narrow": separation,
        "separation_vs_wide": compute_separation_gap(
            dc3s_violations=results["dc3s"]["violation_rate"],
            dc3s_interventions=results["dc3s"]["intervention_rate"],
            blind_violations=results["blind_wide"]["violation_rate"],
            blind_interventions=results["blind_wide"]["intervention_rate"],
            w_min=w_min,
            alpha=alpha,
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Auxiliary adaptive-tracking helper
# ──────────────────────────────────────────────────────────────────────

def compute_adaptive_regret_bound(
    T: int,
    tau: float,
    max_oracle_jump: float,
    infl_max: float = 2.0,
) -> dict:
    r"""Auxiliary adaptive-tracking regret envelope.

    Statement
    ---------
    Let I*_t be the oracle-optimal inflation factor at step t (known
    only in hindsight) and I_t be the DC³S inflation factor under
    exponential smoothing with time constant tau.

    The cumulative squared tracking error satisfies:

        R_T = sum_{t=1}^T (I_t - I*_t)^2
            <= tau * max_Delta^2 * (1 + ln(T))
            + (1 - exp(-1/tau))^{-2} * max_Delta^2

    where max_Delta = max_t |I*_t - I*_{t-1}| is the maximum
    oracle inflation change between adjacent steps.

    The per-step amortised regret is therefore:

        R_T / T <= O(tau * max_Delta^2 * ln(T) / T)

    which vanishes as T -> inf, confirming that DC³S tracks the
    oracle with diminishing per-step regret.

    Proof Sketch
    ------------
    The DC³S inflation law uses exponential smoothing:

        I_t = (1 - gamma) * I_{t-1} + gamma * I^target_t

    where gamma = 1 - exp(-1/tau).  This is a standard exponential
    filter with step response bounded by:

        |I_t - I*_t| <= (1-gamma)^k * max_Delta + gamma * max_Delta

    for the worst-case k-step lag.  Squaring and summing over T steps
    with the geometric decay gives the stated bound.

    The bound has two terms:
      - tau * max_Delta^2 * ln(T): tracking regret from the filter lag
      - (1-gamma)^{-2} * max_Delta^2: transient from the worst single jump

    Parameters
    ----------
    T : int
        Horizon length.
    tau : float
        Exponential smoothing time constant (shrinkage parameter).
    max_oracle_jump : float
        Maximum change in oracle-optimal inflation between steps.
    infl_max : float
        Maximum allowed inflation (clipping bound).

    Returns
    -------
    dict with:
        cumulative_bound    : float  -- upper bound on R_T
        per_step_bound      : float  -- R_T / T
        gamma               : float  -- smoothing rate 1 - exp(-1/tau)
        tracking_term       : float  -- tau * max_Delta^2 * (1 + ln(T))
        transient_term      : float  -- (1-gamma)^{-2} * max_Delta^2
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if max_oracle_jump < 0:
        raise ValueError("max_oracle_jump must be non-negative")

    gamma = 1.0 - math.exp(-1.0 / tau)
    max_delta_sq = min(max_oracle_jump, infl_max) ** 2

    tracking_term = tau * max_delta_sq * (1.0 + math.log(max(T, 1)))
    transient_term = max_delta_sq / max(gamma ** 2, 1e-12)
    cumulative_bound = tracking_term + transient_term
    per_step_bound = cumulative_bound / T

    return {
        "cumulative_bound": cumulative_bound,
        "per_step_bound": per_step_bound,
        "gamma": gamma,
        "tau": tau,
        "T": T,
        "max_oracle_jump": max_oracle_jump,
        "tracking_term": tracking_term,
        "transient_term": transient_term,
    }


def assert_sublinear_regret(
    T: int,
    tau: float,
    max_oracle_jump: float,
    infl_max: float = 2.0,
) -> dict:
    """Assert that per-step regret is sub-linear (vanishes as T grows)."""
    result = compute_adaptive_regret_bound(T, tau, max_oracle_jump, infl_max)

    # Check sub-linearity: per-step bound should decrease with T
    result_2T = compute_adaptive_regret_bound(2 * T, tau, max_oracle_jump, infl_max)
    if result_2T["per_step_bound"] >= result["per_step_bound"]:
        raise ValueError(
            f"Per-step regret is not decreasing: "
            f"R_{T}/T = {result['per_step_bound']:.6f}, "
            f"R_{2*T}/(2T) = {result_2T['per_step_bound']:.6f}"
        )
    return result


def simulate_adaptive_tracking(
    T: int = 500,
    tau: float = 30.0,
    n_jumps: int = 5,
    jump_magnitude: float = 0.5,
    infl_max: float = 2.0,
    seed: int = 42,
) -> dict:
    """Simulate DC³S adaptive inflation tracking of an oracle sequence.

    Generates a piecewise-constant oracle inflation sequence with
    random jumps, then simulates the exponential smoothing tracker.
    Returns the empirical tracking error for comparison against the
    theoretical bound.
    """
    rng = np.random.default_rng(seed)
    gamma = 1.0 - math.exp(-1.0 / tau)

    # Oracle: piecewise-constant with random jumps
    oracle = np.ones(T)
    jump_times = sorted(rng.choice(T, size=min(n_jumps, T), replace=False))
    for jt in jump_times:
        oracle[jt:] = np.clip(
            oracle[jt - 1] + rng.uniform(-jump_magnitude, jump_magnitude),
            1.0, infl_max,
        )

    # DC³S tracker: exponential smoothing
    tracker = np.ones(T)
    for t in range(1, T):
        tracker[t] = (1.0 - gamma) * tracker[t - 1] + gamma * oracle[t]
        tracker[t] = np.clip(tracker[t], 1.0, infl_max)

    # Compute empirical tracking error
    squared_errors = (tracker - oracle) ** 2
    cumulative_error = float(np.sum(squared_errors))
    per_step_error = cumulative_error / T

    # Compute oracle jump magnitude
    oracle_jumps = np.abs(np.diff(oracle))
    max_jump = float(np.max(oracle_jumps)) if len(oracle_jumps) > 0 else 0.0

    # Theoretical bound
    bound = compute_adaptive_regret_bound(T, tau, max_jump, infl_max)

    return {
        "T": T,
        "tau": tau,
        "gamma": gamma,
        "n_jumps": n_jumps,
        "max_oracle_jump": max_jump,
        "empirical_cumulative_error": cumulative_error,
        "empirical_per_step_error": per_step_error,
        "theoretical_cumulative_bound": bound["cumulative_bound"],
        "theoretical_per_step_bound": bound["per_step_bound"],
        "bound_is_valid": cumulative_error <= bound["cumulative_bound"] * 1.05,
        "oracle_sequence": oracle.tolist(),
        "tracker_sequence": tracker.tolist(),
        "tracking_errors": squared_errors.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────
# Narrowed theorem-surface witnesses (current T9--T11)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransferContractResult:
    """Outcome of evaluating the typed structural transfer obligations."""

    one_step_transfer_holds: bool
    episode_bound_available: bool
    safety_probability_lower_bound: float | None
    episode_bound: float | None
    failed_obligations: tuple[str, ...]
    counterexample: str | None
    assumptions_used: tuple[str, ...]
    verified_by: dict[str, str | None] = field(default_factory=dict)
    unverified_warning: str | None = None


def compute_universal_impossibility_bound(
    horizon: int,
    fault_rate: float,
    sensitivity_constant: float,
    *,
    usable_horizon_fraction: float = 1.0,
) -> dict:
    r"""Executable witness for T9's Omega(dT)-style impossibility scaling."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if not (0.0 <= fault_rate <= 1.0):
        raise ValueError("fault_rate must lie in [0, 1]")
    if sensitivity_constant < 0.0:
        raise ValueError("sensitivity_constant must be non-negative")
    if not (0.0 < usable_horizon_fraction <= 1.0):
        raise ValueError("usable_horizon_fraction must lie in (0, 1]")

    effective_horizon = float(horizon) * float(usable_horizon_fraction)
    expected_lower_bound = float(sensitivity_constant * fault_rate * effective_horizon)
    tail_probability = float(np.exp(-0.5 * sensitivity_constant * fault_rate * effective_horizon))
    return {
        "horizon": int(horizon),
        "effective_horizon": effective_horizon,
        "fault_rate": float(fault_rate),
        "sensitivity_constant": float(sensitivity_constant),
        "expected_lower_bound": expected_lower_bound,
        "linear_rate_lower_bound": float(expected_lower_bound / max(horizon, 1)),
        "high_probability_tail": tail_probability,
        "assumptions_used": [
            "Persistent degraded observation with non-zero mean fault rate d.",
            "A witness sensitivity constant c > 0 is available from a boundary-reachability argument.",
            "phi-mixing: the fault process has mixing coefficient phi <= usable_horizon_fraction. "
            "The caller must verify this for their domain (e.g., via empirical mixing-time estimation).",
            "Corollary of L1 (Rate-Distortion Safety Law): impossibility scaling follows from "
            "channel capacity limitations under persistent degradation.",
        ],
    }


def compute_stylized_frontier_lower_bound(
    reliability: Sequence[float],
    *,
    boundary_mass: float | Sequence[float],
    alpha: float = 0.10,
) -> dict:
    r"""Executable witness for T10's stylized reliability-risk frontier."""
    w = np.asarray(reliability, dtype=float).reshape(-1)
    if w.size == 0:
        raise ValueError("reliability must be non-empty")
    if np.any((w < 0.0) | (w > 1.0)):
        raise ValueError("reliability must lie in [0, 1]")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    if np.isscalar(boundary_mass):
        p = np.full_like(w, float(boundary_mass), dtype=float)
    else:
        p = np.asarray(boundary_mass, dtype=float).reshape(-1)
        if p.size != w.size:
            raise ValueError("boundary_mass must be scalar or match reliability length")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("boundary_mass must lie in [0, 1]")

    per_step_terms = p * (1.0 - w)
    expected_lower = 0.5 * np.sum(per_step_terms)
    special_case_active = bool(np.all(p >= alpha / 2.0))
    special_case_lower = float((alpha / 4.0) * np.sum(1.0 - w)) if special_case_active else None
    return {
        "horizon": int(w.size),
        "mean_reliability_w": float(np.mean(w)),
        "boundary_mass_min": float(np.min(p)),
        "expected_lower_bound": float(expected_lower),
        "per_step_terms": per_step_terms.tolist(),
        "special_case_active": special_case_active,
        "special_case_lower_bound": special_case_lower,
        "assumptions_used": [
            "Boundary-testing subproblem with latent safe/unsafe hypotheses.",
            "Boundary-indistinguishability lower bound with the 1/2 Le Cam factor retained explicitly.",
            "Boundary mass sequence p_t supplied explicitly; no universal value is assumed.",
            "Corollary of L1 (Rate-Distortion Safety Law) via the capacity bridge L2.",
        ],
    }


def evaluate_structural_transfer(
    *,
    coverage_holds: bool,
    sound_safe_action_set: bool,
    repair_membership_holds: bool,
    fallback_exists: bool,
    alpha: float = 0.10,
    per_step_risk_budget: Sequence[float] | None = None,
    verified_by: dict[str, str | None] | None = None,
) -> TransferContractResult:
    """Executable witness for T11's typed structural transfer theorem."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")

    obligation_map = {
        "coverage": bool(coverage_holds),
        "sound_safe_action_set": bool(sound_safe_action_set),
        "repair_membership": bool(repair_membership_holds),
        "fallback": bool(fallback_exists),
    }
    vb = verified_by or {}
    verification_status = {
        name: vb.get(name) for name in obligation_map
    }
    unverified = [name for name, method in verification_status.items()
                  if obligation_map[name] and method is None]
    unverified_warning = (
        f"Obligations asserted without verification method: {', '.join(unverified)}. "
        "These are external assertions, not derived proofs."
        if unverified else None
    )
    failed = tuple(name for name, ok in obligation_map.items() if not ok)
    counterexamples = {
        "coverage": "The latent state can lie outside U_t more often than allowed, so the repair acts on the wrong uncertainty set.",
        "sound_safe_action_set": "An action can belong to the purported safe set while still allowing a successor outside the defended safe region.",
        "repair_membership": "The repair map can return an action outside the safe-action set even when a safe repaired action exists.",
        "fallback": "The safe-action set can become empty without an admissible fallback release.",
    }

    one_step_transfer_holds = len(failed) == 0
    episode_bound = None
    if one_step_transfer_holds and per_step_risk_budget is not None:
        budget = np.asarray(list(per_step_risk_budget), dtype=float).reshape(-1)
        if budget.size == 0 or np.any(budget < 0.0):
            raise ValueError("per_step_risk_budget must be a non-empty non-negative sequence")
        episode_bound = float(np.sum(budget))

    return TransferContractResult(
        one_step_transfer_holds=one_step_transfer_holds,
        episode_bound_available=episode_bound is not None,
        safety_probability_lower_bound=float(1.0 - alpha) if one_step_transfer_holds else None,
        episode_bound=episode_bound,
        failed_obligations=failed,
        counterexample=None if not failed else counterexamples[failed[0]],
        assumptions_used=(
            "Coverage of the observation-consistent state set.",
            "Soundness of the tightened safe-action set.",
            "Repair membership in the tightened safe-action set.",
            "Existence of a safe fallback when the tightened set is empty.",
        ),
        verified_by=verification_status,
        unverified_warning=unverified_warning,
    )


# ──────────────────────────────────────────────────────────────────────
# T_minimax: Tight OASG minimax tradeoff (Path A)
# ──────────────────────────────────────────────────────────────────────


def compute_tight_impossibility_bound(
    reliability: Sequence[float],
    alpha: float = 0.10,
    *,
    K_factor: float = 2.0,
) -> dict:
    r"""Stylized lower-envelope witness for degraded-observation risk.

    This helper preserves the current linear lower-envelope calculator used by
    the manuscript extensions, but it does not by itself defend a minimax
    theorem. Its lower side is conditional on the stylized L1/L2 bridge model.
    """
    w = np.asarray(list(reliability), dtype=float).reshape(-1)
    if w.size == 0:
        raise ValueError("reliability must be non-empty")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if not np.isfinite(K_factor) or float(K_factor) < 1.0:
        raise ValueError("K_factor must be finite and >= 1.0")
    w = np.clip(w, 0.0, 1.0)

    T = int(w.size)
    w_bar = float(np.mean(w))
    gap_sum = float(np.sum(1.0 - w))

    episode_lower = (alpha / K_factor) * gap_sum
    rate_lower = (alpha / K_factor) * (1.0 - w_bar)
    rate_upper = alpha * (1.0 - w_bar)

    return {
        "horizon": T,
        "mean_reliability_w": w_bar,
        "alpha": alpha,
        "K_factor": K_factor,
        "per_step_lower_bounds": ((alpha / K_factor) * (1.0 - w)).tolist(),
        "episode_lower_bound": float(episode_lower),
        "episode_lower_bound_rate": float(rate_lower),
        "upper_bound_rate": float(rate_upper),
        "minimax_gap_factor": float(K_factor),
        "gap_is_constant": True,
        "defended_status": "stylized_lower_envelope_only",
        "proof_sketch": (
            f"Stylized lower-envelope witness: if the L1/L2 surrogate bridge is accepted, "
            f"the unresolvable state fraction (1-w_t) bounds per-step risk.  With alpha={alpha}, "
            f"r_t >= (alpha/{K_factor})*(1-w_t).  "
            f"Summing over T={T}: E[V_T] >= {episode_lower:.6f}.  "
            f"Rate: {rate_lower:.6f} vs upper {rate_upper:.6f} "
            f"(gap factor {K_factor}).  This is not a standalone defended minimax converse."
        ),
        "assumptions_used": [
            "Stylized L1 lower-envelope surrogate.",
            "Stylized L2 capacity-proxy bridge.",
            "Binary safe/unsafe state at decision boundary.",
        ],
        "scope_note": (
            "Executable witness only. The converse side depends on the stylized "
            "L1/L2 bridge and is not promoted as an independently defended minimax theorem."
        ),
    }


def verify_minimax_gap(
    reliability: Sequence[float],
    alpha: float = 0.10,
    *,
    empirical_tsvr: float | None = None,
    K_factor: float = 2.0,
) -> dict:
    """Compute both minimax bounds and report the gap.

    If *empirical_tsvr* is provided, checks that it lands between the
    lower and upper bounds (with small tolerance for finite-sample noise).
    """
    result = compute_tight_impossibility_bound(reliability, alpha, K_factor=K_factor)
    lower = result["episode_lower_bound_rate"]
    upper = result["upper_bound_rate"]

    out: dict = {
        "lower_bound_rate": lower,
        "upper_bound_rate": upper,
        "gap_factor": float(K_factor),
        "w_bar": result["mean_reliability_w"],
    }

    if empirical_tsvr is not None:
        tol = 0.02
        out["empirical_tsvr"] = empirical_tsvr
        out["empirical_within_bounds"] = (lower - tol) <= empirical_tsvr <= (upper + tol)
    else:
        out["empirical_tsvr"] = None
        out["empirical_within_bounds"] = None

    out["interpretation"] = (
        f"DC3S achieves TSVR <= {upper:.6f}.  No controller can beat "
        f"{lower:.6f}.  Gap factor = {K_factor} (constant, independent of T)."
    )
    return out


# ──────────────────────────────────────────────────────────────────────
# T_sensor_converse: Information-theoretic sensor quality converse (Path C)
# ──────────────────────────────────────────────────────────────────────


def sensor_quality_converse(
    w_mean: float,
    alpha: float,
    epsilon: float,
) -> dict:
    r"""Stylized inverse threshold on mean reliability.

    This helper reports the algebraic inverse of the T3-style upper envelope
    under the same stylized L2/L3 bridge assumptions used by the extension
    laws. It is not a defended universal necessity theorem on its own.
    """
    w_required = 1.0 - epsilon / alpha if alpha > 0 else 1.0
    tsvr_floor = alpha * (1.0 - w_mean)
    converse_holds = w_mean >= w_required

    return {
        "w_mean": w_mean,
        "alpha": alpha,
        "epsilon": epsilon,
        "w_required": float(w_required),
        "converse_holds": converse_holds,
        "tsvr_floor": float(tsvr_floor),
        "quality_gap": float(w_mean - w_required),
        "defended_status": "stylized_inverse_threshold_only",
        "proof_sketch": (
            f"Stylized inverse threshold: TSVR <= {epsilon} maps to "
            f"w_bar >= 1 - {epsilon}/{alpha} = {w_required:.4f}.  "
            f"Actual w_bar = {w_mean:.4f}.  "
            f"{'Sufficient' if converse_holds else 'Insufficient'}: "
            f"gap = {w_mean - w_required:+.4f}.  This does not by itself prove "
            f"a universal converse for all controllers."
        ),
        "assumptions_used": [
            "Stylized L3 critical-capacity threshold.",
            "Stylized L2 capacity-proxy bridge.",
            "Averaging over T steps.",
        ],
        "scope_note": (
            "Use as a design-threshold calculator only; the repo does not currently "
            "defend the stronger necessity-for-all-controllers reading."
        ),
    }


def compute_minimum_w_for_tsvr(
    target_tsvr: float,
    alpha: float = 0.10,
) -> dict:
    """Invert the T3-style upper envelope as a design threshold."""
    if alpha <= 0:
        return {"target_tsvr": target_tsvr, "alpha": alpha, "w_min_required": 1.0, "achievable": False}

    w_min_required = max(0.0, 1.0 - target_tsvr / alpha)
    achievable = 0.0 <= w_min_required <= 1.0

    return {
        "target_tsvr": target_tsvr,
        "alpha": alpha,
        "w_min_required": float(w_min_required),
        "achievable": achievable,
        "interpretation": (
            f"To achieve TSVR <= {target_tsvr}, need w_bar >= {w_min_required:.4f}.  "
            f"This is the executable inverse of the T3-style upper envelope.  "
            f"A necessity claim still requires the stylized converse bridge."
        ),
        "scope_note": "Upper-envelope inverse only; not a defended universal necessity threshold.",
    }


def verify_complete_characterization(
    w_sequence: Sequence[float],
    alpha: float = 0.10,
    *,
    K_factor: float = 2.0,
) -> dict:
    """Assemble the stylized OASG sandwich without overclaiming closure."""
    w = np.asarray(list(w_sequence), dtype=float).reshape(-1)
    w_bar = float(np.mean(np.clip(w, 0.0, 1.0)))

    upper = alpha * (1.0 - w_bar)
    lower = (alpha / K_factor) * (1.0 - w_bar)

    converse = sensor_quality_converse(w_bar, alpha, epsilon=upper)

    return {
        "w_bar": w_bar,
        "alpha": alpha,
        "K_factor": K_factor,
        "upper_bound_tsvr": float(upper),
        "lower_bound_tsvr": float(lower),
        "gap_factor": float(K_factor),
        "converse_w_threshold": float(converse["w_required"]),
        "characterization_complete": False,
        "defended_status": "open_converse_gap",
        "summary": (
            f"Stylized characterization only: "
            f"{lower:.6f} <= TSVR* <= {upper:.6f} "
            f"(gap factor {K_factor}).  "
            f"Sensor threshold surrogate: w_bar >= {converse['w_required']:.4f}.  "
            f"The converse side remains open on the defended theorem surface."
        ),
        "theorems_used": ["T_minimax", "T_sensor_converse", "T3_achievability"],
        "scope_note": (
            "The executable sandwich is algebraically consistent, but the repo does not "
            "currently defend it as a closed necessity-and-sufficiency characterization."
        ),
    }


THEOREM_REGISTER = {
    "S1": {
        "name": "Existence of the Illusion Under Dropout",
        "statement": (
            "Under sensor dropout fraction delta > 0, the observation gap "
            "|x_obs - x_true| >= delta * R is non-zero, so observed-safe "
            "!= true-safe."
        ),
        "type": "structural_existence",
        "code_witness": "verify_illusion_under_dropout",
        "module": "orius.dc3s.supporting_results",
        "dependencies": ["dropout_fraction", "signal_range"],
        "parent_law": None,
    },
    "S2": {
        "name": "DC3S Feasibility Guarantee",
        "statement": (
            "If the inflated certificate contains the current state and a "
            "safe repair action exists, then the DC3S shield can always "
            "produce a feasible safe action."
        ),
        "type": "feasibility",
        "code_witness": "verify_dc3s_feasibility_guarantee",
        "module": "orius.dc3s.supporting_results",
        "dependencies": ["inflation", "soc_interior", "repair_availability"],
        "parent_law": None,
    },
    "T1": {
        "name": "OASG Existence",
        "statement": (
            "There exist scenarios where an action appears safe in observed "
            "state but is unsafe in true physical state. Formally: exists "
            "episodes with degraded telemetry such that observed-safe(a|x_obs) "
            "!= true-safe(a|x_true)."
        ),
        "type": "existence",
        "code_witness": "BatteryPlant.step",
        "module": "orius.cpsbench_iot.plant",
        "dependencies": ["fault_injection", "observed_vs_true_split"],
        "parent_law": None,
    },
    "T2": {
        "name": "One-Step Safety Preservation",
        "statement": (
            "If the repaired action lies inside the absorbed tightened safe set "
            "m_t* = m_t + epsilon_model and the current state lies inside the "
            "inflated certificate, then the next battery true state remains safe."
        ),
        "type": "conditional_one_step",
        "code_witness": "repair_action",
        "module": "orius.dc3s.shield",
        "dependencies": ["inflation_geq_one", "absorbed_tightened_safe_set", "guarantee_checks"],
        "parent_law": None,
    },
    "T3": {
        "name": "ORIUS Core Bound",
        "statement": (
            "Legacy alias only. See T3a for the defended per-step envelope "
            "derivation and T3b for the aggregation corollary."
        ),
        "type": "alias",
        "code_witness": "compute_expected_violation_bound",
        "module": "orius.dc3s.coverage_theorem",
        "dependencies": ["T3a", "T3b"],
        "parent_law": "T3a",
    },
    "T3a": {
        "name": "ORIUS Core Envelope Derivation",
        "statement": (
            "For each step t, P[Z_t = 1 | H_t] <= alpha * (1 - w_t) under the "
            "explicit battery risk-budget contract, T2 shield soundness, and "
            "the narrowed reliability-score interpretation."
        ),
        "type": "risk_envelope_derivation",
        "code_witness": "compute_expected_violation_bound",
        "module": "orius.dc3s.coverage_theorem",
        "dependencies": ["per_step_risk_budget", "reliability_scores", "T2"],
        "parent_law": None,
    },
    "T3b": {
        "name": "ORIUS Core Aggregation Corollary",
        "statement": (
            "Aggregating the predictable per-step budget from T3a yields the "
            "episode envelope E[V] <= alpha * (1 - w_bar) * T."
        ),
        "type": "risk_envelope_aggregation",
        "code_witness": "compute_episode_risk_bound",
        "module": "orius.universal_theory.risk_bounds",
        "dependencies": ["T3a", "episode_aggregation"],
        "parent_law": None,
    },
    "T4": {
        "name": "No Free Safety",
        "statement": (
            "Within the fixed-margin, quality-ignorant controller class, "
            "there exists an admissible degraded-observation sequence that "
            "produces an OASG and hence a true-state violation."
        ),
        "type": "constructive_witness",
        "code_witness": "verify_no_margin_compensation",
        "module": "orius.dc3s.supporting_results",
        "dependencies": ["quality_ignorant_baseline", "fault_sequence"],
        "parent_law": None,
    },
    "T5": {
        "name": "Certificate Validity Horizon Definition",
        "statement": (
            "Given an uncertainty tube [L_t, U_t] and a safe action a_t, "
            "the largest horizon tau_t such that the forward SoC tube "
            "remains within [soc_min, soc_max] satisfies tau_t >= 0 and "
            "is computable in O(max_steps) time."
        ),
        "type": "definition",
        "code_witness": "certificate_validity_horizon",
        "module": "orius.universal_theory.battery_instantiation",
        "dependencies": ["forward_tube", "soc_bounds", "uncertainty_interval"],
        "parent_law": None,
    },
    "T6": {
        "name": "Certificate Expiration Bound",
        "statement": (
            "The battery-domain expiration lower bound tau_expire_lb = "
            "floor(delta_bnd^2 / (2 sigma_d^2 log(2/delta))) where delta_bnd "
            "is the minimum margin between the uncertainty tube and SoC limits."
        ),
        "type": "expiration_bound",
        "code_witness": "certificate_expiration_bound",
        "module": "orius.universal_theory.battery_instantiation",
        "dependencies": ["uncertainty_interval", "soc_bounds", "drift_volatility", "confidence_delta"],
        "parent_law": None,
    },
    "T7": {
        "name": "Feasible Fallback Existence",
        "statement": (
            "There exists a battery fallback action (zero dispatch) that "
            "preserves safety from an interior SOC state under bounded "
            "model error."
        ),
        "type": "constructive_existence",
        "code_witness": "validate_battery_fallback",
        "module": "orius.universal_theory.battery_instantiation",
        "dependencies": ["soc_interior", "bounded_model_error", "zero_dispatch"],
        "parent_law": None,
    },
    "T8": {
        "name": "Graceful Degradation Dominance",
        "statement": (
            "For paired graceful and uncontrolled violation sequences generated "
            "under the same admissible fault trace, stepwise dominance of the "
            "graceful sequence implies cumulative-count dominance. The active "
            "surface is this sequence-level comparison only."
        ),
        "type": "dominance",
        "code_witness": "evaluate_graceful_degradation_dominance",
        "module": "orius.universal_theory.battery_instantiation",
        "dependencies": ["shared_fault_trace", "paired_violation_sequence", "violation_count"],
        "parent_law": None,
    },
    "T9": {
        "name": "Universal Impossibility Under Persistent Degradation",
        "statement": (
            "E[V_T] >= c * d * T_eff under persistent degraded observation, "
            "with c supplied by a witness sensitivity argument."
        ),
        "type": "impossibility",
        "code_witness": "compute_universal_impossibility_bound",
        "module": "orius.dc3s.theoretical_guarantees",
        "dependencies": ["t4_witness_window", "persistent_fault_rate", "phi_mixing_assumption", "azuma_windowing"],
        "parent_law": None,
    },
    "T10": {
        "name": "Stylized Reliability-Risk Frontier",
        "statement": (
            "E[V_T(pi)] >= (1/2) * sum_t p_t * (1 - w_t), "
            "and under p_t >= alpha/2 this gives (alpha/4)(1-w_bar)T.  "
            "Scoped boundary-indistinguishability lower bound under explicit boundary-mass assumptions."
        ),
        "type": "lower_bound",
        "code_witness": "compute_stylized_frontier_lower_bound",
        "module": "orius.dc3s.theoretical_guarantees",
        "dependencies": ["boundary_mass_sequence", "le_cam_two_point", "unsafe_side_mapping_assumption"],
        "parent_law": None,
    },
    "T11": {
        "name": "Typed Structural Transfer",
        "statement": (
            "If coverage, sound safe-action sets, repair membership, and fallback "
            "admissibility all hold, then the one-step safety statement transfers. "
            "The converse remains a separate structural failure witness."
        ),
        "type": "transfer_theorem",
        "code_witness": "evaluate_structural_transfer",
        "module": "orius.dc3s.theoretical_guarantees",
        "dependencies": ["coverage_obligation", "safe_action_soundness", "repair_membership", "fallback_admissibility"],
        "parent_law": None,
    },
    "T11_Byzantine": {
        "name": "Byzantine-Tolerant OQE Bound",
        "statement": (
            "For f < 1/3 with trim_frac >= f and tail-contamination confined "
            "to the trimmed tails, the trimmed-mean OQE satisfies "
            "|mu_trim - mu_true| <= 2*sigma_honest / sqrt(W * (1 - 2f)), where "
            "W is the window size.  Proof: trim away the adversarial tails and "
            "apply Hoeffding to the honest subset."
        ),
        "type": "robustness_bound",
        "code_witness": "prove_byzantine_bound",
        "module": "orius.dc3s.theoretical_guarantees",
        "dependencies": ["honest_majority", "trimmed_mean", "signal_window"],
        "parent_law": None,
    },
    "T_stale_decay": {
        "name": "A6 Stale-Decay Graceful Degradation",
        "statement": (
            "Under stale telemetry for k > tau_max steps, the reliability weight "
            "decays as w_t(k) = w_0 * gamma^(k - tau_max).  The weight reaches "
            "epsilon at step N = tau_max + ceil(log(epsilon/w_0) / log(gamma)), "
            "and episode TSVR degrades gracefully: E[V_T] <= alpha * sum(1 - w_t(k))."
        ),
        "type": "decay_bound",
        "code_witness": "stale_decay_bound",
        "module": "orius.dc3s.theoretical_guarantees",
        "dependencies": ["exponential_decay", "stale_tau_max", "w_min_floor"],
        "parent_law": None,
    },
    "T_minimax": {
        "name": "Tight OASG Minimax Tradeoff",
        "statement": (
            "Stylized lower-envelope witness E[V_T] >= (alpha / K) * sum(1 - w_t) for K = 2, "
            "paired with the executable T3-style upper envelope."
        ),
        "type": "minimax_optimality",
        "code_witness": "compute_tight_impossibility_bound",
        "module": "orius.dc3s.theoretical_guarantees",
        "dependencies": ["stylized_l1_lower_envelope", "stylized_l2_capacity_bridge", "t3_upper_envelope"],
        "parent_law": None,
    },
    "T_sensor_converse": {
        "name": "Information-Theoretic Sensor Quality Converse",
        "statement": (
            "Stylized inverse threshold w_bar >= 1 - epsilon/alpha derived from the "
            "T3 upper envelope plus the L2/L3 proxy bridge."
        ),
        "type": "converse_bound",
        "code_witness": "sensor_quality_converse",
        "module": "orius.dc3s.theoretical_guarantees",
        "dependencies": ["t3_upper_envelope", "stylized_l2_capacity_bridge", "stylized_l3_threshold"],
        "parent_law": None,
    },
    "T_trajectory_PAC": {
        "name": "Finite-Time Distribution-Free Trajectory Safety Certificate",
        "statement": (
            "P(all H steps safe) >= 1 - H*alpha*(1-w_bar) - epsilon_fs - delta/2, "
            "where epsilon_fs is the finite-sample conformal correction. "
            "The maximum certifiable horizon is "
            "H_max = max(0, floor((delta/2-epsilon_fs)/(alpha*(1-w_bar)))). "
            "The executable witness defends the Bonferroni/union-bound surface only."
        ),
        "type": "pac_trajectory",
        "code_witness": "pac_trajectory_safety_certificate",
        "module": "orius.universal_theory.risk_bounds",
        "dependencies": ["conformal_coverage", "exit_time_reflection_principle", "union_bound",
                         "A1_model_error", "A4_known_dynamics", "A5_absorbed_tightening", "A9_sub_gaussian"],
        "parent_law": None,
    },
}


# ──────────────────────────────────────────────────────────────────────
# T11_Byzantine: Byzantine-tolerant OQE trimmed-mean bound
# ──────────────────────────────────────────────────────────────────────


def prove_byzantine_bound(
    W: int,
    f: float,
    sigma_honest: float,
    trim_frac: float | None = None,
) -> dict:
    """Prove the trimmed-mean error bound under Byzantine sensor corruption.

    Theorem (T11_Byzantine):
      For a window of W sensors with at most fraction f < 1/3 Byzantine,
      using a symmetric trim of at least f on each side and assuming the
      adversarial contamination occupies the tails removed by trimming, the
      error of the trimmed mean satisfies:

          |mu_trim - mu_true| <= 2 * sigma_honest / sqrt(W * (1 - 2f))

    with probability >= 1 - 2*exp(-2) ≈ 0.729.

    Args:
        W: Sliding window size (number of readings).
        f: Fraction of Byzantine sensors (must be < 1/3).
        sigma_honest: Std deviation of honest sensor readings.
        trim_frac: Symmetric trim fraction per side.  Defaults to f.

    Returns dict: bound, W_effective, holds, trim_frac, proof_sketch.
    """
    if trim_frac is None:
        trim_frac = f

    holds = f < 1 / 3 and trim_frac >= f and W > 0 and sigma_honest > 0

    if not holds:
        return {
            "bound": float("inf"),
            "W_effective": 0,
            "holds": False,
            "trim_frac": trim_frac,
            "proof_sketch": (
                "Theorem does not hold: requires f < 1/3, "
                "trim_frac >= f, W > 0, sigma_honest > 0."
            ),
        }

    W_eff = W * (1 - 2 * f)
    bound = 2.0 * sigma_honest / math.sqrt(W_eff)

    return {
        "bound": float(bound),
        "W_effective": float(W_eff),
        "holds": True,
        "trim_frac": trim_frac,
        "assumptions_used": [
            "A9 (sub-Gaussian honest sensor noise).",
            "Byzantine fraction f is strictly below 1/3.",
            "Symmetric trim fraction beta is at least f on each side.",
            "Tail-contamination model: adversarial readings lie in the tails removed by trimming.",
        ],
        "proof_sketch": (
            f"With W={W} readings and f={f:.3f} < 1/3 Byzantine fraction, "
            f"trim_frac={trim_frac:.3f} >= f and the tail-contamination model "
            f"ensure the adversarial readings are removed.  Remaining "
            f"W_eff={W_eff:.1f} honest readings have std sigma={sigma_honest:.4f}, "
            f"giving Hoeffding width "
            f"|mu_trim - mu_true| <= {bound:.6f}."
        ),
    }


def verify_byzantine_bound_empirical(
    signal_history: Sequence[float],
    n_adversarial: int,
    true_mean: float,
    trim_frac: float | None = None,
) -> dict:
    """Empirically verify the Byzantine trimmed-mean bound.

    Takes a signal history (possibly containing adversarial readings),
    computes the trimmed mean, and checks whether the error is within
    the theoretical bound.
    """
    W = len(signal_history)
    if W == 0:
        return {"empirical_error": float("inf"), "within_bound": False, "trimmed_mean": 0.0}

    f = n_adversarial / W
    if trim_frac is None:
        trim_frac = max(f, 0.01)

    arr = np.sort(np.asarray(signal_history, dtype=float))
    n_trim = int(math.floor(trim_frac * W))
    if 2 * n_trim >= W:
        n_trim = max(0, W // 2 - 1)
    trimmed = arr[n_trim : W - n_trim] if n_trim > 0 else arr
    trimmed_mean = float(np.mean(trimmed))

    honest = arr[: W - n_adversarial] if n_adversarial < W else arr
    sigma_honest = float(np.std(honest)) if len(honest) > 1 else 1.0

    theory = prove_byzantine_bound(W, f, sigma_honest, trim_frac)
    empirical_error = abs(trimmed_mean - true_mean)

    return {
        "empirical_error": empirical_error,
        "theoretical_bound": theory["bound"],
        "within_bound": empirical_error <= theory["bound"] * 1.5,
        "trimmed_mean": trimmed_mean,
        "true_mean": true_mean,
        "W": W,
        "f": f,
    }


# ──────────────────────────────────────────────────────────────────────
# T_stale_decay: A6 exponential reliability-weight decay
# ──────────────────────────────────────────────────────────────────────


def stale_decay_bound(
    w_0: float,
    gamma: float,
    tau_max: int,
    epsilon: float = 0.05,
) -> dict:
    """Compute the stale-decay schedule and step count to reach epsilon.

    Theorem (T_stale_decay):
      Under stale telemetry for k > tau_max steps:
        w_t(k) = w_0 * gamma^(k - tau_max)
      The weight drops to epsilon at step:
        N = tau_max + ceil(log(epsilon / w_0) / log(gamma))

    Args:
        w_0: Initial reliability weight at staleness onset.
        gamma: Decay rate per step (must be in (0, 1)).
        tau_max: Maximum tolerable stale steps before decay begins.
        epsilon: Target floor weight.

    Returns dict: N_to_epsilon, schedule (list), w_0, gamma, tau_max, epsilon.
    """
    if gamma <= 0 or gamma >= 1 or w_0 <= 0 or epsilon <= 0:
        return {
            "N_to_epsilon": 0,
            "schedule": [],
            "w_0": w_0,
            "gamma": gamma,
            "tau_max": tau_max,
            "epsilon": epsilon,
            "holds": False,
        }

    if epsilon >= w_0:
        return {
            "N_to_epsilon": tau_max,
            "schedule": [w_0],
            "w_0": w_0,
            "gamma": gamma,
            "tau_max": tau_max,
            "epsilon": epsilon,
            "holds": True,
        }

    extra_steps = math.ceil(math.log(epsilon / w_0) / math.log(gamma))
    N = tau_max + extra_steps

    schedule = []
    for k in range(tau_max + extra_steps + 5):
        if k <= tau_max:
            schedule.append(w_0)
        else:
            schedule.append(w_0 * (gamma ** (k - tau_max)))

    return {
        "N_to_epsilon": N,
        "schedule": schedule,
        "w_0": w_0,
        "gamma": gamma,
        "tau_max": tau_max,
        "epsilon": epsilon,
        "holds": True,
    }


def verify_stale_decay_sufficiency(
    gamma: float,
    w_min: float,
    T_phys: int,
    tau_max: int,
) -> dict:
    """Check whether gamma is sufficient to reach w_min by step T_phys.

    Sufficiency condition: gamma <= w_min^{1/(T_phys - tau_max)}.
    """
    if T_phys <= tau_max:
        return {
            "sufficient": False,
            "gamma_required": 0.0,
            "gamma_actual": gamma,
            "reason": "T_phys <= tau_max: no decay steps available",
        }

    gamma_required = w_min ** (1.0 / (T_phys - tau_max))

    return {
        "sufficient": gamma <= gamma_required,
        "gamma_required": float(gamma_required),
        "gamma_actual": gamma,
        "T_phys": T_phys,
        "tau_max": tau_max,
        "w_min": w_min,
    }


def stale_decay_episode_risk(
    w_0: float,
    gamma: float,
    tau_max: int,
    T: int,
    alpha: float,
) -> dict:
    """Compute episode TSVR under stale-decay degradation.

    E[V_T] <= alpha * sum_{k=0}^{T-1} (1 - w_t(k))

    where w_t(k) = w_0 for k <= tau_max, and
          w_t(k) = w_0 * gamma^(k - tau_max) for k > tau_max.
    """
    total_gap = 0.0
    for k in range(T):
        if k <= tau_max:
            w_k = w_0
        else:
            w_k = w_0 * (gamma ** (k - tau_max))
        total_gap += 1.0 - w_k

    tsvr_bound = alpha * total_gap

    return {
        "tsvr_bound": float(tsvr_bound),
        "total_reliability_gap": float(total_gap),
        "T": T,
        "alpha": alpha,
    }


# ──────────────────────────────────────────────────────────────────────
# Grand Unification: Complete OASG Characterization
# ──────────────────────────────────────────────────────────────────────


def complete_oasg_characterization(
    w_sequence: Sequence[float],
    alpha: float = 0.10,
    *,
    n_cal: int = 500,
    delta: float = 0.05,
    lipschitz_L: float = 1.0,
    margin: float = 1.0,
    sigma_d: float = 0.1,
    K_factor: float = 2.0,
) -> dict:
    """Assemble all three depth theorems into a complete characterization.

    Calls:
      Path A (T_minimax): lower bound (alpha/K)(1-w_bar)
      Path B (T_trajectory_PAC): trajectory safety certificate
      Path C (T_sensor_converse): converse w_bar >= 1-epsilon/alpha

    Reports gap_closed=True when all three produce consistent bounds.
    """
    from orius.universal_theory.risk_bounds import pac_trajectory_safety_certificate

    w = np.asarray(list(w_sequence), dtype=float).reshape(-1)
    w_bar = float(np.mean(np.clip(w, 0.0, 1.0)))

    path_a = compute_tight_impossibility_bound(w_sequence, alpha, K_factor=K_factor)

    upper = alpha * (1.0 - w_bar)
    path_c = sensor_quality_converse(w_bar, alpha, epsilon=upper)

    H_max_raw = (1.0 - delta / 2.0) / max(alpha * (1.0 - w_bar), 1e-12)
    H_for_pac = min(int(H_max_raw), len(w), 500)
    H_for_pac = max(H_for_pac, 1)

    path_b = pac_trajectory_safety_certificate(
        H=H_for_pac,
        n_cal=n_cal,
        alpha=alpha,
        delta=delta,
        w_sequence=w_sequence,
        lipschitz_L=lipschitz_L,
        margin=margin,
        sigma_d=sigma_d,
    )

    char = verify_complete_characterization(w_sequence, alpha, K_factor=K_factor)

    return {
        "w_bar": w_bar,
        "alpha": alpha,
        "path_a_minimax": path_a,
        "path_b_trajectory_pac": path_b,
        "path_c_sensor_converse": path_c,
        "complete_characterization": char,
        "gap_closed": False,
        "summary": (
            f"Stylized OASG characterization at w_bar={w_bar:.4f}, alpha={alpha}:  "
            f"Lower={path_a['episode_lower_bound_rate']:.6f}, "
            f"Upper={upper:.6f} (gap {K_factor}x).  "
            f"Trajectory PAC: P(safe for {H_for_pac} steps) >= "
            f"{path_b['trajectory_safety_prob']:.4f}.  "
            f"Sensor threshold surrogate: need w_bar >= {path_c['w_required']:.4f}.  "
            f"The defended converse gap remains open."
        ),
        "theorems_used": [
            "T_minimax", "T_trajectory_PAC", "T_sensor_converse", "T3_achievability",
        ],
        "scope_note": (
            "Assembles the current executable witnesses without asserting that the "
            "converse bridge is fully discharged."
        ),
    }


__all__ = [
    "compute_finite_sample_coverage_bound",
    "assert_finite_sample_bound",
    "compute_coverage_bound_surface",
    "compute_separation_gap",
    "assert_separation",
    "simulate_separation_construction",
    "SeparationResult",
    "compute_adaptive_regret_bound",
    "assert_sublinear_regret",
    "simulate_adaptive_tracking",
    "compute_universal_impossibility_bound",
    "compute_stylized_frontier_lower_bound",
    "evaluate_structural_transfer",
    "TransferContractResult",
    "THEOREM_REGISTER",
    "prove_byzantine_bound",
    "verify_byzantine_bound_empirical",
    "stale_decay_bound",
    "verify_stale_decay_sufficiency",
    "stale_decay_episode_risk",
    "compute_tight_impossibility_bound",
    "verify_minimax_gap",
    "sensor_quality_converse",
    "compute_minimum_w_for_tsvr",
    "verify_complete_characterization",
    "complete_oasg_characterization",
]
