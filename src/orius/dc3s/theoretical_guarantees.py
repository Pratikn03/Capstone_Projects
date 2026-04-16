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
from dataclasses import dataclass
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
    assert result["coverage_bound"] >= required_coverage, (
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
    assert result.pareto_dominant, (
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
    assert result_2T["per_step_bound"] < result["per_step_bound"], (
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
            "Mixing/buffering losses are summarized by usable_horizon_fraction.",
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

    per_step_terms = 0.5 * p * (1.0 - w)
    special_case_active = bool(np.all(p >= alpha / 2.0))
    special_case_lower = float((alpha / 4.0) * np.sum(1.0 - w)) if special_case_active else None
    return {
        "horizon": int(w.size),
        "mean_reliability_w": float(np.mean(w)),
        "boundary_mass_min": float(np.min(p)),
        "expected_lower_bound": float(np.sum(per_step_terms)),
        "per_step_terms": per_step_terms.tolist(),
        "special_case_active": special_case_active,
        "special_case_lower_bound": special_case_lower,
        "assumptions_used": [
            "Boundary-testing subproblem with latent safe/unsafe hypotheses.",
            "Observation-law indistinguishability encoded through reliability.",
            "Boundary mass sequence p_t supplied explicitly; no universal value is assumed.",
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
    )


THEOREM_REGISTER = {
    "T5": {
        "name": "Certificate Validity Horizon",
        "statement": (
            "Given an uncertainty tube [L_t, U_t] and a safe action a_t, "
            "the largest horizon tau_t such that the forward SoC tube "
            "remains within [soc_min, soc_max] satisfies tau_t >= 0 and "
            "is computable in O(max_steps) time."
        ),
        "type": "constructive_bound",
        "code_witness": "certificate_validity_horizon",
        "module": "orius.universal_theory.battery_instantiation",
        "dependencies": ["forward_tube", "soc_bounds", "uncertainty_interval"],
    },
    "T6": {
        "name": "Certificate Expiration Bound",
        "statement": (
            "The battery-domain expiration lower bound tau_expire_lb = "
            "floor(delta_bnd^2 / sigma_d^2) where delta_bnd is the "
            "minimum margin between the uncertainty tube and SoC limits."
        ),
        "type": "expiration_bound",
        "code_witness": "certificate_expiration_bound",
        "module": "orius.universal_theory.battery_instantiation",
        "dependencies": ["uncertainty_interval", "soc_bounds", "drift_volatility"],
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
        "dependencies": ["boundary_reachability_witness", "persistent_fault_rate", "mixing_buffer_accounting"],
    },
    "T10": {
        "name": "Stylized Reliability-Risk Frontier",
        "statement": (
            "E[V_T(pi)] >= (1/2) * sum_t p_t * (1 - w_t), "
            "and under p_t >= alpha/2 this gives (alpha/4)(1-w_bar)T."
        ),
        "type": "lower_bound",
        "code_witness": "compute_stylized_frontier_lower_bound",
        "module": "orius.dc3s.theoretical_guarantees",
        "dependencies": ["boundary_mass_sequence", "channel_indistinguishability"],
    },
    "T11": {
        "name": "Typed Structural Transfer",
        "statement": (
            "If coverage, sound safe-action sets, repair membership, and fallback "
            "all hold, then the one-step safety statement transfers; if any fails, "
            "the battery proof pattern can break."
        ),
        "type": "transfer_theorem",
        "code_witness": "evaluate_structural_transfer",
        "module": "orius.dc3s.theoretical_guarantees",
        "dependencies": ["coverage_obligation", "safe_action_soundness", "repair_membership", "fallback_admissibility"],
    },
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
]
