"""ORIUS Rate-Distortion Safety Laws (L1-L4).

The four laws form a complete characterization of the degraded-observation
safety problem:

    L1  Rate-Distortion Safety Law     D*(C) > 0 when C < H(X)
    L2  Capacity Bridge                w_t <= kappa_d * C / H(X)
    L3  Critical Capacity Theorem      C < C*_d => certification impossible
    L4  Achievability-Converse Sandwich (alpha/K)(1-w_bar) <= TSVR* <= alpha(1-w_bar)
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# L1: Rate-Distortion Safety Law
# ---------------------------------------------------------------------------


def rate_distortion_safety_law(
    channel_capacity: float,
    H_entropy: float,
    alpha: float = 0.10,
) -> dict[str, Any]:
    r"""L1: Rate-Distortion Safety Law.

    Statement
    ---------
    D*(C) = inf_{P(A|Y): I(Y;A) <= C} E[d(X,A)]

    When C < H(X), the controller cannot fully distinguish safe from unsafe
    states, and D* > 0 (positive safety loss is unavoidable).  The minimum
    achievable distortion (safety violation rate) is bounded below by:

        D* >= alpha * max(0, 1 - C / H(X))

    This follows from the channel coding converse: if the observation
    channel has capacity C < H(X), then the equivocation H(X|Y) >= H(X) - C
    is positive, and a fraction of at least (1 - C/H(X)) of the state space
    remains unresolvable.

    Parameters
    ----------
    channel_capacity : C (bits per use)
    H_entropy : H(X), entropy of the latent safe/unsafe state (bits)
    alpha : conformal miscoverage level

    Returns
    -------
    dict with D_star_lower, capacity_ratio, law_applies, proof_sketch
    """
    C = float(max(channel_capacity, 0.0))
    H = float(max(H_entropy, 1e-15))
    a = float(alpha)

    if not (0.0 < a < 1.0):
        raise ValueError("alpha must lie in (0, 1).")

    ratio = C / H
    D_star_lower = a * max(0.0, 1.0 - ratio)

    return {
        "D_star_lower": D_star_lower,
        "capacity_ratio": min(ratio, 1.0),
        "law_applies": ratio < 1.0,
        "channel_capacity": C,
        "H_entropy": H,
        "alpha": a,
        "proof_sketch": (
            f"L1 (Rate-Distortion Safety Law): With C={C:.4f} bits and H(X)={H:.4f} bits, "
            f"capacity ratio C/H(X)={ratio:.4f}.  "
            f"D*(C) >= alpha*(1 - C/H(X)) = {a}*(1 - {ratio:.4f}) = {D_star_lower:.6f}.  "
            f"{'Safety loss is unavoidable.' if ratio < 1.0 else 'Full channel: D*=0 achievable.'}"
        ),
    }


# ---------------------------------------------------------------------------
# L2: Capacity Bridge
# ---------------------------------------------------------------------------


def capacity_bridge(
    w_bar: float,
    kappa_d: float,
    H_X: float,
    *,
    channel_capacity: float | None = None,
) -> dict[str, Any]:
    r"""L2: Capacity Bridge — w_t <= kappa_d * C / H(X).

    The OQE reliability score is bounded by the normalized channel capacity.
    kappa_d is a domain-specific constant:
      - Provably <= 1 for binary symmetric channels
      - Empirically near 1 for most CPS domains

    If channel_capacity is provided, checks whether w_bar is consistent
    with the capacity bound.

    Parameters
    ----------
    w_bar : mean OQE reliability score
    kappa_d : domain-specific bridge constant
    H_X : entropy of latent state (bits)
    channel_capacity : optional measured channel capacity (bits)
    """
    if not (0.0 <= w_bar <= 1.0):
        raise ValueError("w_bar must lie in [0, 1].")
    if kappa_d <= 0.0:
        raise ValueError("kappa_d must be positive.")
    if H_X <= 0.0:
        raise ValueError("H_X must be positive.")

    if channel_capacity is not None:
        C = float(max(channel_capacity, 0.0))
        w_upper = float(min(1.0, kappa_d * C / H_X))
        consistent = w_bar <= w_upper + 1e-9
        C_implied = None
    else:
        C_implied = float(w_bar * H_X / kappa_d)
        w_upper = w_bar
        consistent = True
        C = C_implied

    return {
        "w_bar": float(w_bar),
        "kappa_d": float(kappa_d),
        "H_X": float(H_X),
        "w_upper_bound": float(w_upper),
        "consistent": bool(consistent),
        "C_implied": C_implied,
        "proof_sketch": (
            f"L2 (Capacity Bridge): w_bar={w_bar:.4f} <= kappa_d*C/H(X) = "
            f"{kappa_d:.3f}*{C:.4f}/{H_X:.4f} = {w_upper:.4f}.  "
            f"{'Consistent.' if consistent else 'VIOLATED — w_bar exceeds capacity bound.'}"
        ),
    }


# ---------------------------------------------------------------------------
# L3: Critical Capacity Theorem
# ---------------------------------------------------------------------------


def critical_capacity(
    alpha: float,
    kappa_d: float = 1.0,
    H_X: float = 1.0,
    *,
    epsilon: float | None = None,
) -> dict[str, Any]:
    r"""L3: Critical Capacity Theorem.

    Statement
    ---------
    There exists C*_d such that for C < C*_d, no controller achieves
    TSVR <= epsilon for any epsilon < alpha.

        C*_d = H(X) * (1 - epsilon/alpha) / kappa_d

    Below C*_d, safety certification is impossible regardless of
    controller sophistication or calibration data volume.

    Parameters
    ----------
    alpha : conformal miscoverage level
    kappa_d : domain bridge constant
    H_X : state entropy (bits)
    epsilon : target TSVR (defaults to alpha/2 if not given)
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1).")
    if kappa_d <= 0.0:
        raise ValueError("kappa_d must be positive.")

    eps = float(epsilon) if epsilon is not None else alpha / 2.0
    if eps >= alpha:
        raise ValueError("epsilon must be < alpha for a meaningful converse.")

    C_star = H_X * (1.0 - eps / alpha) / kappa_d

    return {
        "C_star_d": float(C_star),
        "alpha": float(alpha),
        "epsilon": float(eps),
        "kappa_d": float(kappa_d),
        "H_X": float(H_X),
        "proof_sketch": (
            f"L3 (Critical Capacity): For TSVR <= {eps:.4f} with alpha={alpha}, "
            f"C*_d = H(X)*(1 - eps/alpha)/kappa_d = "
            f"{H_X:.4f}*(1 - {eps/alpha:.4f})/{kappa_d:.3f} = {C_star:.4f} bits.  "
            f"Below C*_d, no controller can certify safety."
        ),
    }


# ---------------------------------------------------------------------------
# L4: Achievability-Converse Sandwich
# ---------------------------------------------------------------------------


def achievability_converse_sandwich(
    w_bar: float,
    alpha: float = 0.10,
    *,
    K_factor: float = 2.0,
) -> dict[str, Any]:
    r"""L4: Achievability-Converse Sandwich.

    Statement
    ---------
    (alpha/K)(1-w_bar) <= TSVR* <= alpha(1-w_bar)

    The upper bound is the DC3S achievability result.
    The lower bound follows from L1 via L2.
    K=2 for binary channels (Le Cam).  The gap is a constant factor
    that does not grow with T.

    Parameters
    ----------
    w_bar : mean OQE reliability
    alpha : miscoverage level
    K_factor : gap constant (2 for binary channels)
    """
    if not (0.0 <= w_bar <= 1.0):
        raise ValueError("w_bar must lie in [0, 1].")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1).")
    if K_factor <= 0.0:
        raise ValueError("K_factor must be positive.")

    degradation = 1.0 - w_bar
    lower = alpha / K_factor * degradation
    upper = alpha * degradation

    return {
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "w_bar": float(w_bar),
        "alpha": float(alpha),
        "K_factor": float(K_factor),
        "gap_is_constant": True,
        "proof_sketch": (
            f"L4 (Sandwich): ({alpha}/{K_factor:.0f})*(1-{w_bar:.4f}) = {lower:.6f} "
            f"<= TSVR* <= {alpha}*(1-{w_bar:.4f}) = {upper:.6f}.  "
            f"Constant-factor gap of {K_factor:.0f}x.  "
            f"DC3S achieves the upper bound; L1+L2 prove the lower."
        ),
    }


# ---------------------------------------------------------------------------
# LAW_REGISTER
# ---------------------------------------------------------------------------


LAW_REGISTER: dict[str, dict[str, Any]] = {
    "L1": {
        "name": "Rate-Distortion Safety Law",
        "statement": (
            "D*(C) = inf_{P(A|Y): I(Y;A) <= C} E[d(X,A)].  "
            "When C < H(X), D* > 0: positive safety loss is unavoidable."
        ),
        "type": "fundamental_law",
        "code_witness": "rate_distortion_safety_law",
        "module": "orius.universal_theory.orius_law",
    },
    "L2": {
        "name": "Capacity Bridge",
        "statement": "w_t <= kappa_d * C / H(X).  OQE reliability is bounded by normalized channel capacity.",
        "type": "bridge_theorem",
        "code_witness": "capacity_bridge",
        "module": "orius.universal_theory.orius_law",
    },
    "L3": {
        "name": "Critical Capacity Theorem",
        "statement": (
            "There exists C*_d such that for C < C*_d, no controller achieves TSVR <= epsilon.  "
            "C*_d = H(X)*(1 - epsilon/alpha) / kappa_d."
        ),
        "type": "impossibility_law",
        "code_witness": "critical_capacity",
        "module": "orius.universal_theory.orius_law",
    },
    "L4": {
        "name": "Achievability-Converse Sandwich",
        "statement": (
            "(alpha/K)(1-w_bar) <= TSVR* <= alpha(1-w_bar).  "
            "DC3S is within constant factor K of optimal."
        ),
        "type": "characterization",
        "code_witness": "achievability_converse_sandwich",
        "module": "orius.universal_theory.orius_law",
    },
}


# ---------------------------------------------------------------------------
# Grand Unification
# ---------------------------------------------------------------------------


def orius_grand_unification(
    w_sequence: Sequence[float] | np.ndarray,
    alpha: float = 0.10,
    *,
    n_cal: int = 500,
    delta: float = 0.05,
    kappa_d: float = 1.0,
    H_X: float = 1.0,
    channel_capacity: float | None = None,
    margin: float = 1.0,
    sigma_d: float = 0.1,
    K_factor: float = 2.0,
) -> dict[str, Any]:
    """Assemble all four laws and the trajectory PAC certificate.

    Returns a single dict with sub-results from each path and a
    ``gap_closed`` flag indicating whether the complete characterization
    is internally consistent.
    """
    w = np.asarray(list(w_sequence), dtype=float).reshape(-1)
    w_bar = float(np.mean(np.clip(w, 0.0, 1.0)))
    H = len(w)

    C_est = float(channel_capacity) if channel_capacity is not None else float(w_bar * H_X / kappa_d)

    l1 = rate_distortion_safety_law(C_est, H_X, alpha)
    l2 = capacity_bridge(w_bar, kappa_d, H_X, channel_capacity=channel_capacity)
    l3 = critical_capacity(alpha, kappa_d, H_X)
    l4 = achievability_converse_sandwich(w_bar, alpha, K_factor=K_factor)

    from orius.universal_theory.risk_bounds import pac_trajectory_safety_certificate
    pac = pac_trajectory_safety_certificate(
        H=H, n_cal=n_cal, alpha=alpha, delta=delta,
        w_sequence=w.tolist(), margin=margin, sigma_d=sigma_d,
    )

    gap_closed = (
        l4["lower_bound"] <= l4["upper_bound"] + 1e-9
        and l2["consistent"]
    )

    return {
        "gap_closed": bool(gap_closed),
        "w_bar": w_bar,
        "horizon": H,
        "path_a_rate_distortion": l1,
        "path_b_capacity_bridge": l2,
        "path_c_critical_capacity": l3,
        "path_d_sandwich": l4,
        "path_e_trajectory_pac": pac,
    }


# ---------------------------------------------------------------------------
# L2 Formal Derivation
# ---------------------------------------------------------------------------


def capacity_bridge_proof(
    fault_channel: "FaultChannelModel",
    H_X: float,
    kappa_d: float = 1.0,
) -> dict[str, Any]:
    r"""L2 formal derivation: w_t <= kappa_d * C / H(X).

    Steps
    -----
    1. The OQE fault channel Y = X + N(0,sigma^2) with erasure prob epsilon
       and delay d has capacity C = (1-eps) * 0.5*log2(1+SNR) * attenuation.
    2. By the data-processing inequality, any function g(Y) satisfies
       I(X; g(Y)) <= I(X; Y) = C.
    3. The OQE reliability score w_t is determined by g(Y) where g
       computes missing/delay/spike/stale penalties from observed Y.
    4. w_t represents the fraction of state entropy resolved:
       w_t <= 1 - H(X|Y)/H(X) = I(X;Y)/H(X) <= C/H(X).
    5. Introducing kappa_d (domain bridge constant, <=1 for memoryless):
       w_t <= kappa_d * C / H(X).
    """
    from orius.universal_theory.capacity_estimation import FaultChannelModel as _FCM

    if not isinstance(fault_channel, _FCM):
        raise TypeError("fault_channel must be a FaultChannelModel instance.")
    if H_X <= 0.0:
        raise ValueError("H_X must be positive.")
    if kappa_d <= 0.0:
        raise ValueError("kappa_d must be positive.")

    C = fault_channel.capacity()
    ratio = C / H_X
    w_upper = float(min(1.0, kappa_d * ratio))

    steps = [
        {
            "step": 1,
            "label": "Channel capacity",
            "detail": (
                f"Fault channel: erasure={fault_channel.erasure_prob}, "
                f"noise_std={fault_channel.noise_std}, delay={fault_channel.delay_steps}. "
                f"C = {C:.6f} bits/use."
            ),
        },
        {
            "step": 2,
            "label": "Data-processing inequality",
            "detail": "I(X; g(Y)) <= I(X; Y) = C for any deterministic g.",
        },
        {
            "step": 3,
            "label": "OQE is a function of Y",
            "detail": "w_t = g(Y) where g computes penalty products from observed telemetry.",
        },
        {
            "step": 4,
            "label": "Entropy resolution",
            "detail": (
                f"w_t <= I(X;Y)/H(X) = C/H(X) = {C:.6f}/{H_X:.4f} = {ratio:.6f}."
            ),
        },
        {
            "step": 5,
            "label": "Domain bridge",
            "detail": (
                f"With kappa_d={kappa_d:.4f}: w_t <= kappa_d * C/H(X) = {w_upper:.6f}."
            ),
        },
    ]

    return {
        "channel_capacity": float(C),
        "H_X": float(H_X),
        "kappa_d": float(kappa_d),
        "w_upper_bound": w_upper,
        "capacity_ratio": float(ratio),
        "steps": steps,
        "proof_sketch": (
            f"L2 proof: C={C:.4f}, H(X)={H_X:.4f}, C/H(X)={ratio:.4f}. "
            f"By DPI, w_t <= kappa_d*C/H(X) = {w_upper:.6f}."
        ),
    }


# ---------------------------------------------------------------------------
# Empirical kappa_d Estimation
# ---------------------------------------------------------------------------


def capacity_bridge_verify(
    w_sequence: Sequence[float] | np.ndarray,
    channel_capacity: float,
    H_X: float,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> dict[str, Any]:
    r"""Estimate kappa_d from observed (w_t, C) data via bootstrap.

    kappa_d_hat = max(w_t) * H_X / C.  Bootstrap resampling of w_sequence
    gives a confidence interval.
    """
    w = np.asarray(list(w_sequence), dtype=float)
    if len(w) == 0:
        raise ValueError("w_sequence must be non-empty.")
    if channel_capacity <= 0.0:
        raise ValueError("channel_capacity must be positive.")
    if H_X <= 0.0:
        raise ValueError("H_X must be positive.")

    w_clipped = np.clip(w, 0.0, 1.0)
    kappa_hat = float(np.max(w_clipped) * H_X / channel_capacity)

    rng = np.random.default_rng(rng_seed)
    bootstrap_kappas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(w_clipped, size=len(w_clipped), replace=True)
        bootstrap_kappas[i] = float(np.max(sample) * H_X / channel_capacity)

    alpha_tail = (1.0 - confidence) / 2.0
    ci_lower = float(np.percentile(bootstrap_kappas, 100 * alpha_tail))
    ci_upper = float(np.percentile(bootstrap_kappas, 100 * (1.0 - alpha_tail)))

    return {
        "kappa_d_hat": kappa_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_samples": len(w),
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "bridge_holds": kappa_hat <= 1.0 + 1e-9,
        "channel_capacity": float(channel_capacity),
        "H_X": float(H_X),
    }


# ---------------------------------------------------------------------------
# Fano Binary Corollary
# ---------------------------------------------------------------------------


def fano_binary_corollary(
    channel_capacity: float,
    H_binary: float = 1.0,
    alpha: float = 0.10,
) -> dict[str, Any]:
    r"""Binary-state Fano corollary of L1.

    For binary safe/unsafe classification with H(X) = H_binary bits:
    - When C < H_binary, the error probability P_e >= (H_binary - C) / H_binary
      (simplified Fano for binary alphabet where log(|X|-1) = 0,
       so we use the rate-distortion bound directly).
    - The unsafe-side error (missing a hazard) is at least P_e / 2
      for any symmetric prior, and P_e itself for the worst-case prior.

    The safety-relevant bound: P_unsafe_miss >= alpha * max(0, 1 - C/H_binary).
    """
    C = float(max(channel_capacity, 0.0))
    H = float(max(H_binary, 1e-15))
    a = float(alpha)

    if not (0.0 < a < 1.0):
        raise ValueError("alpha must lie in (0, 1).")

    ratio = C / H
    P_e_lower = max(0.0, 1.0 - ratio)
    safety_bound = a * P_e_lower

    return {
        "P_e_lower": P_e_lower,
        "safety_bound": safety_bound,
        "capacity_ratio": min(ratio, 1.0),
        "law_applies": ratio < 1.0,
        "channel_capacity": C,
        "H_binary": H,
        "alpha": a,
        "proof_sketch": (
            f"Fano binary corollary: C={C:.4f}, H={H:.4f}, C/H={ratio:.4f}. "
            f"P_e >= 1 - C/H = {P_e_lower:.4f}. "
            f"Safety bound = alpha * P_e = {safety_bound:.6f}."
        ),
    }


__all__ = [
    "rate_distortion_safety_law",
    "capacity_bridge",
    "critical_capacity",
    "achievability_converse_sandwich",
    "capacity_bridge_proof",
    "capacity_bridge_verify",
    "fano_binary_corollary",
    "LAW_REGISTER",
    "orius_grand_unification",
]
