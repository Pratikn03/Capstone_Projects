"""Channel capacity and mutual information estimation for ORIUS.

Provides:
- KSG (Kraskov-Stögbauer-Grassberger) k-NN mutual information estimator
- FaultChannelModel: composable erasure+delay+noise channel model
- Blahut-Arimoto iterative rate-distortion solver
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma


def ksg_mutual_information(
    X: np.ndarray,
    Y: np.ndarray,
    k: int = 5,
) -> dict[str, Any]:
    r"""KSG k-nearest-neighbor mutual information estimator (Algorithm 1 of KSG 2004).

    Estimates I(X; Y) from paired samples without density estimation.
    Uses the Chebyshev (L-infinity) norm for neighbor distances and
    the digamma-based bias correction from Kraskov et al.

    Parameters
    ----------
    X : array of shape (n, d_x) or (n,)
    Y : array of shape (n, d_y) or (n,)
    k : number of nearest neighbors (default 5)

    Returns
    -------
    dict with keys: I_XY, k, n_samples
    """
    x = np.asarray(X, dtype=float)
    y = np.asarray(Y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    n = x.shape[0]
    if n < k + 1:
        raise ValueError(f"Need at least k+1={k+1} samples, got {n}.")

    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dists[:, -1]

    nx = np.array([
        tree_x.query_ball_point(x[i], r=eps[i] - 1e-15, p=np.inf).__len__() - 1
        for i in range(n)
    ])
    ny = np.array([
        tree_y.query_ball_point(y[i], r=eps[i] - 1e-15, p=np.inf).__len__() - 1
        for i in range(n)
    ])

    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)

    mi = float(
        digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    )

    return {
        "I_XY": max(0.0, mi),
        "k": k,
        "n_samples": n,
    }


@dataclass(frozen=True)
class FaultChannelModel:
    """Composable fault channel: erasure + delay + additive Gaussian noise.

    Channel capacity (bits per use):
        C = (1 - erasure_prob) * 0.5 * log2(1 + SNR) * delay_attenuation

    where SNR = signal_power / noise_std^2 and delay_attenuation = 1 / (1 + delay_steps).
    """

    erasure_prob: float = 0.0
    delay_steps: int = 0
    noise_std: float = 0.0
    signal_power: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.erasure_prob <= 1.0):
            raise ValueError("erasure_prob must lie in [0, 1].")
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be non-negative.")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")
        if self.signal_power <= 0.0:
            raise ValueError("signal_power must be positive.")

    def capacity(self) -> float:
        """Shannon capacity of this fault channel (bits per use)."""
        if self.erasure_prob >= 1.0:
            return 0.0
        snr = self.signal_power / max(self.noise_std ** 2, 1e-15) if self.noise_std > 0 else float("inf")
        gaussian_cap = 0.5 * math.log2(1.0 + snr) if snr < float("inf") else float("inf")
        delay_attenuation = 1.0 / (1.0 + self.delay_steps)
        raw = (1.0 - self.erasure_prob) * gaussian_cap * delay_attenuation
        return float(min(raw, 1e6))

    def compose(self, other: FaultChannelModel) -> FaultChannelModel:
        """Serial composition of two fault channels (data processing inequality)."""
        combined_erasure = 1.0 - (1.0 - self.erasure_prob) * (1.0 - other.erasure_prob)
        combined_delay = self.delay_steps + other.delay_steps
        combined_noise_var = self.noise_std ** 2 + other.noise_std ** 2
        return FaultChannelModel(
            erasure_prob=min(combined_erasure, 1.0),
            delay_steps=combined_delay,
            noise_std=math.sqrt(combined_noise_var),
            signal_power=min(self.signal_power, other.signal_power),
        )


def blahut_arimoto(
    p_yx: np.ndarray,
    distortion: np.ndarray,
    beta: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> dict[str, Any]:
    r"""Blahut-Arimoto algorithm for rate-distortion computation.

    Given a source distribution implicit in p_yx (conditional distribution
    P(Y|X)), a distortion matrix d(x, a), and Lagrange multiplier beta,
    iteratively finds the optimal reproduction distribution P(A|Y) that
    minimizes R + beta * D.

    Parameters
    ----------
    p_yx : array (|Y|, |X|), joint or conditional distribution P(Y|X)
           (columns normalized to sum to 1 or will be normalized)
    distortion : array (|X|, |A|), distortion matrix d(x, a)
    beta : Lagrange multiplier (higher = lower distortion, higher rate)
    max_iter : maximum iterations
    tol : convergence tolerance on rate change

    Returns
    -------
    dict with: rate, distortion, optimal_policy (|Y|, |A|), converged, iterations
    """
    p_yx = np.asarray(p_yx, dtype=float)
    d = np.asarray(distortion, dtype=float)

    if p_yx.ndim != 2 or d.ndim != 2:
        raise ValueError("p_yx and distortion must be 2-D arrays.")
    n_y, n_x = p_yx.shape
    n_x2, n_a = d.shape
    if n_x != n_x2:
        raise ValueError("p_yx columns and distortion rows must match (|X|).")

    col_sums = p_yx.sum(axis=0, keepdims=True)
    col_sums = np.maximum(col_sums, 1e-15)
    p_yx = p_yx / col_sums
    p_x = np.ones(n_x, dtype=float) / n_x

    q_a = np.ones(n_a, dtype=float) / n_a

    prev_rate = float("inf")
    converged = False

    for iteration in range(max_iter):
        log_q = np.log(np.maximum(q_a, 1e-15))
        exponent = log_q[np.newaxis, :] - beta * d
        exponent -= exponent.max(axis=1, keepdims=True)
        p_ax = np.exp(exponent)
        p_ax /= p_ax.sum(axis=1, keepdims=True)

        p_ya = p_yx @ (p_ax * p_x[:, np.newaxis])
        q_a_new = p_ya.sum(axis=0)
        q_a_new = np.maximum(q_a_new, 1e-15)
        q_a_new /= q_a_new.sum()

        avg_dist = float(np.sum(p_x[:, np.newaxis] * p_ax * d))

        kl_terms = p_ax * (np.log(np.maximum(p_ax, 1e-15)) - np.log(np.maximum(q_a_new[np.newaxis, :], 1e-15)))
        rate = float(np.sum(p_x[:, np.newaxis] * kl_terms))

        q_a = q_a_new

        if abs(rate - prev_rate) < tol:
            converged = True
            break
        prev_rate = rate

    p_ya_final = p_yx @ (p_ax * p_x[:, np.newaxis])
    row_sums = p_ya_final.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-15)
    policy = p_ya_final / row_sums

    return {
        "rate": max(0.0, rate),
        "distortion": max(0.0, avg_dist),
        "optimal_policy": policy,
        "converged": converged,
        "iterations": iteration + 1,
    }


__all__ = [
    "ksg_mutual_information",
    "FaultChannelModel",
    "blahut_arimoto",
]
