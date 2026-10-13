"""Gap C — richer label-free evidence features (logits-only, drop-in for kga/evidence.py).

Frozen per PROTOCOL_GAPCLOSE_WAVE5_v1.md. All features consume a logits matrix
L (n x K) from the unlabeled deployment batch; no labels touched.

Features:
  mano_score      MaNo (Xie et al., NeurIPS 2024): softrun normalization of
                  logits, then the L_p entrywise matrix norm (p = 4). Robust
                  linear relation to OOD accuracy; less miscalibration-biased
                  than softmax-based scores.
  nuclear_score   Normalized nuclear norm of the softmax matrix (prediction
                  diversity/rank collapse under shift).
  gdscore_proxy   Logits-only gradient-norm proxy: mean L1 norm of
                  (softmax - onehot(argmax)) — the last-layer CE-gradient
                  magnitude at the pseudo-label (GdScore family).
  entropy_score / msp_score   standard baselines, for the uplift comparison.

ProjNorm (Yu et al. 2022) requires training a probe on pseudo-labels (GPU);
documented in GPU_WIRING.md, NOT implemented here.
"""

from __future__ import annotations

from typing import cast

import numpy as np


def _softmax(L: np.ndarray) -> np.ndarray:
    e = np.exp(L - L.max(axis=1, keepdims=True))
    return cast(np.ndarray, e / e.sum(axis=1, keepdims=True))


def softrun(L: np.ndarray) -> np.ndarray:
    """MaNo's data-dependent normalization.

    Uses the paper's criterion: if the batch is over-confident (low uncertainty,
    softmax would amplify miscalibration), use a Taylor-style surrogate
    (1 + x + x^2/2 on centered logits); otherwise plain softmax. The switch
    statistic is the mean negative log-softmax mass off the argmax.
    """
    P = _softmax(L)
    u = -np.log(np.clip(P.max(axis=1), 1e-12, 1.0)).mean()  # batch uncertainty
    if u < 1.0:  # over-confident regime -> avoid exp amplification
        X = L - L.mean(axis=1, keepdims=True)
        X = X / (np.abs(X).max() + 1e-12)
        Q = 1.0 + X + 0.5 * X * X
        Q = Q / Q.sum(axis=1, keepdims=True)
        return cast(np.ndarray, Q)
    return P


def mano_score(L: np.ndarray, p: int = 4) -> float:
    """MaNo: L_p entrywise norm of the softrun-normalized logit matrix, scaled
    to [0, 1] by the matrix size so batches of different n are comparable."""
    Q = softrun(L)
    n, K = Q.shape
    return float((np.abs(Q) ** p).sum() ** (1.0 / p) / (n * K) ** (1.0 / p))


def nuclear_score(L: np.ndarray) -> float:
    """Normalized nuclear norm of softmax matrix: prediction-space diversity."""
    P = _softmax(L)
    n, K = P.shape
    s = np.linalg.svd(P, compute_uv=False)
    return float(s.sum() / np.sqrt(n * min(n, K)))


def gdscore_proxy(L: np.ndarray) -> float:
    """Mean L1 norm of (softmax - onehot(argmax)): last-layer gradient proxy."""
    P = _softmax(L)
    idx = P.argmax(axis=1)
    G = P.copy()
    G[np.arange(len(P)), idx] -= 1.0
    return float(np.abs(G).sum(axis=1).mean())


def entropy_score(L: np.ndarray) -> float:
    P = _softmax(L)
    return float(-(P * np.log(np.clip(P, 1e-12, 1.0))).sum(axis=1).mean())


def msp_score(L: np.ndarray) -> float:
    return float(_softmax(L).max(axis=1).mean())


def _persample_stats(L: np.ndarray) -> dict:
    """WIN_HUNT_v2 Arm B: per-sample DISTRIBUTIONAL evidence (quantiles, not
    just means) from the same logits — no new forward passes. Rationale: the
    NATURAL_WIN_v1 primary arm was evidence-limited at condition-level means;
    harmful and helpful conditions can share a mean while differing in the
    tails of the per-sample uncertainty distribution."""
    P = _softmax(L)
    ent = -(P * np.log(np.clip(P, 1e-12, 1.0))).sum(axis=1)
    top2 = np.sort(P, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    energy = -np.log(np.exp(L - L.max(axis=1, keepdims=True)).sum(axis=1)) - L.max(axis=1)
    q = lambda x, p: float(np.quantile(x, p))  # noqa: E731
    return {
        "ent_q10": q(ent, 0.1),
        "ent_q50": q(ent, 0.5),
        "ent_q90": q(ent, 0.9),
        "margin_q10": q(margin, 0.1),
        "margin_q50": q(margin, 0.5),
        "margin_q90": q(margin, 0.9),
        "energy_mean": float(energy.mean()),
        "energy_q10": q(energy, 0.1),
    }


BASELINE_FEATURES = {"entropy": entropy_score, "msp": msp_score}
NEW_FEATURES = {"mano": mano_score, "nuclear": nuclear_score, "gdscore": gdscore_proxy}

# canonical feature order (panel_capture serializes in this order)
FEATURE_ORDER = [
    "entropy",
    "msp",
    "mano",
    "nuclear",
    "gdscore",
    "ent_q10",
    "ent_q50",
    "ent_q90",
    "margin_q10",
    "margin_q50",
    "margin_q90",
    "energy_mean",
    "energy_q10",
]


def extract_all(L: np.ndarray) -> dict:
    out = {k: f(L) for k, f in BASELINE_FEATURES.items()}
    out.update({k: f(L) for k, f in NEW_FEATURES.items()})
    out.update(_persample_stats(np.asarray(L, dtype=float)))
    return out
