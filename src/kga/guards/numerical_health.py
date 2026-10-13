"""kga.guards.numerical_health -- Numerical health and degeneracy guards for model outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class HealthAssessment:
    """Result of numerical health inspection."""
    is_healthy: bool
    mean_entropy: float
    max_prob: float
    min_prob: float
    rejection_reason: Optional[str] = None


class NumericalHealthGuard:
    """Inspects model outputs (probabilities or logits) for numerical degradation or collapse.
    
    Prevents deploying models experiencing NaN explosions, probability vanishing,
    or degenerate entropy collapse during test-time adaptation.
    """

    def __init__(
        self,
        min_entropy: float = 0.01,
        max_entropy_fraction: float = 0.999,
        min_prob_threshold: float = 0.0,
        max_prob_threshold: float = 1.0,
    ) -> None:
        self.min_entropy = float(min_entropy)
        self.max_entropy_fraction = float(max_entropy_fraction)
        self.min_prob_threshold = float(min_prob_threshold)
        self.max_prob_threshold = float(max_prob_threshold)

    def check_probabilities(self, probs: np.ndarray) -> HealthAssessment:
        """Inspect a batch of predicted probabilities (shape: [N, num_classes] or [N])."""
        arr = np.asarray(probs, dtype=float)
        
        if arr.size == 0:
            return HealthAssessment(
                is_healthy=False,
                mean_entropy=0.0,
                max_prob=0.0,
                min_prob=0.0,
                rejection_reason="Empty prediction array",
            )

        if not np.all(np.isfinite(arr)):
            return HealthAssessment(
                is_healthy=False,
                mean_entropy=0.0,
                max_prob=0.0,
                min_prob=0.0,
                rejection_reason="Predictions contain NaN or Infinite values",
            )

        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        if min_val < -1e-6 or max_val > 1.0 + 1e-6:
            return HealthAssessment(
                is_healthy=False,
                mean_entropy=0.0,
                max_prob=max_val,
                min_prob=min_val,
                rejection_reason=f"Probabilities out of valid [0, 1] range: [{min_val}, {max_val}]",
            )

        # Multiclass entropy evaluation
        if arr.ndim == 2 and arr.shape[1] > 1:
            # Clip for safe log
            clipped = np.clip(arr, 1e-12, 1.0)
            # Row-wise Shannon entropy in nats
            entropies = -np.sum(clipped * np.log(clipped), axis=1)
            mean_ent = float(np.mean(entropies))
            max_possible_ent = float(np.log(arr.shape[1]))

            if mean_ent < self.min_entropy:
                return HealthAssessment(
                    is_healthy=False,
                    mean_entropy=mean_ent,
                    max_prob=max_val,
                    min_prob=min_val,
                    rejection_reason=f"Prediction entropy {mean_ent:.4f} collapsed below threshold {self.min_entropy:.4f}",
                )

            if mean_ent > self.max_entropy_fraction * max_possible_ent:
                return HealthAssessment(
                    is_healthy=False,
                    mean_entropy=mean_ent,
                    max_prob=max_val,
                    min_prob=min_val,
                    rejection_reason=f"Prediction entropy {mean_ent:.4f} near uniform maximum {max_possible_ent:.4f} (lost discriminative capability)",
                )
        else:
            mean_ent = 1.0

        return HealthAssessment(
            is_healthy=True,
            mean_entropy=mean_ent,
            max_prob=max_val,
            min_prob=min_val,
            rejection_reason=None,
        )

    def check_weight_divergence(
        self,
        weights_base: np.ndarray,
        weights_adapted: np.ndarray,
        max_relative_divergence: float = 0.5,
    ) -> Tuple[bool, float, Optional[str]]:
        """Check relative parameter weight divergence ||theta_a - theta_0|| / ||theta_0||."""
        w0 = np.asarray(weights_base, dtype=float).ravel()
        wa = np.asarray(weights_adapted, dtype=float).ravel()

        if w0.shape != wa.shape:
            return False, float("inf"), f"Weight shape mismatch: {w0.shape} vs {wa.shape}"

        norm0 = float(np.linalg.norm(w0))
        if norm0 < 1e-12:
            norm0 = 1.0

        diff_norm = float(np.linalg.norm(wa - w0))
        rel_diff = diff_norm / norm0

        if not np.isfinite(rel_diff):
            return False, float("inf"), "Weight divergence is NaN or Infinite"

        if rel_diff > max_relative_divergence:
            return (
                False,
                rel_diff,
                f"Relative weight divergence {rel_diff:.4f} exceeds safety threshold {max_relative_divergence:.4f}",
            )

        return True, rel_diff, None
