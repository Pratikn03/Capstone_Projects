"""kga.guards.evidence_support -- Out-of-distribution guard for label-free evidence vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SupportAssessment:
    """Result of evidence support assessment."""
    is_supported: bool
    mahalanobis_distance: float
    max_zscore: float
    rejection_reason: Optional[str] = None


class EvidenceSupportGuard:
    """Detects when an incoming test evidence vector Z lies outside calibration support.
    
    Prevents uncertified extrapolation by benefit estimator h(Z).
    """

    def __init__(
        self,
        calibration_features: np.ndarray,
        max_mahalanobis_distance: float = 6.0,
        max_coordinate_zscore: float = 5.0,
        epsilon_cov: float = 1e-5,
    ) -> None:
        calib = np.asarray(calibration_features, dtype=float)
        if calib.ndim != 2 or calib.shape[0] < 2:
            raise ValueError(f"calibration_features must be 2D with at least 2 rows, got shape {calib.shape}")
        
        self.dim = calib.shape[1]
        self.mean = np.mean(calib, axis=0)
        self.std = np.std(calib, axis=0)
        self.std[self.std < 1e-8] = 1.0  # Avoid division by zero on constant features

        # Empirical covariance with regularized pseudo-inverse
        cov = np.cov(calib, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        cov += np.eye(self.dim) * epsilon_cov
        self.inv_cov = np.linalg.pinv(cov)

        self.max_mahalanobis_distance = float(max_mahalanobis_distance)
        self.max_coordinate_zscore = float(max_coordinate_zscore)

    def check(self, z: np.ndarray) -> SupportAssessment:
        """Assess whether evidence vector z is inside the calibration support domain."""
        z_arr = np.asarray(z, dtype=float).ravel()
        if z_arr.shape[0] != self.dim:
            return SupportAssessment(
                is_supported=False,
                mahalanobis_distance=float("inf"),
                max_zscore=float("inf"),
                rejection_reason=f"Dimension mismatch: expected {self.dim}, got {z_arr.shape[0]}",
            )

        if not np.all(np.isfinite(z_arr)):
            return SupportAssessment(
                is_supported=False,
                mahalanobis_distance=float("inf"),
                max_zscore=float("inf"),
                rejection_reason="Evidence vector contains NaN or Inf values",
            )

        diff = z_arr - self.mean
        z_scores = np.abs(diff / self.std)
        max_z = float(np.max(z_scores))

        # Mahalanobis distance: sqrt((z - mu)^T Sigma^-1 (z - mu))
        dist_sq = float(diff.T @ self.inv_cov @ diff)
        dist = float(np.sqrt(max(0.0, dist_sq)))

        if max_z > self.max_coordinate_zscore:
            return SupportAssessment(
                is_supported=False,
                mahalanobis_distance=dist,
                max_zscore=max_z,
                rejection_reason=f"Coordinate z-score {max_z:.2f} exceeds threshold {self.max_coordinate_zscore:.2f}",
            )

        if dist > self.max_mahalanobis_distance:
            return SupportAssessment(
                is_supported=False,
                mahalanobis_distance=dist,
                max_zscore=max_z,
                rejection_reason=f"Mahalanobis distance {dist:.2f} exceeds threshold {self.max_mahalanobis_distance:.2f}",
            )

        return SupportAssessment(
            is_supported=True,
            mahalanobis_distance=dist,
            max_zscore=max_z,
            rejection_reason=None,
        )
