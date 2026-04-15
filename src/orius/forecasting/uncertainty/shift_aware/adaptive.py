from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdaptiveRecalibrationSummary:
    rows: int
    base_coverage: float
    adaptive_coverage: float
    base_mean_width: float
    adaptive_mean_width: float
    coverage_delta: float
    width_delta: float
    mean_adaptive_factor: float
    max_adaptive_factor: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "rows": int(self.rows),
            "base_coverage": float(self.base_coverage),
            "adaptive_coverage": float(self.adaptive_coverage),
            "base_mean_width": float(self.base_mean_width),
            "adaptive_mean_width": float(self.adaptive_mean_width),
            "coverage_delta": float(self.coverage_delta),
            "width_delta": float(self.width_delta),
            "mean_adaptive_factor": float(self.mean_adaptive_factor),
            "max_adaptive_factor": float(self.max_adaptive_factor),
        }


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(values.astype(float), nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def weighted_online_recalibration(
    *,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    reliability: np.ndarray,
    shift_score: np.ndarray,
    alpha: float = 0.10,
    decay: float = 0.90,
    learning_rate: float = 0.75,
    max_scale: float = 3.0,
) -> pd.DataFrame:
    """Apply online weighted interval widening under degraded telemetry.

    The base widening term stays monotone in reliability and shift.  An
    additional exponentially weighted miss term reacts when recent misses
    accumulate under low-reliability or high-shift conditions.
    """

    y_true = np.nan_to_num(np.asarray(y_true, dtype=float), nan=0.0)
    lower = np.nan_to_num(np.asarray(lower, dtype=float), nan=0.0)
    upper = np.nan_to_num(np.asarray(upper, dtype=float), nan=0.0)
    reliability = _clip01(np.asarray(reliability, dtype=float))
    shift_score = _clip01(np.asarray(shift_score, dtype=float))
    alpha = float(np.clip(alpha, 1.0e-6, 0.99))
    decay = float(np.clip(decay, 0.0, 0.999))
    learning_rate = float(max(learning_rate, 0.0))
    max_scale = float(max(max_scale, 1.0))

    rows: list[dict[str, Any]] = []
    weighted_gap_state = 0.0
    for idx in range(len(y_true)):
        center = 0.5 * (lower[idx] + upper[idx])
        half_width = max(0.5 * (upper[idx] - lower[idx]), 1.0e-9)
        base_factor = float(np.clip(1.0 + 0.7 * (1.0 - reliability[idx]) + 0.5 * shift_score[idx], 1.0, max_scale))
        adaptive_boost = max(0.0, weighted_gap_state)
        adaptive_factor = float(np.clip(base_factor * (1.0 + learning_rate * adaptive_boost), 1.0, max_scale))
        adaptive_half = half_width * adaptive_factor
        adaptive_lower = center - adaptive_half
        adaptive_upper = center + adaptive_half
        base_covered = bool(lower[idx] <= y_true[idx] <= upper[idx])
        adaptive_covered = bool(adaptive_lower <= y_true[idx] <= adaptive_upper)
        miss_weight = 1.0 + (1.0 - reliability[idx]) + shift_score[idx]
        weighted_residual = miss_weight * (0.0 if adaptive_covered else 1.0) - alpha
        weighted_gap_state = decay * weighted_gap_state + (1.0 - decay) * weighted_residual
        rows.append(
            {
                "row_index": int(idx),
                "y_true": float(y_true[idx]),
                "base_lower": float(lower[idx]),
                "base_upper": float(upper[idx]),
                "adaptive_lower": float(adaptive_lower),
                "adaptive_upper": float(adaptive_upper),
                "base_width": float(upper[idx] - lower[idx]),
                "adaptive_width": float(adaptive_upper - adaptive_lower),
                "base_factor": float(base_factor),
                "adaptive_factor": float(adaptive_factor),
                "reliability_w": float(reliability[idx]),
                "shift_score": float(shift_score[idx]),
                "base_covered": bool(base_covered),
                "adaptive_covered": bool(adaptive_covered),
                "weighted_gap_state": float(weighted_gap_state),
            }
        )
    return pd.DataFrame(rows)


def summarize_weighted_recalibration(frame: pd.DataFrame) -> AdaptiveRecalibrationSummary:
    if frame.empty:
        return AdaptiveRecalibrationSummary(
            rows=0,
            base_coverage=0.0,
            adaptive_coverage=0.0,
            base_mean_width=0.0,
            adaptive_mean_width=0.0,
            coverage_delta=0.0,
            width_delta=0.0,
            mean_adaptive_factor=1.0,
            max_adaptive_factor=1.0,
        )
    base_coverage = float(frame["base_covered"].mean())
    adaptive_coverage = float(frame["adaptive_covered"].mean())
    base_width = float(frame["base_width"].mean())
    adaptive_width = float(frame["adaptive_width"].mean())
    return AdaptiveRecalibrationSummary(
        rows=int(len(frame)),
        base_coverage=base_coverage,
        adaptive_coverage=adaptive_coverage,
        base_mean_width=base_width,
        adaptive_mean_width=adaptive_width,
        coverage_delta=float(adaptive_coverage - base_coverage),
        width_delta=float(adaptive_width - base_width),
        mean_adaptive_factor=float(frame["adaptive_factor"].mean()),
        max_adaptive_factor=float(frame["adaptive_factor"].max()),
    )
