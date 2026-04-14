"""Residual-stream monitoring for uncertainty validity state."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from orius.dc3s.drift import PageHinkleyDetector


@dataclass
class ResidualValidityMonitor:
    window: int = 64
    error_rate_threshold: float = 0.2
    watch_threshold: float = 0.75
    degraded_threshold: float = 0.55
    invalid_threshold: float = 0.35
    detector: PageHinkleyDetector = field(default_factory=PageHinkleyDetector)

    def __post_init__(self) -> None:
        self._resids: deque[float] = deque(maxlen=self.window)
        self._misses: deque[int] = deque(maxlen=self.window)

    def update(
        self,
        *,
        abs_residual: float,
        covered: bool,
        reliability_score: float,
        telemetry_degraded: bool,
        subgroup_gap: float,
    ) -> dict[str, Any]:
        self._resids.append(float(abs_residual))
        self._misses.append(int(not covered))
        drift = self.detector.update(float(abs_residual))

        arr = np.asarray(self._resids, dtype=float)
        mean = float(arr.mean()) if arr.size else 0.0
        std = float(arr.std(ddof=0)) if arr.size else 0.0
        norm = float(mean / max(std, 1e-6)) if arr.size else 0.0
        err_rate = float(np.mean(self._misses)) if self._misses else 0.0

        # Separate telemetry vs uncertainty validity diagnostics.
        telemetry_bad = bool(telemetry_degraded)
        uq_bad = bool(drift.get("drift", False) or err_rate >= self.error_rate_threshold or subgroup_gap > 0.05)

        score = 1.0
        score -= min(max(norm / 5.0, 0.0), 0.35)
        score -= min(max(err_rate, 0.0), 0.30)
        score -= min(max(subgroup_gap * 4.0, 0.0), 0.20)
        score -= min(max(1.0 - reliability_score, 0.0), 0.15)
        score = float(min(max(score, 0.0), 1.0))

        status = "nominal"
        if score <= self.invalid_threshold:
            status = "invalid"
        elif score <= self.degraded_threshold:
            status = "degraded"
        elif score <= self.watch_threshold:
            status = "watch"

        return {
            "rolling_residual_mean": mean,
            "rolling_residual_std": std,
            "normalized_residual_score": norm,
            "error_rate": err_rate,
            "drift": bool(drift.get("drift", False)),
            "drift_score": float(drift.get("score", 0.0)),
            "telemetry_degraded": telemetry_bad,
            "uncertainty_validity_loss": uq_bad,
            "both_degraded": telemetry_bad and uq_bad,
            "state": status,
            "validity_score": score,
            "subgroup_alert": subgroup_gap > 0.05,
        }
