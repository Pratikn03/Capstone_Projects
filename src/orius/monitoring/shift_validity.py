from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from orius.dc3s.drift import PageHinkleyDetector


@dataclass
class ResidualStreamMonitor:
    window: int = 64
    error_rate_threshold: float = 0.2
    ph_detector: PageHinkleyDetector = field(default_factory=PageHinkleyDetector)
    _residuals: deque[float] = field(default_factory=deque)
    _misses: deque[int] = field(default_factory=deque)

    def update(self, *, residual: float, miss: bool, reliability_score: float) -> dict[str, Any]:
        r = abs(float(residual))
        self._residuals.append(r)
        self._misses.append(1 if miss else 0)
        while len(self._residuals) > self.window:
            self._residuals.popleft()
        while len(self._misses) > self.window:
            self._misses.popleft()

        n = max(1, len(self._residuals))
        mean = sum(self._residuals) / n
        var = sum((x - mean) ** 2 for x in self._residuals) / n
        std = var ** 0.5
        norm = r / max(1e-6, std if std > 0 else mean if mean > 0 else 1.0)
        ph = self.ph_detector.update(r)
        error_rate = sum(self._misses) / max(1, len(self._misses))

        degraded_telemetry = reliability_score < 0.6
        validity_loss = bool(ph.get("drift", False)) or error_rate > self.error_rate_threshold
        if degraded_telemetry and validity_loss:
            state = "invalid"
        elif degraded_telemetry or validity_loss or norm > 1.5:
            state = "degraded"
        elif norm > 1.0:
            state = "watch"
        else:
            state = "nominal"

        return {
            "state": state,
            "residual_mean": mean,
            "residual_std": std,
            "normalized_residual": norm,
            "error_rate": error_rate,
            "drift": bool(ph.get("drift", False)),
            "drift_score": float(ph.get("score", 0.0)),
            "degraded_telemetry": degraded_telemetry,
            "uncertainty_validity_loss": validity_loss,
        }
