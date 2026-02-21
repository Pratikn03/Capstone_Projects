"""Drift detection for DC3S using Page-Hinkley on residual magnitude."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PageHinkleyDetector:
    """
    Page-Hinkley detector over r_t = |y_t - yhat_t|.

    Drift score is max(0, cumulative_sum - running_min_sum).
    """

    delta: float = 0.01
    threshold: float = 5.0
    warmup_steps: int = 48
    cooldown_steps: int = 24
    count: int = 0
    mean: float = 0.0
    cumulative_sum: float = 0.0
    min_cumulative_sum: float = 0.0
    cooldown_remaining: int = 0

    def update(self, r_t: float) -> Dict[str, float | bool | int]:
        value = float(r_t)
        self.count += 1

        if self.count == 1:
            self.mean = value
        else:
            self.mean += (value - self.mean) / float(self.count)

        self.cumulative_sum += value - self.mean - self.delta
        self.min_cumulative_sum = min(self.min_cumulative_sum, self.cumulative_sum)
        score = max(0.0, self.cumulative_sum - self.min_cumulative_sum)

        drift = False
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        elif self.count > self.warmup_steps and score > self.threshold:
            drift = True
            self.cooldown_remaining = max(0, int(self.cooldown_steps))
            self.cumulative_sum = 0.0
            self.min_cumulative_sum = 0.0
            score = 0.0

        return {
            "drift": bool(drift),
            "score": float(score),
            "count": int(self.count),
            "cooldown_remaining": int(self.cooldown_remaining),
            "mean_residual": float(self.mean),
        }

    def to_state(self) -> Dict[str, Any]:
        return {
            "delta": float(self.delta),
            "threshold": float(self.threshold),
            "warmup_steps": int(self.warmup_steps),
            "cooldown_steps": int(self.cooldown_steps),
            "count": int(self.count),
            "mean": float(self.mean),
            "cumulative_sum": float(self.cumulative_sum),
            "min_cumulative_sum": float(self.min_cumulative_sum),
            "cooldown_remaining": int(self.cooldown_remaining),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any] | None, cfg: Dict[str, Any] | None = None) -> "PageHinkleyDetector":
        cfg = cfg or {}
        detector = cls(
            delta=float(cfg.get("ph_delta", 0.01)),
            threshold=float(cfg.get("ph_lambda", 5.0)),
            warmup_steps=int(cfg.get("warmup_steps", 48)),
            cooldown_steps=int(cfg.get("cooldown_steps", 24)),
        )
        if not state:
            return detector

        detector.count = int(state.get("count", detector.count))
        detector.mean = float(state.get("mean", detector.mean))
        detector.cumulative_sum = float(state.get("cumulative_sum", detector.cumulative_sum))
        detector.min_cumulative_sum = float(state.get("min_cumulative_sum", detector.min_cumulative_sum))
        detector.cooldown_remaining = int(state.get("cooldown_remaining", detector.cooldown_remaining))
        detector.delta = float(state.get("delta", detector.delta))
        detector.threshold = float(state.get("threshold", detector.threshold))
        detector.warmup_steps = int(state.get("warmup_steps", detector.warmup_steps))
        detector.cooldown_steps = int(state.get("cooldown_steps", detector.cooldown_steps))
        return detector
