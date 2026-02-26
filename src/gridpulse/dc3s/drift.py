"""Drift detection for DC³S.

Provides two detectors operating on the residual magnitude r_t = |y_t - ŷ_t|:

  1. PageHinkleyDetector  — cumulative-sum based, low memory, fast.
  2. ADWINDetector        — Adaptive Windowing (Bifet & Gavalda, 2007),
                           maintains two sub-windows and triggers when their
                           means diverge beyond a Hoeffding-style confidence
                           bound. Adapts window size automatically.

Both conform to the same .update(r_t) -> dict interface and .to_state() /
.from_state() for serializable online persistence.

References:
    Bifet, A. & Gavalda, R. (2007). Learning from time-changing data with
    adaptive windowing. SIAM International Conference on Data Mining.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict


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


@dataclass
class ADWINDetector:
    """
    Adaptive Windowing (ADWIN) drift detector over r_t = |y_t - ŷ_t|.

    ADWIN maintains a sliding window W of recent residuals and detects
    concept drift when the mean of a sub-window W1 (older) significantly
    differs from W2 (recent), as judged by a Hoeffding-style bound:

        |mean(W1) - mean(W2)| ≥ ε_cut(δ, |W1|, |W2|)

    where ε_cut is derived from the Hoeffding inequality with confidence δ.
    When drift is detected, the window is trimmed to W2 (the recent portion),
    effectively forgetting stale data.

    Advantages over Page-Hinkley:
        - No need to specify a mean shift (delta) parameter upfront.
        - Automatically adapts window size: large window during stability,
          small window after drift.
        - Provides variance-aware detection.

    Reference:
        Bifet, A. & Gavalda, R. (2007). Learning from time-changing data with
        adaptive windowing. SIAM International Conference on Data Mining.

    Attributes:
        delta:          Confidence parameter δ ∈ (0,1). Smaller = more cautious.
        max_window:     Hard cap on window size to bound memory.
        min_window:     Minimum window size before detection is allowed.
        cooldown_steps: Steps to wait after a drift detection before re-arming.
    """

    delta: float = 0.002
    max_window: int = 1200
    min_window: int = 32
    cooldown_steps: int = 24
    _window: Deque[float] = field(default_factory=deque, repr=False)
    _count: int = 0
    _cooldown_remaining: int = 0

    def update(self, r_t: float) -> Dict[str, float | bool | int]:
        """
        Ingest one residual sample and return drift status.

        Args:
            r_t: Residual magnitude |y_t - ŷ_t| at the current step.

        Returns:
            dict with keys:
                drift (bool): True if drift detected this step.
                score (float): Maximum sub-window mean difference observed.
                count (int): Total samples ingested.
                window_size (int): Current window length.
                cooldown_remaining (int): Steps remaining in post-drift cooldown.
                mean_residual (float): Mean of current window.
        """
        value = float(r_t)
        self._window.append(value)
        self._count += 1

        # Enforce max window size (FIFO trim)
        while len(self._window) > self.max_window:
            self._window.popleft()

        drift = False
        max_diff = 0.0

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
        elif len(self._window) >= self.min_window:
            drift, max_diff = self._detect_and_trim()

        window_list = list(self._window)
        return {
            "drift": bool(drift),
            "score": float(max_diff),
            "count": int(self._count),
            "window_size": int(len(self._window)),
            "cooldown_remaining": int(self._cooldown_remaining),
            "mean_residual": float(sum(window_list) / len(window_list)) if window_list else 0.0,
        }

    def _detect_and_trim(self) -> tuple[bool, float]:
        """
        Scan sub-window splits and test each for the Hoeffding cut criterion.

        Returns:
            (drift_detected, max_abs_diff)
        """
        window_list = list(self._window)
        n = len(window_list)
        if n < self.min_window:
            return False, 0.0

        total = sum(window_list)
        total_sq = sum(v * v for v in window_list)
        prefix_sum = 0.0
        prefix_sq = 0.0
        max_diff = 0.0
        best_split = -1

        # Try every split point; keep track of the most significant divergence
        for i in range(1, n - 1):
            prefix_sum += window_list[i - 1]
            prefix_sq += window_list[i - 1] ** 2
            n0 = i
            n1 = n - i

            mean0 = prefix_sum / n0
            mean1 = (total - prefix_sum) / n1

            # Variance estimate: V[W] ≤ (max - min)^2 / 4 (range-based bound)
            # We use per-sub-window variance for tighter bound
            var0 = max(0.0, prefix_sq / n0 - mean0 ** 2)
            var1 = max(0.0, (total_sq - prefix_sq) / n1 - mean1 ** 2)

            # Hoeffding-style ε_cut (Bifet & Gavalda eq. 1):
            # ε_cut = sqrt((var_pool * 2 / m) * ln(4n² / delta))
            # where var_pool is estimated and m = (n0*n1)/(n0+n1)
            m = (n0 * n1) / n
            var_pool = (n0 * var0 + n1 * var1) / n
            log_term = math.log(4.0 * n * n / self.delta)
            eps_cut = math.sqrt((var_pool * 2.0 / max(m, 1e-9)) * log_term) if m > 0 else float("inf")

            abs_diff = abs(mean0 - mean1)
            if abs_diff > max_diff:
                max_diff = abs_diff
                if abs_diff >= eps_cut:
                    best_split = i

        if best_split > 0:
            # Trim: discard older sub-window (W1), keep recent sub-window (W2)
            for _ in range(best_split):
                if self._window:
                    self._window.popleft()
            self._cooldown_remaining = max(0, int(self.cooldown_steps))
            return True, max_diff

        return False, max_diff

    def to_state(self) -> Dict[str, Any]:
        return {
            "delta": float(self.delta),
            "max_window": int(self.max_window),
            "min_window": int(self.min_window),
            "cooldown_steps": int(self.cooldown_steps),
            "count": int(self._count),
            "cooldown_remaining": int(self._cooldown_remaining),
            "window": list(self._window),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any] | None, cfg: Dict[str, Any] | None = None) -> "ADWINDetector":
        cfg = cfg or {}
        detector = cls(
            delta=float(cfg.get("adwin_delta", 0.002)),
            max_window=int(cfg.get("adwin_max_window", 1200)),
            min_window=int(cfg.get("adwin_min_window", 32)),
            cooldown_steps=int(cfg.get("cooldown_steps", 24)),
        )
        if not state:
            return detector
        detector.delta = float(state.get("delta", detector.delta))
        detector.max_window = int(state.get("max_window", detector.max_window))
        detector.min_window = int(state.get("min_window", detector.min_window))
        detector.cooldown_steps = int(state.get("cooldown_steps", detector.cooldown_steps))
        detector._count = int(state.get("count", 0))
        detector._cooldown_remaining = int(state.get("cooldown_remaining", 0))
        detector._window = deque(state.get("window", []), maxlen=None)
        return detector


def make_detector(cfg: Dict[str, Any] | None = None) -> "PageHinkleyDetector | ADWINDetector":
    """
    Factory: instantiate either PageHinkley or ADWIN based on cfg['detector'].

    Config keys:
        detector: "page_hinkley" (default) or "adwin"
        ph_delta, ph_lambda, warmup_steps, cooldown_steps  (PageHinkley)
        adwin_delta, adwin_max_window, adwin_min_window     (ADWIN)

    Returns:
        Configured detector instance.
    """
    cfg = cfg or {}
    detector_type = str(cfg.get("detector", "page_hinkley")).lower().replace("-", "_")
    if detector_type == "adwin":
        return ADWINDetector.from_state(None, cfg)
    return PageHinkleyDetector.from_state(None, cfg)


__all__ = ["PageHinkleyDetector", "ADWINDetector", "make_detector"]
