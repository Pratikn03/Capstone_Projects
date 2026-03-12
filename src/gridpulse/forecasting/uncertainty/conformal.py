"""
Conformal prediction intervals for regression forecasts.

Supports:
- global q (single quantile for all horizons)
- horizon-wise q_h (separate quantile per horizon step)
- rolling calibration (update residual window over time)

This is model-agnostic: works with GBM/LSTM/TCN as long as you provide y_true and y_pred arrays.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import json
import numpy as np


@dataclass
class ConformalConfig:
    alpha: float = 0.10
    method: Literal["residual", "cqr"] = "residual"
    horizon_wise: bool = True
    rolling: bool = True
    rolling_window: int = 720
    eps: float = 1e-6


class AdaptiveConformal:
    """Fully Adaptive Conformal Inference (FACI) with online alpha updates."""

    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.05,
        mode: Literal["global", "horizon_wise"] = "global",
        alpha_min: float = 0.01,
        alpha_max: float = 0.99,
        eps: float = 1e-6,
    ) -> None:
        if mode not in {"global", "horizon_wise"}:
            raise ValueError("mode must be one of: 'global', 'horizon_wise'")
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if alpha_min >= alpha_max:
            raise ValueError("alpha_min must be < alpha_max")
        if alpha < alpha_min or alpha > alpha_max:
            raise ValueError("alpha must be within [alpha_min, alpha_max]")

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.mode = mode
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.eps = float(eps)

        self.alpha_t: float | np.ndarray = float(alpha)
        self._alpha0_h: Optional[np.ndarray] = None

    def _normalize_inputs(
        self,
        y_true: np.ndarray | list[float] | float,
        y_pred_interval: tuple[np.ndarray | list[float] | float, np.ndarray | list[float] | float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(y_pred_interval, (tuple, list)) or len(y_pred_interval) != 2:
            raise ValueError("y_pred_interval must be a tuple/list of (lower, upper)")

        lower, upper = y_pred_interval

        y_true_arr = np.asarray(y_true, dtype=float)
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)

        try:
            y_true_b, lower_b, upper_b = np.broadcast_arrays(y_true_arr, lower_arr, upper_arr)
        except ValueError as exc:
            raise ValueError("y_true and interval bounds must be broadcast-compatible") from exc

        if np.any(lower_b > upper_b):
            raise ValueError("Interval lower bound must be <= upper bound for all elements")

        return y_true_b, lower_b, upper_b

    def _step_alpha(self, outside: np.ndarray) -> None:
        if self.mode == "global":
            outside_any = bool(np.any(outside))
            next_alpha = float(self.alpha_t) + (self.gamma if outside_any else -self.gamma)
            self.alpha_t = float(np.clip(next_alpha, self.alpha_min, self.alpha_max))
            return

        if outside.ndim == 0:
            misses = np.asarray([bool(outside)], dtype=bool)
        elif outside.ndim == 1:
            misses = outside.astype(bool)
        else:
            axes = tuple(range(outside.ndim - 1))
            misses = np.any(outside, axis=axes)

        horizon = int(misses.shape[-1]) if misses.ndim > 0 else 1

        if isinstance(self.alpha_t, np.ndarray):
            if self.alpha_t.shape != misses.shape:
                if self.alpha_t.size == 1:
                    self.alpha_t = np.full(misses.shape, float(self.alpha_t.reshape(-1)[0]), dtype=float)
                else:
                    raise ValueError("AdaptiveConformal horizon size changed after initialization")
        else:
            self.alpha_t = np.full(horizon, float(self.alpha_t), dtype=float)

        if self._alpha0_h is None:
            self._alpha0_h = np.full_like(self.alpha_t, self.alpha, dtype=float)
        elif self._alpha0_h.shape != self.alpha_t.shape:
            raise ValueError("AdaptiveConformal baseline alpha shape does not match current horizon")

        step = np.where(misses, self.gamma, -self.gamma)
        self.alpha_t = np.clip(self.alpha_t + step, self.alpha_min, self.alpha_max)

    def update(
        self,
        y_true: np.ndarray | list[float] | float,
        y_pred_interval: tuple[np.ndarray | list[float] | float, np.ndarray | list[float] | float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update adaptive alpha and return interval widths for the next step."""
        y_true_b, lower_b, upper_b = self._normalize_inputs(y_true, y_pred_interval)

        # Boundary points are treated as inside intervals.
        outside = (y_true_b < lower_b) | (y_true_b > upper_b)
        self._step_alpha(outside)

        midpoint = 0.5 * (lower_b + upper_b)
        half_width = np.maximum(0.5 * (upper_b - lower_b), self.eps)

        if self.mode == "global":
            alpha_ref = max(self.alpha, self.eps)
            scale = float(self.alpha_t) / alpha_ref
            lower_new = midpoint - half_width * scale
            upper_new = midpoint + half_width * scale
            return np.asarray(lower_new, dtype=float), np.asarray(upper_new, dtype=float)

        if not isinstance(self.alpha_t, np.ndarray) or self._alpha0_h is None:
            raise RuntimeError("AdaptiveConformal internal alpha state was not initialized")

        scale_vec = self.alpha_t / np.maximum(self._alpha0_h, self.eps)
        if midpoint.ndim == 0:
            scale = float(scale_vec.reshape(-1)[0])
        else:
            scale = scale_vec.reshape((1,) * (midpoint.ndim - 1) + scale_vec.shape)
        lower_new = midpoint - half_width * scale
        upper_new = midpoint + half_width * scale
        return np.asarray(lower_new, dtype=float), np.asarray(upper_new, dtype=float)


class ConformalInterval:
    def __init__(self, cfg: ConformalConfig):
        self.cfg = cfg
        self.q_global: Optional[float] = None
        self.q_h: Optional[np.ndarray] = None
        self._resid_buffers = None
        self.method: Literal["residual", "cqr"] = cfg.method

    def fit_calibration(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        y_true: shape (N, H) or (N,)
        y_pred: shape (N, H) or (N,)

        Stores quantile(s) of |residual| for interval construction.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        resid = np.abs(y_true - y_pred)
        if resid.ndim == 1:
            resid = resid.reshape(-1, 1)

        _, horizon = resid.shape

        if self.cfg.horizon_wise:
            self.q_h = np.quantile(resid, 1.0 - self.cfg.alpha, axis=0)
            self.q_global = float(np.quantile(resid.flatten(), 1.0 - self.cfg.alpha))
            if self.cfg.rolling:
                self._resid_buffers = [
                    deque(resid[:, h].tolist(), maxlen=self.cfg.rolling_window) for h in range(horizon)
                ]
        else:
            self.q_global = float(np.quantile(resid.flatten(), 1.0 - self.cfg.alpha))
            if self.cfg.rolling:
                self._resid_buffers = [deque(resid.flatten().tolist(), maxlen=self.cfg.rolling_window)]

    def update(self, y_true_new: np.ndarray, y_pred_new: np.ndarray) -> None:
        """Rolling update using new observations."""
        if not self.cfg.rolling or self._resid_buffers is None:
            return

        residuals = np.abs(np.asarray(y_true_new) - np.asarray(y_pred_new))
        if residuals.ndim == 0:
            residuals = np.array([residuals])
        if residuals.ndim == 1:
            if self.cfg.horizon_wise:
                for h, val in enumerate(residuals.tolist()):
                    self._resid_buffers[h].append(float(val))
            else:
                for val in residuals.tolist():
                    self._resid_buffers[0].append(float(val))
        else:
            if self.cfg.horizon_wise:
                for row in residuals:
                    for h, val in enumerate(row.tolist()):
                        self._resid_buffers[h].append(float(val))
            else:
                for val in residuals.flatten().tolist():
                    self._resid_buffers[0].append(float(val))

        if self.cfg.horizon_wise:
            self.q_h = np.array([
                np.quantile(np.array(buf), 1.0 - self.cfg.alpha) for buf in self._resid_buffers
            ])
            all_vals = np.concatenate([np.array(buf) for buf in self._resid_buffers if len(buf) > 0])
            if all_vals.size:
                self.q_global = float(np.quantile(all_vals, 1.0 - self.cfg.alpha))
        else:
            self.q_global = float(np.quantile(np.array(self._resid_buffers[0]), 1.0 - self.cfg.alpha))

    def fit_calibration_cqr(self, y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> None:
        """
        Conformalized Quantile Regression calibration.

        nonconformity score: s = max(q_lo - y, y - q_hi)
        """
        y_true = np.asarray(y_true)
        q_lo = np.asarray(q_lo)
        q_hi = np.asarray(q_hi)
        if y_true.shape != q_lo.shape or y_true.shape != q_hi.shape:
            raise ValueError("y_true, q_lo, and q_hi must have identical shape for CQR calibration")

        scores = np.maximum(q_lo - y_true, y_true - q_hi)
        scores = np.maximum(scores, 0.0)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        _, horizon = scores.shape
        if self.cfg.horizon_wise:
            self.q_h = np.quantile(scores, 1.0 - self.cfg.alpha, axis=0)
            self.q_global = float(np.quantile(scores.flatten(), 1.0 - self.cfg.alpha))
            if self.cfg.rolling:
                self._resid_buffers = [
                    deque(scores[:, h].tolist(), maxlen=self.cfg.rolling_window) for h in range(horizon)
                ]
        else:
            self.q_global = float(np.quantile(scores.flatten(), 1.0 - self.cfg.alpha))
            if self.cfg.rolling:
                self._resid_buffers = [deque(scores.flatten().tolist(), maxlen=self.cfg.rolling_window)]

        self.method = "cqr"

    def predict_interval(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower/upper arrays with the same shape as y_pred."""
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            # Treat 1D input as a single horizon vector.
            y_pred2 = y_pred.reshape(1, -1)
        else:
            y_pred2 = y_pred

        _, horizon = y_pred2.shape

        if self.cfg.horizon_wise:
            if self.q_h is None:
                raise RuntimeError("ConformalInterval not calibrated. Call fit_calibration() first.")
            if len(self.q_h) == horizon:
                q = self.q_h.reshape(1, horizon)
            elif self.q_global is not None:
                q = np.full((1, horizon), self.q_global)
            else:
                raise RuntimeError(
                    f"Conformal horizon {len(self.q_h)} does not match request horizon {horizon}."
                )
        else:
            if self.q_global is None:
                raise RuntimeError("ConformalInterval not calibrated. Call fit_calibration() first.")
            q = np.full((1, horizon), self.q_global)

        lower = y_pred2 - q
        upper = y_pred2 + q

        if y_pred.ndim == 1:
            return lower.flatten(), upper.flatten()
        return lower, upper

    def coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        lo, hi = self.predict_interval(y_pred)
        y_true = np.asarray(y_true)
        return float(np.mean((y_true >= lo) & (y_true <= hi)))

    def predict_interval_cqr(self, q_lo: np.ndarray, q_hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return CQR-corrected intervals: [q_lo - q_hat, q_hi + q_hat]."""
        q_lo = np.asarray(q_lo)
        q_hi = np.asarray(q_hi)
        if q_lo.shape != q_hi.shape:
            raise ValueError("q_lo and q_hi must have identical shape")
        if q_lo.ndim == 1:
            q_lo2 = q_lo.reshape(1, -1)
            q_hi2 = q_hi.reshape(1, -1)
        else:
            q_lo2 = q_lo
            q_hi2 = q_hi
        _, horizon = q_lo2.shape

        if self.cfg.horizon_wise:
            if self.q_h is None:
                raise RuntimeError("ConformalInterval not calibrated for CQR. Call fit_calibration_cqr() first.")
            if len(self.q_h) == horizon:
                q_hat = self.q_h.reshape(1, horizon)
            elif self.q_global is not None:
                q_hat = np.full((1, horizon), self.q_global)
            else:
                raise RuntimeError(
                    f"Conformal horizon {len(self.q_h)} does not match request horizon {horizon} and no global fallback is available."
                )
        else:
            if self.q_global is None:
                raise RuntimeError("ConformalInterval not calibrated for CQR. Call fit_calibration_cqr() first.")
            q_hat = np.full((1, horizon), self.q_global)

        lower = q_lo2 - q_hat
        upper = q_hi2 + q_hat
        if q_lo.ndim == 1:
            return lower.flatten(), upper.flatten()
        return lower, upper

    def mean_width(self, y_pred: np.ndarray) -> float:
        lo, hi = self.predict_interval(y_pred)
        return float(np.mean(hi - lo))

    @staticmethod
    def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        error = y_true - y_pred
        q = float(quantile)
        return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))

    @staticmethod
    def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
        y_true = np.asarray(y_true, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        alpha = max(float(alpha), 1e-9)
        width = upper - lower
        below = y_true < lower
        above = y_true > upper
        score = width.copy()
        score[below] = width[below] + (2.0 / alpha) * (lower[below] - y_true[below])
        score[above] = width[above] + (2.0 / alpha) * (y_true[above] - upper[above])
        return float(np.mean(score))

    def interval_metrics_from_bounds(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> dict[str, float]:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        coverage_90 = float(np.mean((y_true >= lower) & (y_true <= upper)))
        width = upper - lower
        alpha = float(self.cfg.alpha)
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * width
        lower_95 = midpoint - 1.2 * half_width
        upper_95 = midpoint + 1.2 * half_width
        lower_q = alpha / 2.0
        upper_q = 1.0 - alpha / 2.0
        return {
            "picp_90": coverage_90,
            "picp_95": float(np.mean((y_true >= lower_95) & (y_true <= upper_95))),
            "mean_interval_width": float(np.mean(width)),
            "pinball_loss_q05": self.pinball_loss(y_true, lower, lower_q),
            "pinball_loss_q50": self.pinball_loss(y_true, y_pred, 0.5),
            "pinball_loss_q95": self.pinball_loss(y_true, upper, upper_q),
            "pinball_loss_mean": float(
                np.mean(
                    [
                        self.pinball_loss(y_true, lower, lower_q),
                        self.pinball_loss(y_true, y_pred, 0.5),
                        self.pinball_loss(y_true, upper, upper_q),
                    ]
                )
            ),
            "winkler_score_90": self.winkler_score(y_true, lower, upper, alpha),
        }

    def per_horizon_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Compute PICP (Prediction Interval Coverage Probability) per horizon step.
        
        Returns:
            Dictionary mapping horizon step ("h1", "h2", ...) to coverage probability
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        lo, hi = self.predict_interval(y_pred)
        
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
            lo = lo.reshape(-1, 1)
            hi = hi.reshape(-1, 1)
        
        _, horizon = y_true.shape
        coverage_per_h = {}
        
        for h in range(horizon):
            yt = y_true[:, h]
            yt_lo = lo[:, h]
            yt_hi = hi[:, h]
            coverage = np.mean((yt >= yt_lo) & (yt <= yt_hi))
            coverage_per_h[f"h{h+1}"] = float(coverage)
        
        return coverage_per_h

    def per_horizon_width(self, y_pred: np.ndarray) -> dict[str, float]:
        """
        Compute MPIW (Mean Prediction Interval Width) per horizon step.
        
        Returns:
            Dictionary mapping horizon step ("h1", "h2", ...) to mean interval width
        """
        y_pred = np.asarray(y_pred)
        lo, hi = self.predict_interval(y_pred)
        
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
            lo = lo.reshape(-1, 1)
            hi = hi.reshape(-1, 1)
        
        _, horizon = y_pred.shape
        width_per_h = {}
        
        for h in range(horizon):
            width = np.mean(hi[:, h] - lo[:, h])
            width_per_h[f"h{h+1}"] = float(width)
        
        return width_per_h

    def evaluate_intervals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        per_horizon: bool = True
    ) -> dict[str, Any]:
        """
        Comprehensive interval evaluation: global + per-horizon PICP/MPIW.
        
        Args:
            y_true: True values (n_samples, horizon) or (n_samples,)
            y_pred: Predicted values (same shape as y_true)
            per_horizon: Whether to compute per-horizon metrics
        
        Returns:
            Dictionary with global and per-horizon coverage/width metrics
        """
        results = {
            "global_coverage": self.coverage(y_true, y_pred),
            "global_mean_width": self.mean_width(y_pred),
        }
        lo, hi = self.predict_interval(y_pred)
        results.update(
            self.interval_metrics_from_bounds(
                y_true=y_true,
                y_pred=y_pred,
                lower=lo,
                upper=hi,
            )
        )
        
        if per_horizon:
            results["per_horizon_picp"] = self.per_horizon_coverage(y_true, y_pred)
            results["per_horizon_mpiw"] = self.per_horizon_width(y_pred)
        
        return results

    def evaluate_intervals_cqr(
        self,
        y_true: np.ndarray,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
        per_horizon: bool = True,
    ) -> dict[str, Any]:
        """Evaluate CQR-corrected intervals."""
        y_true = np.asarray(y_true)
        lower, upper = self.predict_interval_cqr(q_lo, q_hi)
        global_coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
        global_width = float(np.mean(upper - lower))
        results = {
            "global_coverage": global_coverage,
            "global_mean_width": global_width,
        }
        midpoint = 0.5 * (np.asarray(lower, dtype=float) + np.asarray(upper, dtype=float))
        results.update(
            self.interval_metrics_from_bounds(
                y_true=y_true,
                y_pred=midpoint,
                lower=lower,
                upper=upper,
            )
        )
        if per_horizon:
            if y_true.ndim == 1:
                y_true = y_true.reshape(-1, 1)
                lower = lower.reshape(-1, 1)
                upper = upper.reshape(-1, 1)
            results["per_horizon_picp"] = {
                f"h{i+1}": float(np.mean((y_true[:, i] >= lower[:, i]) & (y_true[:, i] <= upper[:, i])))
                for i in range(y_true.shape[1])
            }
            results["per_horizon_mpiw"] = {
                f"h{i+1}": float(np.mean(upper[:, i] - lower[:, i]))
                for i in range(y_true.shape[1])
            }
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "alpha": self.cfg.alpha,
                "method": self.cfg.method,
                "horizon_wise": self.cfg.horizon_wise,
                "rolling": self.cfg.rolling,
                "rolling_window": self.cfg.rolling_window,
                "eps": self.cfg.eps,
            },
            "q_global": self.q_global,
            "q_h": self.q_h.tolist() if self.q_h is not None else None,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConformalInterval":
        cfg = ConformalConfig(**payload.get("config", {}))
        inst = cls(cfg)
        if payload.get("q_global") is not None:
            inst.q_global = float(payload["q_global"])
        if payload.get("q_h") is not None:
            inst.q_h = np.asarray(payload["q_h"], dtype=float)
        method = payload.get("method")
        if method in {"residual", "cqr"}:
            inst.method = method
        return inst


def save_conformal(path: str | Path, interval: ConformalInterval, meta: Optional[dict[str, Any]] = None) -> None:
    payload = interval.to_dict()
    if meta:
        payload["meta"] = meta
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_conformal(path: str | Path) -> ConformalInterval:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ConformalInterval.from_dict(payload)
