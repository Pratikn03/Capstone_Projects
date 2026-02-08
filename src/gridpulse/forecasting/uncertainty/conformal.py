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
from typing import Any, Optional, Tuple

import json
import numpy as np


@dataclass
class ConformalConfig:
    alpha: float = 0.10
    horizon_wise: bool = True
    rolling: bool = True
    rolling_window: int = 720
    eps: float = 1e-6


class ConformalInterval:
    def __init__(self, cfg: ConformalConfig):
        self.cfg = cfg
        self.q_global: Optional[float] = None
        self.q_h: Optional[np.ndarray] = None
        self._resid_buffers = None

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

    def mean_width(self, y_pred: np.ndarray) -> float:
        lo, hi = self.predict_interval(y_pred)
        return float(np.mean(hi - lo))

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "alpha": self.cfg.alpha,
                "horizon_wise": self.cfg.horizon_wise,
                "rolling": self.cfg.rolling,
                "rolling_window": self.cfg.rolling_window,
                "eps": self.cfg.eps,
            },
            "q_global": self.q_global,
            "q_h": self.q_h.tolist() if self.q_h is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConformalInterval":
        cfg = ConformalConfig(**payload.get("config", {}))
        inst = cls(cfg)
        if payload.get("q_global") is not None:
            inst.q_global = float(payload["q_global"])
        if payload.get("q_h") is not None:
            inst.q_h = np.asarray(payload["q_h"], dtype=float)
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
