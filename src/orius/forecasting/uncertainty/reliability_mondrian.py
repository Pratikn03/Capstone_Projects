"""Reliability-conditioned conformal intervals for grouped coverage analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("split conformal quantile requires at least one score")
    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("alpha must lie in [0, 1) for split conformal quantiles")
    ordered = np.sort(values, kind="mergesort")
    rank = int(np.ceil((ordered.size + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), ordered.size)
    return float(ordered[rank - 1])


@dataclass(frozen=True)
class ReliabilityMondrianConfig:
    """Configuration for reliability-binned conformal calibration."""

    alpha: float = 0.10
    n_bins: int = 10
    min_bin_size: int = 25
    binning: str = "quantile"
    eps: float = 1e-6


class ReliabilityMondrian:
    """Residual conformal intervals grouped by telemetry reliability bins."""

    def __init__(self, config: ReliabilityMondrianConfig | None = None) -> None:
        self.config = config or ReliabilityMondrianConfig()
        self.bin_edges_: np.ndarray | None = None
        self.q_by_bin_: dict[int, float] = {}
        self.global_q_: float | None = None
        self.n_bins_: int = 0

    def _validate(self, y_true: np.ndarray, y_pred: np.ndarray, reliability: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
        reliability_arr = np.asarray(reliability, dtype=float).reshape(-1)
        if y_true_arr.shape != y_pred_arr.shape or y_true_arr.shape != reliability_arr.shape:
            raise ValueError("y_true, y_pred, and reliability must have identical 1D shapes")
        if y_true_arr.size == 0:
            raise ValueError("ReliabilityMondrian requires at least one sample")
        return y_true_arr, y_pred_arr, np.clip(reliability_arr, 0.0, 1.0)

    def _compute_edges(self, reliability: np.ndarray) -> np.ndarray:
        n_bins = max(int(self.config.n_bins), 1)
        if self.config.binning == "uniform":
            edges = np.linspace(0.0, 1.0, num=n_bins + 1, dtype=float)
        elif self.config.binning == "quantile":
            quantiles = np.linspace(0.0, 1.0, num=n_bins + 1, dtype=float)
            edges = np.quantile(reliability, quantiles)
        else:
            raise ValueError("binning must be 'quantile' or 'uniform'")
        edges[0] = min(edges[0], 0.0)
        edges[-1] = max(edges[-1], 1.0)
        unique = np.unique(edges)
        if unique.size < 2:
            unique = np.array([0.0, 1.0], dtype=float)
        return unique

    def _assign_bins(self, reliability: np.ndarray) -> np.ndarray:
        if self.bin_edges_ is None:
            raise RuntimeError("ReliabilityMondrian must be fit before assigning bins")
        bin_ids = np.digitize(reliability, self.bin_edges_[1:-1], right=False)
        return np.clip(bin_ids.astype(int), 0, self.n_bins_ - 1)

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, reliability: np.ndarray) -> "ReliabilityMondrian":
        y_true_arr, y_pred_arr, reliability_arr = self._validate(y_true, y_pred, reliability)
        residuals = np.abs(y_true_arr - y_pred_arr)
        self.bin_edges_ = self._compute_edges(reliability_arr)
        self.n_bins_ = int(len(self.bin_edges_) - 1)
        self.global_q_ = _split_conformal_quantile(residuals, self.config.alpha)

        bin_ids = self._assign_bins(reliability_arr)
        self.q_by_bin_ = {}
        for bin_idx in range(self.n_bins_):
            vals = residuals[bin_ids == bin_idx]
            if vals.size < max(int(self.config.min_bin_size), 1):
                self.q_by_bin_[bin_idx] = float(self.global_q_)
            else:
                self.q_by_bin_[bin_idx] = _split_conformal_quantile(vals, self.config.alpha)
        return self

    def predict_interval(self, y_pred: np.ndarray, reliability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.global_q_ is None or self.bin_edges_ is None:
            raise RuntimeError("ReliabilityMondrian must be fit before prediction")
        y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
        reliability_arr = np.clip(np.asarray(reliability, dtype=float).reshape(-1), 0.0, 1.0)
        if y_pred_arr.shape != reliability_arr.shape:
            raise ValueError("y_pred and reliability must have identical 1D shapes")
        bin_ids = self._assign_bins(reliability_arr)
        widths = np.asarray([self.q_by_bin_.get(int(bin_idx), float(self.global_q_)) for bin_idx in bin_ids], dtype=float)
        widths = np.maximum(widths, float(self.config.eps))
        return y_pred_arr - widths, y_pred_arr + widths

    def group_coverage(
        self,
        y_true: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        reliability: np.ndarray,
    ) -> list[dict[str, float | int]]:
        if self.bin_edges_ is None:
            raise RuntimeError("ReliabilityMondrian must be fit before coverage analysis")
        y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
        lower_arr = np.asarray(lower, dtype=float).reshape(-1)
        upper_arr = np.asarray(upper, dtype=float).reshape(-1)
        reliability_arr = np.clip(np.asarray(reliability, dtype=float).reshape(-1), 0.0, 1.0)
        if not (y_true_arr.shape == lower_arr.shape == upper_arr.shape == reliability_arr.shape):
            raise ValueError("group_coverage inputs must share identical 1D shapes")

        bin_ids = self._assign_bins(reliability_arr)
        rows: list[dict[str, float | int]] = []
        for bin_idx in range(self.n_bins_):
            mask = bin_ids == bin_idx
            n = int(mask.sum())
            if n == 0:
                picp = float("nan")
                width = float("nan")
            else:
                picp = float(np.mean((y_true_arr[mask] >= lower_arr[mask]) & (y_true_arr[mask] <= upper_arr[mask])))
                width = float(np.mean(upper_arr[mask] - lower_arr[mask]))
            rows.append(
                {
                    "bin_id": int(bin_idx),
                    "reliability_lower": float(self.bin_edges_[bin_idx]),
                    "reliability_upper": float(self.bin_edges_[bin_idx + 1]),
                    "n": n,
                    "picp": picp,
                    "mean_interval_width": width,
                    "qhat": float(self.q_by_bin_.get(bin_idx, float(self.global_q_ or 0.0))),
                }
            )
        return rows
