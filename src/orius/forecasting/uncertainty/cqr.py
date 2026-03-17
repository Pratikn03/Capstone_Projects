from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np


@dataclass
class RegimeCQRConfig:
    alpha: float = 0.10
    n_bins: int = 3
    vol_window: int = 24
    eps: float = 1e-9
    fail_fast_quantile_backend: bool = True


def rolling_volatility(y: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(y, dtype=float).reshape(-1)
    if window <= 1 or values.size == 0:
        return np.zeros_like(values, dtype=float)
    vol = np.zeros_like(values, dtype=float)
    for i in range(values.size):
        start = max(0, i - window + 1)
        seg = values[start : i + 1]
        vol[i] = float(np.std(seg)) if seg.size > 1 else 0.0
    return vol


def assign_bins(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float).reshape(-1)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if v.size == 0:
        edges = np.array([-np.inf, np.inf], dtype=float)
        return np.zeros(0, dtype=int), edges
    if n_bins == 1:
        edges = np.array([-np.inf, np.inf], dtype=float)
        return np.zeros(v.size, dtype=int), edges

    # Quantile edges for low/mid/high regimes.
    qs = np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1))
    for i in range(1, len(qs)):
        if qs[i] <= qs[i - 1]:
            qs[i] = qs[i - 1] + 1e-12

    edges = np.concatenate(([-np.inf], qs[1:-1], [np.inf])).astype(float)
    bins = np.digitize(v, edges[1:-1], right=False)
    bins = np.clip(bins, 0, n_bins - 1).astype(int)
    return bins, edges


def cqr_scores(y_true: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    lo = np.asarray(q_lo, dtype=float).reshape(-1)
    hi = np.asarray(q_hi, dtype=float).reshape(-1)
    if not (y.size == lo.size == hi.size):
        raise ValueError("y_true, q_lo, q_hi must have the same length")
    return np.maximum(np.maximum(lo - y, y - hi), 0.0)


@dataclass
class RegimeCQR:
    cfg: RegimeCQRConfig
    edges: np.ndarray | None = None
    qhat_by_bin: np.ndarray | None = None

    def fit(self, y_cal: np.ndarray, q_lo_cal: np.ndarray, q_hi_cal: np.ndarray) -> dict[str, Any]:
        y_cal = np.asarray(y_cal, dtype=float).reshape(-1)
        if y_cal.size == 0:
            raise ValueError("y_cal must be non-empty")

        vol = rolling_volatility(y_cal, self.cfg.vol_window)
        bins, edges = assign_bins(vol, self.cfg.n_bins)
        scores = cqr_scores(y_cal, q_lo_cal, q_hi_cal)

        qhat = np.zeros(self.cfg.n_bins, dtype=float)
        global_q = float(np.quantile(scores, max(self.cfg.eps, 1.0 - self.cfg.alpha)))
        for b in range(self.cfg.n_bins):
            sb = scores[bins == b]
            if sb.size == 0:
                qhat[b] = global_q
            else:
                qhat[b] = float(np.quantile(sb, max(self.cfg.eps, 1.0 - self.cfg.alpha)))

        self.edges = edges
        self.qhat_by_bin = qhat
        return {
            "alpha": float(self.cfg.alpha),
            "n_bins": int(self.cfg.n_bins),
            "vol_window": int(self.cfg.vol_window),
            "edges": edges.tolist(),
            "qhat_by_bin": qhat.tolist(),
            "global_qhat": float(global_q),
        }

    def predict_interval(
        self,
        *,
        y_context: np.ndarray,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.edges is None or self.qhat_by_bin is None:
            raise RuntimeError("RegimeCQR is not fitted")

        context = np.asarray(y_context, dtype=float).reshape(-1)
        lo = np.asarray(q_lo, dtype=float).reshape(-1)
        hi = np.asarray(q_hi, dtype=float).reshape(-1)
        if lo.size != hi.size:
            raise ValueError("q_lo and q_hi must have the same length")
        if lo.size == 0:
            return lo, hi, np.zeros(0, dtype=int)

        vol = rolling_volatility(context, self.cfg.vol_window)
        bins = np.digitize(vol, self.edges[1:-1], right=False)
        bins = np.clip(bins, 0, self.cfg.n_bins - 1).astype(int)

        if bins.size == 1 and lo.size > 1:
            bins = np.full(lo.size, int(bins[0]), dtype=int)
        elif bins.size != lo.size:
            fallback_bin = int(bins[-1]) if bins.size else 0
            bins = np.full(lo.size, fallback_bin, dtype=int)

        adj = self.qhat_by_bin[bins]
        lower = lo - adj
        upper = hi + adj
        return lower.astype(float), upper.astype(float), bins

    def to_json(self) -> str:
        payload = {
            "cfg": {
                "alpha": float(self.cfg.alpha),
                "n_bins": int(self.cfg.n_bins),
                "vol_window": int(self.cfg.vol_window),
                "eps": float(self.cfg.eps),
                "fail_fast_quantile_backend": bool(self.cfg.fail_fast_quantile_backend),
            },
            "edges": None if self.edges is None else self.edges.tolist(),
            "qhat_by_bin": None if self.qhat_by_bin is None else self.qhat_by_bin.tolist(),
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    def from_json(data: str) -> "RegimeCQR":
        payload = json.loads(data)
        cfg = RegimeCQRConfig(**payload.get("cfg", {}))
        obj = RegimeCQR(cfg=cfg)
        if payload.get("edges") is not None:
            obj.edges = np.asarray(payload["edges"], dtype=float)
        if payload.get("qhat_by_bin") is not None:
            obj.qhat_by_bin = np.asarray(payload["qhat_by_bin"], dtype=float)
        return obj
