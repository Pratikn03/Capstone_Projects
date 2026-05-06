"""RAC-Cert utilities for reliability-adaptive conformal uncertainty.

Public API
----------
RACCertConfig
    Calibration hyper-parameters (alpha, vol bins, sensitivity weights, etc.).
RACCertModel
    Volatility-binned conformal quantile model.  Fit on a calibration split,
    then serialise/deserialise with ``to_json`` / ``from_json``.
compute_q_multiplier
    Scale factor for the conformal threshold based on w_t and sensitivity.
compute_inflation
    Inflation factor per the DC3S linear law.
compute_dispatch_sensitivity
    Numerical perturbation-based dispatch sensitivity probe.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from orius.forecasting.uncertainty.cqr import assign_bins, cqr_scores, rolling_volatility

__all__ = [
    "RACCertConfig",
    "RACCertModel",
    "compute_dispatch_sensitivity",
    "compute_inflation",
    "compute_q_multiplier",
    "normalize_sensitivity",
]


@dataclass
class RACCertConfig:
    alpha: float = 0.10
    n_vol_bins: int = 3
    vol_window: int = 24
    beta_reliability: float = 0.5
    beta_sensitivity: float = 0.3
    k_sensitivity: float = 0.05
    infl_max: float = 2.0
    sens_eps_mw: float = 25.0
    sens_norm_ref: float = 0.5
    qhat_shrink_tau: float = 30.0
    max_q_multiplier: float = 2.0
    min_w: float = 0.05
    eps: float = 1e-9


@dataclass
class RACCertModel:
    cfg: RACCertConfig
    vol_edges: np.ndarray | None = None
    qhat_by_vol_bin: np.ndarray | None = None
    global_qhat: float = 0.0
    fit_meta: dict[str, Any] | None = None

    def fit(self, y_cal: np.ndarray, q_lo_cal: np.ndarray, q_hi_cal: np.ndarray) -> dict[str, Any]:
        y = np.asarray(y_cal, dtype=float).reshape(-1)
        lo = np.asarray(q_lo_cal, dtype=float).reshape(-1)
        hi = np.asarray(q_hi_cal, dtype=float).reshape(-1)
        if not (len(y) == len(lo) == len(hi)):
            raise ValueError("y_cal, q_lo_cal, q_hi_cal must have the same length")
        if len(y) == 0:
            raise ValueError("Calibration arrays must be non-empty")

        vol = rolling_volatility(y, self.cfg.vol_window)
        bins, edges = assign_bins(vol, self.cfg.n_vol_bins)
        scores = cqr_scores(y, lo, hi)

        global_q = float(np.quantile(scores, max(self.cfg.eps, 1.0 - self.cfg.alpha)))
        qhat = np.zeros(self.cfg.n_vol_bins, dtype=float)
        counts = np.zeros(self.cfg.n_vol_bins, dtype=int)

        tau = max(float(self.cfg.qhat_shrink_tau), 0.0)
        for b in range(self.cfg.n_vol_bins):
            sb = scores[bins == b]
            n = int(sb.size)
            counts[b] = n
            q_bin = global_q if n == 0 else float(np.quantile(sb, max(self.cfg.eps, 1.0 - self.cfg.alpha)))
            w_local = float(n / (n + tau)) if tau > 0.0 else 1.0
            qhat[b] = w_local * q_bin + (1.0 - w_local) * global_q

        self.vol_edges = edges.astype(float)
        self.qhat_by_vol_bin = qhat.astype(float)
        self.global_qhat = float(global_q)
        self.fit_meta = {
            "alpha": float(self.cfg.alpha),
            "n_vol_bins": int(self.cfg.n_vol_bins),
            "vol_window": int(self.cfg.vol_window),
            "global_qhat": float(global_q),
            "qhat_by_vol_bin": qhat.tolist(),
            "vol_edges": edges.tolist(),
            "bin_counts": counts.tolist(),
            "qhat_shrink_tau": float(self.cfg.qhat_shrink_tau),
        }
        return dict(self.fit_meta)

    def _require_fitted(self) -> None:
        if self.vol_edges is None or self.qhat_by_vol_bin is None:
            raise RuntimeError("RACCertModel is not fitted")

    def assign_context_bins(self, y_context: np.ndarray) -> np.ndarray:
        self._require_fitted()
        y = np.asarray(y_context, dtype=float).reshape(-1)
        vol = rolling_volatility(y, self.cfg.vol_window)
        if self.vol_edges is None:  # guaranteed by _require_fitted, but explicit for type checkers
            raise RuntimeError("RACCertModel vol_edges is None after fit — this is a bug")
        bins = np.digitize(vol, self.vol_edges[1:-1], right=False)
        bins = np.clip(bins, 0, self.cfg.n_vol_bins - 1).astype(int)
        return bins

    def qhat_for_context(self, y_context: np.ndarray, horizon: int) -> np.ndarray:
        bins = self.assign_context_bins(y_context)
        if bins.size == 0:
            return np.zeros(horizon, dtype=float)
        if bins.size == 1 and horizon > 1:
            bins = np.full(horizon, int(bins[0]), dtype=int)
        elif bins.size != horizon:
            bins = np.full(horizon, int(bins[-1]), dtype=int)
        if self.qhat_by_vol_bin is None:  # guaranteed by _require_fitted, but explicit for type checkers
            raise RuntimeError("RACCertModel qhat_by_vol_bin is None after fit — this is a bug")
        return self.qhat_by_vol_bin[bins].astype(float)

    def to_json(self) -> str:
        payload = {
            "cfg": asdict(self.cfg),
            "vol_edges": None if self.vol_edges is None else self.vol_edges.tolist(),
            "qhat_by_vol_bin": None if self.qhat_by_vol_bin is None else self.qhat_by_vol_bin.tolist(),
            "global_qhat": float(self.global_qhat),
            "fit_meta": self.fit_meta or {},
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @staticmethod
    def from_json(s: str) -> RACCertModel:
        payload = json.loads(s)
        cfg = RACCertConfig(**payload.get("cfg", {}))
        model = RACCertModel(cfg=cfg)
        if payload.get("vol_edges") is not None:
            model.vol_edges = np.asarray(payload["vol_edges"], dtype=float)
        if payload.get("qhat_by_vol_bin") is not None:
            model.qhat_by_vol_bin = np.asarray(payload["qhat_by_vol_bin"], dtype=float)
        model.global_qhat = float(payload.get("global_qhat", 0.0))
        fit_meta = payload.get("fit_meta")
        model.fit_meta = fit_meta if isinstance(fit_meta, dict) else {}
        return model


def normalize_sensitivity(sensitivity: float, norm_ref: float) -> float:
    ref = max(float(norm_ref), 1e-9)
    return float(np.clip(float(sensitivity) / ref, 0.0, 1.0))


def compute_q_multiplier(
    *,
    w_t: float,
    sensitivity_norm: float,
    cfg: RACCertConfig,
) -> tuple[float, dict[str, float]]:
    w_used = max(float(cfg.min_w), float(w_t))
    sens = float(np.clip(float(sensitivity_norm), 0.0, 1.0))
    q_mult_raw = 1.0 + float(cfg.beta_reliability) * (1.0 - w_used) + float(cfg.beta_sensitivity) * sens
    q_mult = float(np.clip(q_mult_raw, 1.0, float(cfg.max_q_multiplier)))
    return q_mult, {"w_used": float(w_used), "q_multiplier_raw": float(q_mult_raw)}


def compute_inflation(
    *,
    w_t: float,
    drift_flag: bool,
    sensitivity_norm: float,
    k_quality: float,
    k_drift: float,
    cfg: RACCertConfig,
) -> tuple[float, dict[str, float]]:
    w_used = max(float(cfg.min_w), float(w_t))
    sens = float(np.clip(float(sensitivity_norm), 0.0, 1.0))
    quality_component = float(k_quality) * (1.0 - w_used)
    drift_component = float(k_drift) * (1.0 if drift_flag else 0.0)
    sensitivity_component = float(cfg.k_sensitivity) * sens
    raw = 1.0 + quality_component + drift_component + sensitivity_component
    inflation = float(np.clip(raw, 1.0, float(cfg.infl_max)))
    return inflation, {
        "quality": float(quality_component),
        "drift": float(drift_component),
        "sensitivity": float(sensitivity_component),
        "inflation_raw": float(raw),
        "w_used": float(w_used),
    }


def compute_dispatch_sensitivity(
    *,
    load_window: np.ndarray,
    dispatch_probe: Callable[[np.ndarray], tuple[float, float]],
    sens_eps_mw: float,
) -> float:
    load = np.asarray(load_window, dtype=float).reshape(-1)
    if load.size == 0:
        return 0.0
    eps_raw = float(sens_eps_mw)
    eps = max(eps_raw, 1e-6) if np.isfinite(eps_raw) else 1e-6

    plus = load.copy()
    minus = load.copy()
    plus[0] += eps
    minus[0] = max(0.0, minus[0] - eps)

    ch_plus, dis_plus = dispatch_probe(plus)
    ch_minus, dis_minus = dispatch_probe(minus)
    net_plus = float(dis_plus) - float(ch_plus)
    net_minus = float(dis_minus) - float(ch_minus)
    sens = abs(net_plus - net_minus) / (2.0 * eps)
    return float(max(0.0, sens))
