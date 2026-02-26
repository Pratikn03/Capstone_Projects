"""Uncertainty inflation utilities for DC3S."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .ambiguity import AmbiguityConfig, widen_bounds
from .rac_cert import RACCertConfig, compute_q_multiplier, normalize_sensitivity


def _as_1d(value: float | list[float] | np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{label} must be non-empty")
    return arr


def inflate_q(q: float | list[float] | np.ndarray, inflation: float) -> np.ndarray:
    q_arr = _as_1d(q, "q")
    return q_arr * float(inflation)


def inflate_interval(
    lower: float | list[float] | np.ndarray,
    upper: float | list[float] | np.ndarray,
    inflation: float,
) -> tuple[np.ndarray, np.ndarray]:
    lo = _as_1d(lower, "lower")
    hi = _as_1d(upper, "upper")
    if lo.size != hi.size:
        raise ValueError("lower and upper must have the same length")
    if np.any(lo > hi):
        raise ValueError("lower cannot exceed upper")
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    half_new = half * float(inflation)
    return mid - half_new, mid + half_new


def calibrate_ambiguity_lambda(
    residuals_mw: float | list[float] | np.ndarray,
    *,
    quantile: float = 0.95,
    scale: float = 1.0,
    min_lambda: float = 0.0,
    max_lambda: float | None = None,
) -> float:
    residuals = np.abs(_as_1d(residuals_mw, "residuals_mw"))
    q = float(np.clip(float(quantile), 1e-6, 1.0))
    lam = float(np.quantile(residuals, q)) * float(scale)
    lam = max(float(min_lambda), lam)
    if max_lambda is not None:
        lam = min(float(max_lambda), lam)
    return float(lam)


def build_uncertainty_set(
    yhat: float | list[float] | np.ndarray,
    q: float | list[float] | np.ndarray,
    w_t: float,
    drift_flag: bool,
    cfg: Mapping[str, Any],
    prev_inflation: float | None = None,
    base_lower: float | list[float] | np.ndarray | None = None,
    base_upper: float | list[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Build DC3S uncertainty set with the locked linear law:

    infl = clip(1 + k_q*(1 - w_t) + k_d*1[drift], 1, infl_max)
    lower = yhat - q*infl
    upper = yhat + q*infl
    """
    dcfg = dict(cfg)
    reliability_cfg = dict(dcfg.get("reliability", {}))
    rac_cfg_raw = dict(dcfg.get("rac_cert", {})) if isinstance(dcfg.get("rac_cert"), Mapping) else {}

    k_q = float(dcfg.get("k_quality", dcfg.get("k_q", 0.8)))
    k_d = float(dcfg.get("k_drift", 0.6))
    infl_max = float(dcfg.get("infl_max", rac_cfg_raw.get("infl_max", 3.0)))
    min_w = float(reliability_cfg.get("min_w", 0.05))
    k_s = float(dcfg.get("k_sensitivity", rac_cfg_raw.get("k_sensitivity", 0.0)))

    rac_cfg = RACCertConfig(
        alpha=float(rac_cfg_raw.get("alpha", 0.10)),
        n_vol_bins=int(rac_cfg_raw.get("n_vol_bins", 3)),
        vol_window=int(rac_cfg_raw.get("vol_window", 24)),
        beta_reliability=float(rac_cfg_raw.get("beta_reliability", 0.0)),
        beta_sensitivity=float(rac_cfg_raw.get("beta_sensitivity", 0.0)),
        k_sensitivity=float(k_s),
        infl_max=float(infl_max),
        sens_eps_mw=float(rac_cfg_raw.get("sens_eps_mw", 25.0)),
        sens_norm_ref=float(rac_cfg_raw.get("sens_norm_ref", 0.5)),
        qhat_shrink_tau=float(rac_cfg_raw.get("qhat_shrink_tau", 30.0)),
        max_q_multiplier=float(rac_cfg_raw.get("max_q_multiplier", 3.0)),
        min_w=float(rac_cfg_raw.get("min_w", min_w)),
        eps=float(rac_cfg_raw.get("eps", 1e-9)),
    )

    yhat_arr = _as_1d(yhat, "yhat")
    q_arr = _as_1d(q, "q")
    if q_arr.size == 1 and yhat_arr.size > 1:
        q_arr = np.full(yhat_arr.size, float(q_arr[0]), dtype=float)
    if q_arr.size != yhat_arr.size:
        raise ValueError("q must have size 1 or match yhat length")

    w_eff = max(float(w_t), min_w)
    drift_term = 1.0 if bool(drift_flag) else 0.0
    sensitivity_t = float(dcfg.get("sensitivity_t", dcfg.get("runtime_sensitivity_t", 0.0)))
    sensitivity_norm = dcfg.get("sensitivity_norm", dcfg.get("runtime_sensitivity_norm"))
    if sensitivity_norm is None:
        sensitivity_norm_val = normalize_sensitivity(sensitivity_t, norm_ref=rac_cfg.sens_norm_ref)
    else:
        sensitivity_norm_val = float(np.clip(float(sensitivity_norm), 0.0, 1.0))

    q_multiplier, q_meta = compute_q_multiplier(
        w_t=float(w_t),
        sensitivity_norm=float(sensitivity_norm_val),
        cfg=rac_cfg,
    )

    infl_raw = 1.0 + k_q * (1.0 - w_eff) + k_d * drift_term + k_s * float(sensitivity_norm_val)
    inflation = float(np.clip(infl_raw, 1.0, infl_max))

    smoothing = float(dcfg.get("cooldown_smoothing", 0.0))
    if prev_inflation is not None and 0.0 < smoothing < 1.0:
        inflation = float(np.clip(smoothing * float(prev_inflation) + (1.0 - smoothing) * inflation, 1.0, infl_max))

    if base_lower is not None or base_upper is not None:
        if base_lower is None or base_upper is None:
            raise ValueError("base_lower and base_upper must be provided together")
        base_lo = _as_1d(base_lower, "base_lower")
        base_hi = _as_1d(base_upper, "base_upper")
        if base_lo.size == 1 and yhat_arr.size > 1:
            base_lo = np.full(yhat_arr.size, float(base_lo[0]), dtype=float)
        if base_hi.size == 1 and yhat_arr.size > 1:
            base_hi = np.full(yhat_arr.size, float(base_hi[0]), dtype=float)
        if base_lo.size != yhat_arr.size or base_hi.size != yhat_arr.size:
            raise ValueError("base_lower/base_upper must have size 1 or match yhat length")
        base_mid = 0.5 * (base_lo + base_hi)
        base_half = 0.5 * (base_hi - base_lo)
        rac_half = base_half * float(q_multiplier)
        pre_infl_lo = base_mid - rac_half
        pre_infl_hi = base_mid + rac_half
        lower, upper = inflate_interval(pre_infl_lo, pre_infl_hi, inflation)
        q_eff = rac_half
    else:
        q_eff = q_arr * float(q_multiplier)
        half_width = q_eff * inflation
        lower = yhat_arr - half_width
        upper = yhat_arr + half_width

    ambiguity_cfg = dict(dcfg.get("ambiguity", {}))
    lambda_mw = float(ambiguity_cfg.get("lambda_mw", 0.0))
    if lambda_mw <= 0.0 and bool(ambiguity_cfg.get("learn_lambda_from_quantile", False)):
        residual_source = ambiguity_cfg.get("calibration_residuals_mw")
        if residual_source is None:
            residual_source = q_arr
        lambda_mw = calibrate_ambiguity_lambda(
            residuals_mw=residual_source,
            quantile=float(ambiguity_cfg.get("lambda_quantile", 0.95)),
            scale=float(ambiguity_cfg.get("lambda_scale", 1.0)),
            min_lambda=float(ambiguity_cfg.get("lambda_min_mw", 0.0)),
            max_lambda=float(ambiguity_cfg["lambda_max_mw"]) if "lambda_max_mw" in ambiguity_cfg else None,
        )
    amb = AmbiguityConfig(
        lambda_mw=float(lambda_mw),
        min_w=float(ambiguity_cfg.get("min_w", min_w)),
        max_extra=float(ambiguity_cfg.get("max_extra", 1.0)),
    )
    widen_meta = {"w_t": float(w_t), "w_used": float(w_eff), "delta_mw": 0.0}
    if amb.lambda_mw > 0.0:
        lower, upper, widen_meta = widen_bounds(lower=lower, upper=upper, w_t=float(w_t), cfg=amb)

    meta = {
        "w_t_raw": float(w_t),
        "w_t_used": float(w_eff),
        "w_t": float(w_t),
        "delta_mw": float(widen_meta.get("delta_mw", 0.0)),
        "drift_flag": bool(drift_flag),
        "inflation_raw": float(infl_raw),
        "inflation": float(inflation),
        "infl_max": float(infl_max),
        "k_quality": float(k_q),
        "k_drift": float(k_d),
        "k_sensitivity": float(k_s),
        "lambda_mw_used": float(amb.lambda_mw),
        "interval_width": float(max(0.0, upper[0] - lower[0])) if len(lower) else 0.0,
        "sensitivity_t": float(sensitivity_t),
        "sensitivity_norm": float(sensitivity_norm_val),
        "q_eff": float(np.asarray(q_eff, dtype=float)[0]) if len(np.asarray(q_eff).reshape(-1)) else 0.0,
        "q_multiplier": float(q_multiplier),
        "inflation_components": {
            "quality": float(k_q * (1.0 - w_eff)),
            "drift": float(k_d * drift_term),
            "sensitivity": float(k_s * float(sensitivity_norm_val)),
            "q_multiplier_raw": float(q_meta.get("q_multiplier_raw", q_multiplier)),
        },
    }
    return lower, upper, meta
