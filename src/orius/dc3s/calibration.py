"""Uncertainty inflation utilities for DC3S.

Public API
----------
DC3SConfig
    Typed configuration dataclass. Construct with :meth:`DC3SConfig.from_mapping`
    from a raw YAML-sourced dict.
build_uncertainty_set
    Main entry point: builds prediction intervals adjusted for telemetry
    reliability, drift, and sensitivity.
build_uncertainty_set_kappa
    Explicit kappa-inflation variant (secondary law).
inflate_q / inflate_interval
    Low-level primitives used by tests and the forecasting layer.
calibrate_ambiguity_lambda
    One-shot ambiguity parameter calibration from residual quantiles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .ambiguity import AmbiguityConfig, widen_bounds
from .rac_cert import RACCertConfig, compute_q_multiplier, normalize_sensitivity

__all__ = [
    "DC3SConfig",
    "build_uncertainty_set",
    "build_uncertainty_set_kappa",
    "inflate_q",
    "inflate_interval",
    "calibrate_ambiguity_lambda",
]


def _as_1d(value: float | list[float] | np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{label} must be non-empty")
    return arr


def _cfg_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Try each key in order, returning the first match or the default.

    Use this instead of nested ``dict.get(key1, dict.get(key2, default))``
    chains, which are hard to read and silently skip misnamed keys.
    """
    for key in keys:
        if key in d:
            return d[key]
    return default


def _scalar_or_zero(arr: Any) -> float:
    """Return the first scalar element of an array-like, or 0.0 if empty."""
    a = np.asarray(arr, dtype=float).reshape(-1)
    return float(a[0]) if a.size > 0 else 0.0


def inflate_q(q: float | list[float] | np.ndarray, inflation: float) -> np.ndarray:
    """Scale a conformal quantile array by *inflation*."""
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
    """Return a lambda_mw value from the *quantile* of absolute residuals.

    Used for one-shot ambiguity parameter calibration from a held-out
    calibration set.  The result is the ``q``-th percentile of
    ``abs(residuals_mw)``, scaled and clipped to ``[min_lambda, max_lambda]``.
    """
    residuals = np.abs(_as_1d(residuals_mw, "residuals_mw"))
    q = float(np.clip(float(quantile), 1e-6, 1.0))
    lam = float(np.quantile(residuals, q)) * float(scale)
    lam = max(float(min_lambda), lam)
    if max_lambda is not None:
        lam = min(float(max_lambda), lam)
    return float(lam)




@dataclass
class DC3SConfig:
    """Typed configuration for :func:`build_uncertainty_set`.

    Replaces the raw ``Mapping[str, Any]`` dict and its 15+ individual
    ``.get()`` calls.  Renaming a YAML key now produces an ``AttributeError``
    at parse time rather than silently using the wrong default at run time.

    Construct via :meth:`from_mapping` to support the existing config-dict
    calling convention without changing any call sites.
    """

    # --- Inflation law ---
    law: str = "linear"
    k_quality: float = 0.8
    k_drift: float = 0.6
    k_sensitivity: float = 0.0
    infl_max: float = 2.0
    cooldown_smoothing: float = 0.0

    # --- Reliability ---
    min_w: float = 0.05

    # --- Sensitivity / staleness ---
    sensitivity_t: float = 0.0
    sensitivity_norm: float | None = None
    staleness_counter: float | None = None

    # --- FTIT-RO runtime values (populated by runner, not YAML) ---
    ftit_sigma2: float | None = None
    ftit_delta: float = 0.05
    ftit_eps_interval: float = 1.0e-6

    # --- RAC-Cert sub-config (kept as a raw dict for RACCertConfig construction) ---
    rac_cert_raw: dict = field(default_factory=dict)

    # --- FTIT sub-config (kept as a raw dict for FTIT helpers) ---
    ftit_raw: dict = field(default_factory=dict)

    # --- Ambiguity sub-config ---
    ambiguity_raw: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "DC3SConfig":
        """Parse a raw config mapping into a typed DC3SConfig.

        Supports both canonical keys (``k_quality``) and the legacy ``k_q``
        alias so that older callers continue to work without changes.
        """
        d = dict(cfg)
        reliability_raw = dict(d.get("reliability", {}))
        rac_raw = dict(d.get("rac_cert", {})) if isinstance(d.get("rac_cert"), Mapping) else {}
        ftit_raw = dict(d.get("ftit", {})) if isinstance(d.get("ftit"), Mapping) else {}
        ambiguity_raw = dict(d.get("ambiguity", {}))
        ftit_runtime = dict(d.get("ftit_runtime", {})) if isinstance(d.get("ftit_runtime"), Mapping) else {}

        # Resolve sigma2 for ftit_ro law in one place (was previously a 6-level
        # nested expression inside the meta dict).
        sigma2_floor = max(float(ftit_raw.get("sigma2_floor", 1.0e-6)), 1.0e-12)
        sigma2_init = float(ftit_raw.get("sigma2_init", 1.0))
        ftit_sigma2 = max(float(ftit_runtime.get("sigma2", sigma2_init)), sigma2_floor)

        return cls(
            law=str(d.get("law", "linear")).strip().lower(),
            k_quality=float(d.get("k_quality", d.get("k_q", 0.8))),
            k_drift=float(d.get("k_drift", 0.6)),
            k_sensitivity=float(d.get("k_sensitivity", rac_raw.get("k_sensitivity", 0.0))),
            infl_max=float(d.get("infl_max", rac_raw.get("infl_max", 2.0))),
            cooldown_smoothing=float(d.get("cooldown_smoothing", 0.0)),
            min_w=float(reliability_raw.get("min_w", 0.05)),
            sensitivity_t=float(_cfg_get(d, "sensitivity_t", "runtime_sensitivity_t", default=0.0)),
            sensitivity_norm=_cfg_get(d, "sensitivity_norm", "runtime_sensitivity_norm"),
            staleness_counter=_cfg_get(d, "staleness_counter", "runtime_staleness_counter"),
            ftit_sigma2=ftit_sigma2,
            ftit_delta=float(ftit_raw.get("delta", 0.05)),
            ftit_eps_interval=float(ftit_raw.get("eps_interval", 1.0e-6)),
            rac_cert_raw=rac_raw,
            ftit_raw=ftit_raw,
            ambiguity_raw=ambiguity_raw,
        )


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
    # Parse the raw config dict into a typed object so that misspelled or
    # missing YAML keys are caught at parse time, not silently defaulted.
    dc = DC3SConfig.from_mapping(cfg)
    law = dc.law
    k_q = dc.k_quality
    k_d = dc.k_drift
    k_s = dc.k_sensitivity
    infl_max = dc.infl_max
    min_w = dc.min_w
    rac_cfg_raw = dc.rac_cert_raw
    ftit_cfg = dc.ftit_raw

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
        max_q_multiplier=float(rac_cfg_raw.get("max_q_multiplier", 2.0)),
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

    # Resolve sensitivity normalisation value from the typed config.
    # PDF §4: staleness counter takes priority over the dispatch-sensitivity probe.
    sensitivity_t = dc.sensitivity_t
    sensitivity_norm = dc.sensitivity_norm
    staleness_counter = dc.staleness_counter
    if staleness_counter is not None:
        sensitivity_norm_val = float(np.clip(float(staleness_counter), 0.0, 1.0))
    elif sensitivity_norm is None:
        sensitivity_norm_val = normalize_sensitivity(sensitivity_t, norm_ref=rac_cfg.sens_norm_ref)
    else:
        sensitivity_norm_val = float(np.clip(float(sensitivity_norm), 0.0, 1.0))

    q_multiplier, q_meta = compute_q_multiplier(
        w_t=float(w_t),
        sensitivity_norm=float(sensitivity_norm_val),
        cfg=rac_cfg,
    )

    if law == "ftit_ro":
        sigma2 = dc.ftit_sigma2  # already clamped to sigma2_floor in from_mapping
        delta = dc.ftit_delta
        eps_interval = dc.ftit_eps_interval
        interval_abs = np.maximum(2.0 * q_arr, eps_interval)
        log_term = math.log(1.0 / max(delta * w_eff, 1.0e-12))
        kappa = 1.0 + np.sqrt(2.0 * sigma2 * log_term) / interval_abs
        lower = yhat_arr - q_arr * kappa
        upper = yhat_arr + q_arr * kappa
        q_eff = q_arr
        infl_raw = float(np.asarray(kappa, dtype=float).reshape(-1)[0])
        inflation = infl_raw
        q_multiplier = 1.0
    else:
        infl_raw = 1.0 + k_q * (1.0 - w_eff) + k_d * drift_term + k_s * float(sensitivity_norm_val)
        inflation = float(np.clip(infl_raw, 1.0, infl_max))

        smoothing = dc.cooldown_smoothing
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

    ambiguity_cfg = dc.ambiguity_raw
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

    # --- Extract meta values into named variables before building the dict ---
    # (Avoids multi-level nested expressions inside dict literals which are
    # hard to read and easy to get wrong when sigma2_floor changes.)
    interval_width_meta = float(max(0.0, upper[0] - lower[0])) if len(lower) else 0.0
    q_eff_meta = _scalar_or_zero(q_eff)
    if law == "ftit_ro":
        # Reuse sigma2_floor / sigma2 computed above in the ftit_ro branch
        meta_delta = float(ftit_cfg.get("delta", 0.05))
        meta_sigma2 = float(sigma2)  # already clamped to sigma2_floor above
    else:
        meta_delta = None
        meta_sigma2 = None

    meta = {
        "w_t_raw": float(w_t),
        "w_t_used": float(w_eff),
        "w_t": float(w_t),
        "w_used": float(w_eff),
        "delta_mw": float(widen_meta.get("delta_mw", 0.0)),
        "drift_flag": bool(drift_flag),
        "inflation_raw": float(infl_raw),
        "inflation": float(inflation),
        "infl_max": float(infl_max),
        "k_quality": float(k_q),
        "k_drift": float(k_d),
        "k_sensitivity": float(k_s),
        "lambda_mw_used": float(amb.lambda_mw),
        "interval_width": interval_width_meta,
        "sensitivity_t": float(sensitivity_t),
        "sensitivity_norm": float(sensitivity_norm_val),
        "q_eff": q_eff_meta,
        "q_multiplier": float(q_multiplier),
        "delta": meta_delta,
        "sigma2": meta_sigma2,
        "inflation_rule": law,
        "inflation_components": {
            "quality": float(k_q * (1.0 - w_eff)) if law != "ftit_ro" else 0.0,
            "drift": float(k_d * drift_term) if law != "ftit_ro" else 0.0,
            "sensitivity": float(k_s * float(sensitivity_norm_val)) if law != "ftit_ro" else 0.0,
            "q_multiplier_raw": float(q_meta.get("q_multiplier_raw", q_multiplier)) if law != "ftit_ro" else 1.0,
        },
    }
    return lower, upper, meta


def build_uncertainty_set_kappa(
    yhat: float | list[float] | np.ndarray,
    q: float | list[float] | np.ndarray,
    w_t: float,
    drift_flag: bool,
    cfg: Mapping[str, Any],
    sigma_sq: float,
    delta: float = 0.10,
    eps_floor: float = 50.0,
    prev_inflation: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a DC3S uncertainty set using the explicit kappa inflation rule."""
    dcfg = dict(cfg)
    reliability_cfg = dict(dcfg.get("reliability", {}))
    min_w = float(reliability_cfg.get("min_w", 0.05))
    infl_max = float(dcfg.get("infl_max", 3.0))
    drift_penalty = float(dcfg.get("kappa_drift_penalty", 0.5))

    yhat_arr = _as_1d(yhat, "yhat")
    q_arr = _as_1d(q, "q")
    if q_arr.size == 1 and yhat_arr.size > 1:
        q_arr = np.full(yhat_arr.size, float(q_arr[0]), dtype=float)
    if q_arr.size != yhat_arr.size:
        raise ValueError("q must have size 1 or match yhat length")

    w_t_raw = float(w_t)
    w_eff = max(min_w, w_t_raw)
    if bool(drift_flag):
        w_eff = max(min_w, w_eff * drift_penalty)

    sigma_sq_used = max(float(sigma_sq), 0.0)
    delta_used = float(delta)
    eps_used = max(float(eps_floor), 1.0e-12)
    u_width = 2.0 * q_arr
    denom = np.maximum(u_width, eps_used)
    log_term = np.log(1.0 / max(delta_used * w_eff, 1.0e-12))
    kappa_raw = 1.0 + np.sqrt(2.0 * sigma_sq_used * log_term) / denom
    kappa = np.clip(kappa_raw, 1.0, infl_max)

    smoothing = float(dcfg.get("cooldown_smoothing", 0.0))
    if prev_inflation is not None and 0.0 < smoothing < 1.0:
        kappa = np.clip(smoothing * float(prev_inflation) + (1.0 - smoothing) * kappa, 1.0, infl_max)

    lower = yhat_arr - q_arr * kappa
    upper = yhat_arr + q_arr * kappa
    meta = {
        "inflation_law": "kappa",
        "inflation_rule": "kappa",
        "kappa": float(kappa[0]),
        "w_t_raw": float(w_t_raw),
        "w_t_used": float(w_eff),
        "sigma_sq": float(sigma_sq_used),
        "delta_t": float(delta_used),
        "U_half": float(q_arr[0]),
        "drift_flag": bool(drift_flag),
        "inflation": float(kappa[0]),
        "inflation_raw": float(kappa_raw[0]),
        "w_t": float(w_t_raw),
        "w_used": float(w_eff),
        "interval_width": float(max(0.0, upper[0] - lower[0])) if len(lower) else 0.0,
        "delta_mw": 0.0,
        "infl_max": float(infl_max),
        "q_eff": float(q_arr[0]),
        "q_multiplier": 1.0,
    }
    return lower, upper, meta
