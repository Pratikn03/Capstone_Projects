"""DC3S one-step pipeline orchestration.

Provides ``run_dc3s_step`` — a single entry point that executes all five
DC3S stages in order:

    1. Detect  (OQE / reliability scoring)
    2. Calibrate (RAC-Cert uncertainty inflation)
    3. Constrain (FTIT / safety-filter tightening)
    4. Shield  (L2 projection repair)
    5. Certify (dispatch certificate)
"""
from __future__ import annotations

import uuid
from typing import Any, Mapping

import numpy as np

from .quality import compute_reliability
from .calibration import build_uncertainty_set
from .drift import PageHinkleyDetector
from .shield import repair_action
from .certificate import make_certificate, compute_config_hash
from .safety_filter_theory import tightened_soc_bounds, reliability_error_bound


def run_dc3s_step(
    *,
    event: Mapping[str, Any],
    last_event: Mapping[str, Any] | None,
    yhat: float | list[float] | np.ndarray,
    q: float | list[float] | np.ndarray,
    candidate_action: Mapping[str, Any],
    domain_adapter: Any,
    state: Any,
    drift_detector: PageHinkleyDetector | None = None,
    residual: float | None = None,
    cfg: Mapping[str, Any] | None = None,
    adaptive_state: Mapping[str, Any] | None = None,
    prev_inflation: float | None = None,
    prev_cert_hash: str | None = None,
    model_hash: str = "",
    device_id: str = "battery-0",
    zone_id: str = "zone-0",
    controller: str = "dc3s",
) -> dict[str, Any]:
    """Execute one full DC3S step and return the dispatch certificate + metadata.

    Returns a dict with keys:
        certificate, safe_action, shield_meta, reliability_w, drift_flag,
        inflation, uncertainty_set, lower, upper, tightened_soc, meta.
    """
    dcfg = dict(cfg or {})
    reliability_cfg = dcfg.get("reliability")
    ftit_cfg = dcfg.get("ftit")
    expected_cadence = float(dcfg.get("expected_cadence_s", 3600))
    assumptions_version = str(dcfg.get("assumptions_version", "dc3s-assumptions-v1"))

    # ── Stage 1: Detect (OQE) ────────────────────────────────────────
    w_t, reliability_flags = compute_reliability(
        event=event,
        last_event=last_event,
        expected_cadence_s=expected_cadence,
        reliability_cfg=reliability_cfg,
        adaptive_state=adaptive_state,
        ftit_cfg=ftit_cfg,
    )

    # ── Drift detection ──────────────────────────────────────────────
    drift_flag = False
    drift_meta: dict[str, Any] = {"drift": False, "score": 0.0}
    if drift_detector is not None and residual is not None:
        drift_meta = drift_detector.update(abs(float(residual)))
        drift_flag = bool(drift_meta.get("drift", False))

    # ── Stage 2: Calibrate (RAC-Cert inflation) ──────────────────────
    lower, upper, cal_meta = build_uncertainty_set(
        yhat=yhat,
        q=q,
        w_t=float(w_t),
        drift_flag=drift_flag,
        cfg=dcfg,
        prev_inflation=prev_inflation,
    )
    inflation = float(cal_meta.get("inflation", 1.0))

    uncertainty_set: dict[str, Any] = {
        "lower": lower,
        "upper": upper,
        "meta": {
            "w_t": float(w_t),
            "drift_flag": drift_flag,
            "inflation": inflation,
        },
    }

    # Extract the RAC effective quantile scalar from the calibration meta dict.
    # cal_meta["q_eff"] may be a scalar float or a 1-D numpy/list array depending
    # on whether the RAC-Cert multiplier was applied before or after the inflation.
    _q_eff_raw = cal_meta.get("q_eff_scalar") or cal_meta.get("q_eff", 0.0)
    if isinstance(_q_eff_raw, (list, np.ndarray)):
        _q_eff_raw = _q_eff_raw[0] if len(_q_eff_raw) else 0.0
    q_rac_mwh = float(_q_eff_raw)

    if hasattr(state, "current_soc_mwh"):
        constraints_for_ftit = {
            "min_soc_mwh": getattr(state, "min_soc_mwh", 0.0),
            "max_soc_mwh": getattr(state, "max_soc_mwh", getattr(state, "capacity_mwh", 10000.0)),
            "capacity_mwh": getattr(state, "capacity_mwh", 10000.0),
        }
        err_bound = reliability_error_bound(
            reliability_w=float(w_t),
            max_error_mwh=float(constraints_for_ftit.get("capacity_mwh", 10000.0)) * 0.01,
        )
        ftit_min, ftit_max = tightened_soc_bounds(
            min_soc_mwh=float(constraints_for_ftit["min_soc_mwh"]),
            max_soc_mwh=float(constraints_for_ftit["max_soc_mwh"]),
            error_bound_mwh=err_bound,
            q_rac_mwh=q_rac_mwh if q_rac_mwh > 0 else None,
        )
        uncertainty_set["ftit_soc_min_mwh"] = ftit_min
        uncertainty_set["ftit_soc_max_mwh"] = ftit_max

    # ── Stage 3 + 4: Constrain + Shield ──────────────────────────────
    safe_action, shield_meta = repair_action(
        a_star=candidate_action,
        uncertainty_set=uncertainty_set,
        domain_adapter=domain_adapter,
        state=state,
    )

    # ── Stage 5: Certify ─────────────────────────────────────────────
    command_id = str(uuid.uuid4())
    config_bytes = str(dcfg).encode("utf-8")
    certificate = make_certificate(
        command_id=command_id,
        device_id=device_id,
        zone_id=zone_id,
        controller=controller,
        proposed_action=dict(candidate_action),
        safe_action=dict(safe_action),
        uncertainty=cal_meta,
        reliability=reliability_flags,
        drift=drift_meta,
        model_hash=model_hash,
        config_hash=compute_config_hash(config_bytes),
        prev_hash=prev_cert_hash,
        intervened=bool(shield_meta.get("repaired", False)),
        intervention_reason=shield_meta.get("mode", "projection"),
        reliability_w=float(w_t),
        drift_flag=drift_flag,
        inflation=inflation,
        assumptions_version=assumptions_version,
    )

    return {
        "certificate": certificate,
        "safe_action": dict(safe_action),
        "shield_meta": shield_meta,
        "reliability_w": float(w_t),
        "reliability_flags": reliability_flags,
        "drift_flag": drift_flag,
        "drift_meta": drift_meta,
        "inflation": inflation,
        "uncertainty_set": uncertainty_set,
        "lower": lower,
        "upper": upper,
        "calibration_meta": cal_meta,
    }
