"""Universal DC3S pipeline for any domain adapter.

Five stages (thesis Ch 16, 18):
  1. Detect  — OQE / reliability (observation quality)
  2. Calibrate — uncertainty set from conformal / inflation
  3. Constrain — tightened safe action set
  4. Shield  — repair candidate action into safe set
  5. Certify — dispatch certificate
"""
from __future__ import annotations

import uuid
from typing import Any, Mapping, Sequence

from orius.dc3s.drift import PageHinkleyDetector

# Explicit stage labels for documentation and tooling
PIPELINE_STAGES = ("Detect", "Calibrate", "Constrain", "Shield", "Certify")


def run_universal_step(
    *,
    domain_adapter: Any,
    raw_telemetry: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None,
    candidate_action: Mapping[str, Any],
    constraints: Mapping[str, Any],
    quantile: float = 50.0,
    cfg: Mapping[str, Any] | None = None,
    drift_detector: PageHinkleyDetector | None = None,
    residual: float | None = None,
    prev_cert_hash: str | None = None,
    device_id: str = "device-0",
    zone_id: str = "zone-0",
    controller: str = "orius-universal",
) -> dict[str, Any]:
    """Execute one full universal DC3S step for any domain.

    Stages:
        1. Detect  (OQE / reliability)
        2. Calibrate (uncertainty set)
        3. Constrain (tightened safe set)
        4. Shield  (repair)
        5. Certify (dispatch certificate)

    Returns dict with: certificate, safe_action, reliability_w, uncertainty_set, repair_meta, ...
    """
    dcfg = dict(cfg or {})
    expected_cadence = float(dcfg.get("expected_cadence_s", 3600))

    # Stage 1: Ingest + Detect (OQE)
    state = domain_adapter.ingest_telemetry(raw_telemetry)
    w_t, reliability_flags = domain_adapter.compute_oqe(state, history)

    # Drift detection
    drift_flag = False
    drift_meta: dict[str, Any] = {"drift": False, "score": 0.0}
    if drift_detector is not None and residual is not None:
        drift_meta = drift_detector.update(abs(float(residual)))
        drift_flag = bool(drift_meta.get("drift", False))

    # Stage 2: Calibrate (uncertainty set)
    uncertainty, cal_meta = domain_adapter.build_uncertainty_set(
        state=state,
        reliability_w=float(w_t),
        quantile=quantile,
        cfg=dcfg,
        drift_flag=drift_flag,
        prev_meta=None,
    )

    # Stage 3: Constrain (tightened action set)
    tightened = domain_adapter.tighten_action_set(
        uncertainty=uncertainty,
        constraints=constraints,
        cfg=dcfg,
    )

    # Stage 4: Shield (repair)
    safe_action, repair_meta = domain_adapter.repair_action(
        candidate_action=candidate_action,
        tightened_set=tightened,
        state=state,
        uncertainty=uncertainty,
        constraints=constraints,
        cfg=dcfg,
    )

    # Stage 5: Certify
    command_id = str(uuid.uuid4())
    certificate = domain_adapter.emit_certificate(
        command_id=command_id,
        device_id=device_id,
        zone_id=zone_id,
        controller=controller,
        proposed_action=dict(candidate_action),
        safe_action=dict(safe_action),
        uncertainty=uncertainty,
        reliability={"w_t": float(w_t), **reliability_flags},
        drift=drift_meta,
        cfg=dcfg,
        prev_hash=prev_cert_hash,
        repair_meta=repair_meta,
    )

    return {
        "certificate": certificate,
        "safe_action": dict(safe_action),
        "reliability_w": float(w_t),
        "reliability_flags": reliability_flags,
        "drift_flag": drift_flag,
        "drift_meta": drift_meta,
        "uncertainty_set": uncertainty,
        "repair_meta": repair_meta,
        "state": state,
    }
