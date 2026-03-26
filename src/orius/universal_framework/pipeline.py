"""Universal DC3S pipeline for any domain adapter.

The public entrypoint remains stable, but the implementation now delegates to
the typed universal degraded-observation kernel in ``orius.universal_theory``.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from orius.dc3s.drift import PageHinkleyDetector
from orius.universal_theory import UniversalStepResult, execute_universal_step

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
) -> UniversalStepResult:
    """Execute one full universal DC3S step for any domain.

    Stages:
        1. Detect  (OQE / reliability)
        2. Calibrate (uncertainty set)
        3. Constrain (tightened safe set)
        4. Shield  (repair)
        5. Certify (dispatch certificate)

    Returns a mapping-compatible ``UniversalStepResult`` so legacy callers can
    keep using ``result["safe_action"]`` while new code gets a typed object.
    """
    return execute_universal_step(
        domain_adapter=domain_adapter,
        raw_telemetry=raw_telemetry,
        history=history,
        candidate_action=candidate_action,
        constraints=constraints,
        quantile=quantile,
        cfg=cfg,
        drift_detector=drift_detector,
        residual=residual,
        prev_cert_hash=prev_cert_hash,
        device_id=device_id,
        zone_id=zone_id,
        controller=controller,
    )
