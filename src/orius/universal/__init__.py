"""Compatibility exports for the typed universal theory kernel.

The canonical implementation now lives under ``orius.universal_theory``.
This package remains as a stable import surface for older notebooks and
experiments that imported ``orius.universal`` directly.
"""

from orius.universal_theory import (
    ContractVerifier,
    ContractViolation,
    DomainInstantiation as UniversalAdapterProtocol,
    FrontierPoint,
    ObservationConsistentStateSet as TightenedSet,
    RepairDecision as RepairResult,
    SafetyCertificate,
    UniversalStepResult,
    compute_episode_risk_bound,
    compute_frontier,
    compute_step_risk_bound,
    execute_universal_step,
    minimum_reliability_for_target,
)

SafetyBound = dict[str, float]

__all__ = [
    "ContractVerifier",
    "ContractViolation",
    "UniversalAdapterProtocol",
    "TightenedSet",
    "RepairResult",
    "SafetyBound",
    "SafetyCertificate",
    "UniversalStepResult",
    "FrontierPoint",
    "compute_frontier",
    "minimum_reliability_for_target",
    "compute_step_risk_bound",
    "compute_episode_risk_bound",
    "execute_universal_step",
]
