"""Compatibility exports for the typed universal theory kernel.

The canonical implementation now lives under ``orius.universal_theory``.
This package remains as a stable import surface for older notebooks and
experiments that imported ``orius.universal`` directly.
"""

from orius.universal_theory import (
    ContractVerifier,
    ContractViolation,
    FrontierPoint,
    SafetyCertificate,
    UniversalStepResult,
    compute_episode_risk_bound,
    compute_frontier,
    compute_step_risk_bound,
    execute_universal_step,
    minimum_reliability_for_target,
)
from orius.universal_theory import (
    DomainInstantiation as UniversalAdapterProtocol,
)
from orius.universal_theory import (
    ObservationConsistentStateSet as TightenedSet,
)
from orius.universal_theory import (
    RepairDecision as RepairResult,
)

SafetyBound = dict[str, float]

__all__ = [
    "ContractVerifier",
    "ContractViolation",
    "FrontierPoint",
    "RepairResult",
    "SafetyBound",
    "SafetyCertificate",
    "TightenedSet",
    "UniversalAdapterProtocol",
    "UniversalStepResult",
    "compute_episode_risk_bound",
    "compute_frontier",
    "compute_step_risk_bound",
    "execute_universal_step",
    "minimum_reliability_for_target",
]
