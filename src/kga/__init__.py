"""kga -- Knowability-Guided Adaptation (KGA), the K-Bound decision algorithm.

KGA can decide without deployment target labels after a benefit estimator has
been fitted on labelled development conditions and its residual radius calibrated
on a disjoint split.  Deployment maps label-free evidence ``Z`` to a certificate
``Delta_hat +/- epsilon`` and applies the trichotomy

    ADAPT   if Delta_hat - epsilon > 0
    FREEZE  if Delta_hat + epsilon < 0
    ABSTAIN otherwise

with false-adapt probability bounded by ``alpha`` conditional on the stated
coverage/transfer assumptions (Theorem 3 of the K-Bound paper).

Controlled-grid replay rule
---------------------------
The controlled-grid re-scoring path routes through
:func:`kga.policy.decide_kga`. That function pins the degrees of freedom that
used to vary between copy-pasted historical scripts:

* the radius is the **exact split-conformal rank** quantile
  ``eps = r_(k)``, ``k = ceil((n + 1)(1 - alpha))`` -- never an interpolated
  ``np.quantile``, and ``+inf`` (forced ABSTAIN) when ``k > n``;
* the pool is **leave-one-out-of-pool**: cell ``i``'s radius excludes its own
  labelled residual;
* the trichotomy uses **strict** inequalities, matching the ``|M| > beta``
  commitment convention of the knowability frontier.

Public API
----------
KGA                        -- facade gate (evidence -> frozen estimator -> decide).
BenefitEstimator           -- protocol for a frozen benefit model.
FrozenLinearBenefitEstimator -- auditable reference estimator artifact.
Decision                   -- the ADAPT/FREEZE/ABSTAIN enum.
Certificate                -- a finite-sample certificate ``Delta_hat +/- eps``.
Evidence                   -- the label-free evidence ``Z`` container.
decide_kga                 -- the canonical end-to-end rule (see above).
decide_batch               -- vectorised trichotomy over stored arrays.
split_conformal_rank_radius / conformal_radii_loo / min_calibration_size
                           -- the radius primitives.

The submodules ``kga.certificate`` and ``kga.policy`` remain the stable import
surface for scripts; nothing there has been renamed.

The core decision path is ``numpy``/``scipy`` (no torch) and deterministic.
"""

from __future__ import annotations

from kga._version import __version__
from kga.assumptions import (
    AssumptionReport,
    CoverageClaimBasis,
    CoverageType,
    FallbackAction,
    GateDecision,
    GateThresholds,
    ProtocolRecord,
    Status,
    conformal_radius,
    evidence_support_overlap,
    observed_coverage,
    radius_stability,
    risk_alignment_audit,
    run_gate,
    write_report,
)

try:
    from kga.benefit import (
        BenefitEstimator,
        FrozenLinearBenefitEstimator,
        fit_frozen_linear_benefit_estimator,
    )
except ImportError:
    BenefitEstimator = None  # type: ignore
    FrozenLinearBenefitEstimator = None  # type: ignore
    fit_frozen_linear_benefit_estimator = None  # type: ignore
from kga.certificate import (
    Certificate,
    InsufficientCalibrationError,
    conformal_attained_level,
    conformal_radii_loo,
    conformal_split,
    min_calibration_size,
    split_conformal_rank_radius,
)
from kga.crossfit import (
    ControlledGridCrossfitResult,
    controlled_grid_crossfit,
    controlled_grid_input_sha256,
    controlled_grid_sample_id,
)
from kga.evidence import EVIDENCE_FEATURE_NAMES, EVIDENCE_SCHEMA_VERSION, Evidence
from kga.frontier import (
    FrontierAssessment,
    assess_frontier,
    frontier_action,
    frontier_sensitivity,
)
from kga.kga import KGA
from kga.policy import (
    Decision,
    HierarchicalSelection,
    decide,
    decide_batch,
    decide_hierarchical_candidates,
    decide_kga,
)
from kga.population_transfer import (
    ConditionalPopulationInterval,
    compose_conditional_population_interval,
    hoeffding_paired_accuracy_radius,
)
from kga.routing import (
    AnytimeMulticandidatePanel,
    CandidateCertificate,
    RoutingDecision,
    bonferroni_multicandidate_route,
    multiclass_benefit,
    route_panel,
)
from kga.gateway import (
    AsyncSafeInferenceGateway,
    DeploymentMode,
    FallbackReason,
    GatewayDecision,
    SafeInferenceGateway,
)
from kga.guards import (
    CircuitBreaker,
    CircuitBreakerStatus,
    CircuitState,
    DistributedCircuitState,
    EvidenceSupportGuard,
    FileStateBackend,
    InMemoryStateBackend,
    NumericalHealthGuard,
    StateBackend,
    StreamingDriftMonitor,
)
from kga.integrations import TorchModelAdapter
from kga.registry import ProtocolIntegrityError, SealedProtocolManifest
from kga.observability import AuditLogger, METRICS

from kga.sensitivity import (
    SensitivityFrontier,
    SensitivityPoint,
    compute_break_even_beta,
    compute_sensitivity_frontier,
    compute_sensitivity_interval,
)

__all__ = [
    "KGA",
    "BenefitEstimator",
    "FrozenLinearBenefitEstimator",
    "fit_frozen_linear_benefit_estimator",
    "Decision",
    "Certificate",
    "SensitivityFrontier",
    "SensitivityPoint",
    "compute_break_even_beta",
    "compute_sensitivity_frontier",
    "compute_sensitivity_interval",
    "Evidence",
    "EVIDENCE_FEATURE_NAMES",
    "EVIDENCE_SCHEMA_VERSION",
    "FrontierAssessment",
    "InsufficientCalibrationError",
    "decide",
    "decide_batch",
    "decide_kga",
    "assess_frontier",
    "frontier_action",
    "frontier_sensitivity",
    "conformal_split",
    "conformal_radii_loo",
    "conformal_attained_level",
    "decide_kga",
    "decide_batch",
    "HierarchicalSelection",
    "decide_hierarchical_candidates",
    "split_conformal_rank_radius",
    "ControlledGridCrossfitResult",
    "controlled_grid_crossfit",
    "controlled_grid_input_sha256",
    "controlled_grid_sample_id",
    "AnytimeMulticandidatePanel",
    "CandidateCertificate",
    "RoutingDecision",
    "bonferroni_multicandidate_route",
    "multiclass_benefit",
    "route_panel",
    "AssumptionReport",
    "CoverageClaimBasis",
    "CoverageType",
    "FallbackAction",
    "GateDecision",
    "GateThresholds",
    "ProtocolRecord",
    "Status",
    "conformal_radius",
    "evidence_support_overlap",
    "observed_coverage",
    "radius_stability",
    "risk_alignment_audit",
    "run_gate",
    "write_report",
    "DeploymentMode",
    "FallbackReason",
    "GatewayDecision",
    "SafeInferenceGateway",
    "AsyncSafeInferenceGateway",
    "CircuitBreaker",
    "CircuitBreakerStatus",
    "CircuitState",
    "DistributedCircuitState",
    "StateBackend",
    "InMemoryStateBackend",
    "FileStateBackend",
    "EvidenceSupportGuard",
    "NumericalHealthGuard",
    "StreamingDriftMonitor",
    "TorchModelAdapter",
    "ProtocolIntegrityError",
    "SealedProtocolManifest",
    "AuditLogger",
    "METRICS",
    "__version__",
]
