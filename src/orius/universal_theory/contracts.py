"""Typed contracts for the ORIUS degraded-observation safety kernel.

The legacy universal runtime exposed loosely typed ``dict[str, Any]`` payloads.
This module keeps the runtime interoperable with those adapters while making
the theorem-facing objects explicit and auditable.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from orius.dc3s.certificate import (
    CERTIFICATE_SCHEMA_VERSION,
    DEFAULT_CERTIFICATE_ISSUER,
    normalize_certificate_schema,
)

AssumptionTag = str


def _as_dict(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(payload or {})


def _tuple_tags(tags: Sequence[AssumptionTag] | None) -> tuple[AssumptionTag, ...]:
    return tuple(str(tag) for tag in (tags or ()))


def _bool_from_payload(payload: Mapping[str, Any], *keys: str, default: bool = False) -> bool:
    for key in keys:
        if key in payload:
            return bool(payload.get(key))
    return bool(default)


@dataclass(slots=True)
class ObservationPacket:
    """One causally available telemetry packet and its parsed state view."""

    raw_telemetry: dict[str, Any]
    parsed_state: dict[str, Any]
    timestamp_utc: str
    history_length: int = 0
    domain_id: str = ""


@dataclass(slots=True)
class ReliabilityAssessment:
    """Reliability score and detector output used by the universal kernel."""

    weight: float
    flags: dict[str, Any] = field(default_factory=dict)
    drift_flag: bool = False
    drift_meta: dict[str, Any] = field(default_factory=dict)
    assumption_tags: tuple[AssumptionTag, ...] = ("A2", "A5", "A6")

    def __post_init__(self) -> None:
        self.weight = float(self.weight)
        if not math.isfinite(self.weight) or not (0.0 <= self.weight <= 1.0):
            raise ValueError(f"ReliabilityAssessment.weight must lie in [0, 1]. Got {self.weight!r}.")
        self.flags = dict(self.flags)
        self.drift_meta = dict(self.drift_meta)
        self.assumption_tags = _tuple_tags(self.assumption_tags)


@dataclass(slots=True)
class ObservationConsistentStateSet:
    """Conservative state envelope consistent with degraded observation."""

    observed_state: dict[str, Any]
    lower_bounds: dict[str, float]
    upper_bounds: dict[str, float]
    raw_uncertainty: dict[str, Any]
    inflation: float
    quantile: float
    reliability_weight: float
    assumption_tags: tuple[AssumptionTag, ...] = ("A2", "A5")

    def __post_init__(self) -> None:
        self.observed_state = dict(self.observed_state)
        self.lower_bounds = {str(k): float(v) for k, v in self.lower_bounds.items()}
        self.upper_bounds = {str(k): float(v) for k, v in self.upper_bounds.items()}
        self.raw_uncertainty = dict(self.raw_uncertainty)
        self.inflation = float(self.inflation)
        self.quantile = float(self.quantile)
        self.reliability_weight = float(self.reliability_weight)
        if not math.isfinite(self.inflation) or self.inflation < 1.0:
            raise ValueError(
                f"ObservationConsistentStateSet.inflation must be finite and >= 1.0. Got {self.inflation!r}."
            )
        if not math.isfinite(self.reliability_weight) or not (0.0 <= self.reliability_weight <= 1.0):
            raise ValueError(
                "ObservationConsistentStateSet.reliability_weight must lie in [0, 1]. "
                f"Got {self.reliability_weight!r}."
            )
        self.assumption_tags = _tuple_tags(self.assumption_tags)


@dataclass(slots=True)
class SafetySpec:
    """Safety constraints and fallback semantics for one control step."""

    constraints: dict[str, Any]
    fallback_action: dict[str, float] | None = None
    assumption_tags: tuple[AssumptionTag, ...] = ("A3", "A4", "A8")

    def __post_init__(self) -> None:
        self.constraints = dict(self.constraints)
        self.fallback_action = (
            None
            if self.fallback_action is None
            else {str(k): float(v) for k, v in self.fallback_action.items()}
        )
        self.assumption_tags = _tuple_tags(self.assumption_tags)


@dataclass(slots=True)
class SafeActionSet:
    """Safe-action representation returned by a domain instantiation."""

    representation: dict[str, Any]
    viable: bool
    fallback_action: dict[str, float] | None = None
    assumption_tags: tuple[AssumptionTag, ...] = ("A3", "A4", "A8")

    def __post_init__(self) -> None:
        self.representation = dict(self.representation)
        self.fallback_action = (
            None
            if self.fallback_action is None
            else {str(k): float(v) for k, v in self.fallback_action.items()}
        )
        self.assumption_tags = _tuple_tags(self.assumption_tags)


@dataclass(slots=True)
class RepairDecision:
    """Projected action together with repair metadata and theorem tags."""

    proposed_action: dict[str, Any]
    safe_action: dict[str, Any]
    repaired: bool
    mode: str
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    assumption_tags: tuple[AssumptionTag, ...] = ("A3", "A8")

    def __post_init__(self) -> None:
        self.proposed_action = dict(self.proposed_action)
        self.safe_action = dict(self.safe_action)
        self.repaired = bool(self.repaired)
        self.mode = str(self.mode)
        self.reason = None if self.reason in (None, "") else str(self.reason)
        self.metadata = dict(self.metadata)
        self.assumption_tags = _tuple_tags(self.assumption_tags)


@dataclass(slots=True)
class SafetyCertificate(Mapping[str, Any]):
    """Structured certificate that remains mapping-compatible for callers."""

    certificate_schema_version: str
    issuer: str
    domain: str
    action: dict[str, Any]
    validity_horizon_H_t: int | None
    expires_at_step: int | None
    theorem_contracts: dict[str, Any]
    command_id: str
    certificate_id: str
    created_at: str
    device_id: str
    zone_id: str
    controller: str
    proposed_action: dict[str, Any]
    safe_action: dict[str, Any]
    uncertainty: dict[str, Any]
    reliability: dict[str, Any]
    drift: dict[str, Any]
    certificate_hash: str | None = None
    prev_hash: str | None = None
    signature: str | None = None
    signature_algorithm: str | None = None
    public_key_id: str | None = None
    assumptions_checked: tuple[AssumptionTag, ...] = ()
    certificate_horizon_steps: int | None = None
    source_domain: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.certificate_schema_version = str(self.certificate_schema_version or CERTIFICATE_SCHEMA_VERSION)
        self.issuer = str(self.issuer or DEFAULT_CERTIFICATE_ISSUER)
        self.domain = str(self.domain or self.zone_id)
        self.action = dict(self.action)
        self.validity_horizon_H_t = (
            None if self.validity_horizon_H_t is None else int(self.validity_horizon_H_t)
        )
        self.expires_at_step = None if self.expires_at_step is None else int(self.expires_at_step)
        self.theorem_contracts = {
            str(key): dict(value) if isinstance(value, Mapping) else value
            for key, value in self.theorem_contracts.items()
        }
        self.command_id = str(self.command_id)
        self.certificate_id = str(self.certificate_id)
        self.created_at = str(self.created_at)
        self.device_id = str(self.device_id)
        self.zone_id = str(self.zone_id)
        self.controller = str(self.controller)
        self.proposed_action = dict(self.proposed_action)
        self.safe_action = dict(self.safe_action)
        self.uncertainty = dict(self.uncertainty)
        self.reliability = dict(self.reliability)
        self.drift = dict(self.drift)
        self.certificate_hash = None if self.certificate_hash in (None, "") else str(self.certificate_hash)
        self.prev_hash = None if self.prev_hash in (None, "") else str(self.prev_hash)
        self.signature = None if self.signature in (None, "") else str(self.signature)
        self.signature_algorithm = (
            None if self.signature_algorithm in (None, "") else str(self.signature_algorithm)
        )
        self.public_key_id = None if self.public_key_id in (None, "") else str(self.public_key_id)
        self.assumptions_checked = _tuple_tags(self.assumptions_checked)
        self.certificate_horizon_steps = (
            None if self.certificate_horizon_steps is None else int(self.certificate_horizon_steps)
        )
        self.source_domain = str(self.source_domain)
        self.extras = dict(self.extras)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        assumptions_checked: Sequence[AssumptionTag] | None = None,
        source_domain: str = "",
    ) -> SafetyCertificate:
        raw = normalize_certificate_schema(payload)
        known_keys = {
            "certificate_schema_version",
            "issuer",
            "domain",
            "action",
            "validity_horizon_H_t",
            "expires_at_step",
            "theorem_contracts",
            "command_id",
            "certificate_id",
            "created_at",
            "device_id",
            "zone_id",
            "controller",
            "proposed_action",
            "safe_action",
            "uncertainty",
            "reliability",
            "drift",
            "certificate_hash",
            "cert_hash",
            "prev_hash",
            "signature",
            "signature_algorithm",
            "public_key_id",
            "assumptions_checked",
            "certificate_horizon_steps",
            "tau_t",
            "source_domain",
        }
        horizon = raw.get("certificate_horizon_steps", raw.get("validity_horizon_H_t", raw.get("tau_t")))
        assumption_tags = assumptions_checked or raw.get("assumptions_checked") or ()
        return cls(
            certificate_schema_version=str(raw.get("certificate_schema_version", CERTIFICATE_SCHEMA_VERSION)),
            issuer=str(raw.get("issuer", DEFAULT_CERTIFICATE_ISSUER)),
            domain=str(raw.get("domain", raw.get("zone_id", ""))),
            action=_as_dict(raw.get("action")),
            validity_horizon_H_t=None
            if raw.get("validity_horizon_H_t") is None
            else int(raw["validity_horizon_H_t"]),
            expires_at_step=None if raw.get("expires_at_step") is None else int(raw["expires_at_step"]),
            theorem_contracts=_as_dict(raw.get("theorem_contracts")),
            command_id=str(raw.get("command_id", raw.get("certificate_id", ""))),
            certificate_id=str(raw.get("certificate_id", raw.get("command_id", ""))),
            created_at=str(raw.get("created_at", "")),
            device_id=str(raw.get("device_id", "")),
            zone_id=str(raw.get("zone_id", "")),
            controller=str(raw.get("controller", "")),
            proposed_action=_as_dict(raw.get("proposed_action")),
            safe_action=_as_dict(raw.get("safe_action")),
            uncertainty=_as_dict(raw.get("uncertainty")),
            reliability=_as_dict(raw.get("reliability")),
            drift=_as_dict(raw.get("drift")),
            certificate_hash=raw.get("certificate_hash"),
            prev_hash=raw.get("prev_hash"),
            signature=raw.get("signature"),
            signature_algorithm=raw.get("signature_algorithm"),
            public_key_id=raw.get("public_key_id"),
            assumptions_checked=_tuple_tags(assumption_tags),
            certificate_horizon_steps=None if horizon is None else int(horizon),
            source_domain=str(source_domain or raw.get("source_domain", "")),
            extras={key: value for key, value in raw.items() if key not in known_keys},
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = {
            "certificate_schema_version": self.certificate_schema_version,
            "issuer": self.issuer,
            "domain": self.domain,
            "action": dict(self.action),
            "validity_horizon_H_t": self.validity_horizon_H_t,
            "expires_at_step": self.expires_at_step,
            "theorem_contracts": dict(self.theorem_contracts),
            "command_id": self.command_id,
            "certificate_id": self.certificate_id,
            "created_at": self.created_at,
            "device_id": self.device_id,
            "zone_id": self.zone_id,
            "controller": self.controller,
            "proposed_action": dict(self.proposed_action),
            "safe_action": dict(self.safe_action),
            "uncertainty": dict(self.uncertainty),
            "reliability": dict(self.reliability),
            "drift": dict(self.drift),
            "certificate_hash": self.certificate_hash,
            "prev_hash": self.prev_hash,
            "signature": self.signature,
            "signature_algorithm": self.signature_algorithm,
            "public_key_id": self.public_key_id,
            "assumptions_checked": list(self.assumptions_checked),
            "source_domain": self.source_domain,
        }
        if self.certificate_horizon_steps is not None:
            payload["certificate_horizon_steps"] = int(self.certificate_horizon_steps)
        payload.update(self.extras)
        return payload

    def __getitem__(self, key: str) -> Any:
        return self.to_mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_mapping())

    def __len__(self) -> int:
        return len(self.to_mapping())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_mapping().get(key, default)


@dataclass(slots=True)
class UniversalStepResult(Mapping[str, Any]):
    """Structured runtime result with backward-compatible mapping access."""

    certificate: SafetyCertificate
    safe_action: dict[str, Any]
    reliability: ReliabilityAssessment
    uncertainty_set: ObservationConsistentStateSet
    safe_action_set: SafeActionSet
    repair_decision: RepairDecision
    observation: ObservationPacket
    state: dict[str, Any]
    step_risk_bound: float
    episode_risk_bound: dict[str, Any]
    contract_checks: dict[str, Any] = field(default_factory=dict)
    theorem_contracts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.safe_action = dict(self.safe_action)
        self.state = dict(self.state)
        self.step_risk_bound = float(self.step_risk_bound)
        self.episode_risk_bound = {
            str(key): float(value)
            if isinstance(value, int | float) and not isinstance(value, bool)
            else value
            for key, value in self.episode_risk_bound.items()
        }
        self.contract_checks = {
            str(key): dict(value) if isinstance(value, Mapping) else value
            for key, value in self.contract_checks.items()
        }
        self.theorem_contracts = {
            str(key): dict(value) if isinstance(value, Mapping) else value
            for key, value in self.theorem_contracts.items()
        }

    def to_mapping(self) -> dict[str, Any]:
        runtime_surface = self.certificate.get(
            "runtime_surface",
            self.reliability.flags.get("runtime_surface", ""),
        )
        closure_tier = self.certificate.get(
            "closure_tier",
            self.reliability.flags.get("closure_tier", ""),
        )
        feature_basis = self.certificate.get(
            "reliability_feature_basis",
            self.reliability.flags.get("reliability_feature_basis", {}),
        )
        return {
            "certificate": self.certificate,
            "safe_action": dict(self.safe_action),
            "reliability_w": float(self.reliability.weight),
            "reliability_flags": dict(self.reliability.flags),
            "runtime_surface": runtime_surface,
            "closure_tier": closure_tier,
            "reliability_feature_basis": dict(feature_basis),
            "drift_flag": bool(self.reliability.drift_flag),
            "drift_meta": dict(self.reliability.drift_meta),
            "uncertainty_set": dict(self.uncertainty_set.raw_uncertainty),
            "repair_meta": {
                **dict(self.repair_decision.metadata),
                "repaired": bool(self.repair_decision.repaired),
                "mode": self.repair_decision.mode,
                "intervention_reason": self.repair_decision.reason,
            },
            "state": dict(self.state),
            "observation_packet": self.observation,
            "step_risk_bound": float(self.step_risk_bound),
            "episode_risk_bound": dict(self.episode_risk_bound),
            "safe_action_set": dict(self.safe_action_set.representation),
            "contract_checks": dict(self.contract_checks),
            "theorem_contracts": dict(self.theorem_contracts),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_mapping())

    def __len__(self) -> int:
        return len(self.to_mapping())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_mapping().get(key, default)


class DomainInstantiation(Protocol):
    """Minimal universal contract that a domain adapter must satisfy."""

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]: ...

    def build_uncertainty_set(
        self,
        state: Mapping[str, Any],
        reliability_w: float,
        quantile: float,
        *,
        cfg: Mapping[str, Any],
        drift_flag: bool | None = None,
        prev_meta: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...

    def tighten_action_set(
        self,
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        *,
        cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def repair_action(
        self,
        candidate_action: Mapping[str, Any],
        tightened_set: Mapping[str, Any],
        *,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        cfg: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...

    def emit_certificate(
        self,
        *,
        command_id: str,
        device_id: str,
        zone_id: str,
        controller: str,
        proposed_action: Mapping[str, Any],
        safe_action: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
        reliability: Mapping[str, Any],
        drift: Mapping[str, Any],
        cfg: Mapping[str, Any],
        prev_hash: str | None = None,
        dispatch_plan: Mapping[str, Any] | None = None,
        repair_meta: Mapping[str, Any] | None = None,
        guarantee_meta: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


class ContractViolation(ValueError):
    """Raised when a domain adapter fails the universal theory contract."""


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _extract_interval_widths(payload: Mapping[str, Any]) -> dict[str, float]:
    """Extract interval widths from lower/upper style payload keys when possible."""

    lower_bounds: dict[str, float] = {}
    upper_bounds: dict[str, float] = {}
    for key, value in payload.items():
        scalar = _as_float(value)
        if scalar is None:
            continue
        if key.endswith("_lower") or key.endswith("_lo"):
            lower_bounds[key.rsplit("_", 1)[0]] = scalar
        elif key.endswith("_upper") or key.endswith("_hi"):
            upper_bounds[key.rsplit("_", 1)[0]] = scalar
        elif "_lower_" in key:
            base, _, suffix = key.partition("_lower_")
            lower_bounds[f"{base}_{suffix}"] = scalar
        elif "_upper_" in key:
            base, _, suffix = key.partition("_upper_")
            upper_bounds[f"{base}_{suffix}"] = scalar
    return {
        key: float(upper_bounds[key] - lower_bounds[key])
        for key in sorted(lower_bounds.keys() & upper_bounds.keys())
    }


def _extract_inflation(payload: Mapping[str, Any], fallback: float | None = None) -> float | None:
    meta = payload.get("meta")
    if isinstance(meta, Mapping):
        scalar = _as_float(meta.get("inflation"))
        if scalar is not None:
            return scalar
    return fallback


def _extract_reliability_weight(payload: Mapping[str, Any]) -> float | None:
    for key in ("w_t", "weight", "w"):
        scalar = _as_float(payload.get(key))
        if scalar is not None:
            return scalar
    return None


def _extract_action_bounds(payload: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    lower_bounds: dict[str, float] = {}
    upper_bounds: dict[str, float] = {}
    for key, value in payload.items():
        scalar = _as_float(value)
        if scalar is None:
            continue
        if key.endswith("_lower"):
            lower_bounds[key[: -len("_lower")]] = scalar
        elif key.endswith("_upper"):
            upper_bounds[key[: -len("_upper")]] = scalar
    return lower_bounds, upper_bounds


def _action_within_safe_set_bounds(
    action: Mapping[str, Any],
    safe_action_set: SafeActionSet,
) -> tuple[bool, list[str]]:
    lower_bounds, upper_bounds = _extract_action_bounds(safe_action_set.representation)
    comparable_keys = sorted(set(action.keys()) & set(lower_bounds.keys()) & set(upper_bounds.keys()))
    if not comparable_keys:
        return bool(safe_action_set.viable or safe_action_set.fallback_action is not None), comparable_keys
    passed = all(
        lower_bounds[key] - 1e-9 <= float(action[key]) <= upper_bounds[key] + 1e-9 for key in comparable_keys
    )
    return passed, comparable_keys


class ContractVerifier:
    """Lightweight verifier for the universal theory adapter contract."""

    REQUIRED_METHODS = (
        "ingest_telemetry",
        "compute_oqe",
        "build_uncertainty_set",
        "tighten_action_set",
        "repair_action",
        "emit_certificate",
    )

    @staticmethod
    def _record_check(
        *,
        name: str,
        passed: bool,
        detail: str,
        assumption_tags: Sequence[AssumptionTag] = (),
    ) -> dict[str, Any]:
        return {
            "name": str(name),
            "passed": bool(passed),
            "detail": str(detail),
            "assumption_tags": list(_tuple_tags(assumption_tags)),
        }

    @classmethod
    def check(cls, adapter: Any) -> None:
        missing = [name for name in cls.REQUIRED_METHODS if not callable(getattr(adapter, name, None))]
        if missing:
            raise ContractViolation(
                f"{adapter.__class__.__name__} is not a universal domain instantiation. "
                f"Missing methods: {', '.join(missing)}"
            )

    @classmethod
    def summary(cls, adapter: Any) -> dict[str, Any]:
        missing = [name for name in cls.REQUIRED_METHODS if not callable(getattr(adapter, name, None))]
        return {
            "adapter": adapter.__class__.__name__,
            "contract_passed": len(missing) == 0,
            "missing_methods": missing,
            "structural_only": True,
        }

    @classmethod
    def validate_runtime_step(
        cls,
        *,
        adapter: DomainInstantiation,
        state: Mapping[str, Any],
        constraints: Mapping[str, Any],
        quantile: float,
        cfg: Mapping[str, Any],
        reliability: ReliabilityAssessment,
        uncertainty_set: ObservationConsistentStateSet,
        safe_action_set: SafeActionSet,
        repair_decision: RepairDecision,
        certificate: SafetyCertificate,
        step_risk_bound: float,
        episode_risk_bound: Mapping[str, Any],
        alpha: float,
    ) -> dict[str, Any]:
        """Validate theorem-aligned invariants on the actual runtime step objects."""

        cls.check(adapter)
        checks: dict[str, dict[str, Any]] = {}
        failures: list[str] = []

        def record(
            name: str,
            passed: bool,
            detail: str,
            assumption_tags: Sequence[AssumptionTag] = (),
        ) -> None:
            checks[name] = cls._record_check(
                name=name,
                passed=passed,
                detail=detail,
                assumption_tags=assumption_tags,
            )
            if not passed:
                failures.append(f"{name}: {detail}")

        record(
            "reliability_range",
            0.0 <= float(reliability.weight) <= 1.0,
            f"w_t={float(reliability.weight):.6f} lies in [0, 1].",
            assumption_tags=("A2", "A5", "A6"),
        )
        record(
            "inflation_preserves_base_surface",
            float(uncertainty_set.inflation) >= 1.0,
            f"inflation={float(uncertainty_set.inflation):.6f} preserves the base conformal surface.",
            assumption_tags=("A2", "A5"),
        )
        record(
            "state_set_matches_reliability",
            abs(float(uncertainty_set.reliability_weight) - float(reliability.weight)) <= 1e-9,
            "Observation-consistent set is tagged with the active runtime reliability weight.",
            assumption_tags=("A2", "A5"),
        )

        degraded_weight = max(0.0, float(reliability.weight) - 0.25)
        if degraded_weight < float(reliability.weight):
            degraded_uncertainty, degraded_meta = adapter.build_uncertainty_set(
                state=dict(state),
                reliability_w=degraded_weight,
                quantile=float(quantile),
                cfg=dict(cfg),
                drift_flag=reliability.drift_flag,
                prev_meta=uncertainty_set.raw_uncertainty,
            )
            degraded_inflation = _extract_inflation(
                degraded_uncertainty,
                fallback=_extract_inflation(degraded_meta, fallback=float(uncertainty_set.inflation)),
            )
            record(
                "monotone_inflation",
                degraded_inflation is not None
                and degraded_inflation + 1e-9 >= float(uncertainty_set.inflation),
                (
                    f"Lower reliability probe w'={degraded_weight:.3f} yields inflation "
                    f"{float(degraded_inflation or 0.0):.6f}, current inflation={float(uncertainty_set.inflation):.6f}."
                ),
                assumption_tags=("A2", "A5"),
            )

            degraded_tightened = adapter.tighten_action_set(
                uncertainty=degraded_uncertainty,
                constraints=constraints,
                cfg=dict(cfg),
            )
            current_widths = _extract_interval_widths(safe_action_set.representation)
            degraded_widths = _extract_interval_widths(degraded_tightened)
            comparable_keys = sorted(current_widths.keys() & degraded_widths.keys())
            if comparable_keys:
                monotone_widths = all(
                    degraded_widths[key] <= current_widths[key] + 1e-9 for key in comparable_keys
                )
                record(
                    "tightened_set_monotonicity",
                    monotone_widths,
                    "Lower-reliability tightened-set widths do not exceed current widths on comparable axes.",
                    assumption_tags=("A3", "A4", "A5"),
                )

        if safe_action_set.viable:
            record(
                "viable_set_returns_safe_action",
                bool(repair_decision.safe_action),
                "Viable tightened set returned a non-empty repaired action.",
                assumption_tags=("A3", "A4", "A8"),
            )
        else:
            has_fallback = safe_action_set.fallback_action is not None
            record(
                "fallback_action_available",
                has_fallback,
                "Non-viable tightened set exposes a fallback action.",
                assumption_tags=("A4", "A8"),
            )
            record(
                "fallback_mode_when_infeasible",
                has_fallback and repair_decision.mode == "fallback",
                "Non-viable tightened set triggers fallback mode.",
                assumption_tags=("A4", "A8"),
            )
            if has_fallback:
                record(
                    "fallback_action_matches_certificate",
                    dict(repair_decision.safe_action) == dict(safe_action_set.fallback_action or {}),
                    "Fallback repair action matches the declared fallback action.",
                    assumption_tags=("A4", "A8"),
                )

        record(
            "certificate_matches_repair",
            dict(certificate.safe_action) == dict(repair_decision.safe_action)
            and dict(certificate.proposed_action) == dict(repair_decision.proposed_action),
            "Certificate actions match the runtime repair decision.",
            assumption_tags=("A7", "A8"),
        )
        certificate_weight = _extract_reliability_weight(certificate.reliability)
        record(
            "certificate_reliability_consistency",
            certificate_weight is not None and abs(certificate_weight - float(reliability.weight)) <= 1e-9,
            (
                "Certificate reliability payload matches the active runtime reliability weight."
                if certificate_weight is not None
                else "Certificate reliability payload is missing w_t/weight."
            ),
            assumption_tags=("A7", "A8"),
        )
        record(
            "certificate_assumptions_present",
            len(tuple(certificate.assumptions_checked)) > 0,
            "Certificate carries checked assumption tags.",
            assumption_tags=("A7",),
        )
        horizon = _as_float(episode_risk_bound.get("horizon"))
        mean_reliability = _as_float(episode_risk_bound.get("mean_reliability_w"))
        expected_episode = None
        if horizon is not None and mean_reliability is not None:
            expected_episode = float(alpha) * (1.0 - mean_reliability) * horizon
        observed_episode = _as_float(episode_risk_bound.get("bound_expected_violations"))
        scope = str(episode_risk_bound.get("scope", ""))
        expected_step = float(alpha) * (1.0 - float(reliability.weight))
        record(
            "risk_bound_semantics",
            abs(float(step_risk_bound) - expected_step) <= 1e-9
            and expected_episode is not None
            and observed_episode is not None
            and abs(observed_episode - expected_episode) <= 1e-9
            and scope in {"current_step_only", "observed_prefix"},
            (
                f"step={float(step_risk_bound):.6f}, expected_step={expected_step:.6f}, "
                f"episode={observed_episode if observed_episode is not None else 'missing'}, "
                f"expected_episode={expected_episode if expected_episode is not None else 'missing'}, "
                f"scope={scope or 'missing'}."
            ),
            assumption_tags=("A2", "A5"),
        )

        summary = {
            "adapter": adapter.__class__.__name__,
            "contract_passed": len(failures) == 0,
            "checked_invariants": checks,
            "failed_invariants": failures,
        }
        if failures:
            raise ContractViolation(
                f"{adapter.__class__.__name__} failed runtime semantic contract checks: "
                + "; ".join(failures)
            )
        return summary

    @classmethod
    def build_transfer_theorem_summary(
        cls,
        *,
        adapter: DomainInstantiation,
        reliability: ReliabilityAssessment,
        uncertainty_set: ObservationConsistentStateSet,
        safe_action_set: SafeActionSet,
        repair_decision: RepairDecision,
        certificate: SafetyCertificate,
        contract_checks: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Summarize the typed four-obligation T11 surface on runtime artifacts."""

        checked_invariants = dict(contract_checks.get("checked_invariants", {}))

        def invariant_passed(name: str) -> bool:
            return bool(checked_invariants.get(name, {}).get("passed", False))

        membership_passed, comparable_keys = _action_within_safe_set_bounds(
            repair_decision.safe_action,
            safe_action_set,
        )
        adapter_summary = cls.summary(adapter)
        coverage_passed = (
            invariant_passed("inflation_preserves_base_surface")
            and invariant_passed("state_set_matches_reliability")
            and abs(float(uncertainty_set.reliability_weight) - float(reliability.weight)) <= 1e-9
        )
        repair_membership_passed = (
            invariant_passed("certificate_matches_repair") and membership_passed
            if safe_action_set.viable
            else invariant_passed("fallback_action_matches_certificate")
        )
        fallback_passed = (
            True
            if safe_action_set.viable
            else (
                invariant_passed("fallback_action_available")
                and invariant_passed("fallback_mode_when_infeasible")
                and invariant_passed("fallback_action_matches_certificate")
            )
        )
        sound_surface_passed = (
            bool(adapter_summary.get("contract_passed"))
            and bool(safe_action_set.representation)
            and invariant_passed("certificate_reliability_consistency")
            and (
                invariant_passed("tightened_set_monotonicity")
                if "tightened_set_monotonicity" in checked_invariants
                else True
            )
        )
        obligations = {
            "coverage": {
                "passed": bool(coverage_passed),
                "detail": (
                    "Observation-consistent state set preserves inflation >= 1 "
                    "and carries the active runtime reliability tag."
                ),
            },
            "sound_safe_action_set": {
                "passed": bool(sound_surface_passed),
                "detail": (
                    "Typed safe-action surface exists, passes the adapter contract, "
                    "and preserves the canonical reliability/certificate semantics."
                ),
            },
            "repair_membership": {
                "passed": bool(repair_membership_passed),
                "detail": (
                    "Repair action remains inside the typed safe-action set bounds "
                    f"for comparable keys {comparable_keys or ['<structural-only>']}."
                ),
            },
            "fallback_admissibility": {
                "passed": bool(fallback_passed),
                "detail": (
                    "Fallback branch is either not needed (viable safe-action set) "
                    "or is explicitly carried by the typed fallback action."
                ),
            },
        }
        failed_obligations = [name for name, result in obligations.items() if not bool(result["passed"])]
        return {
            "theorem_id": "T11",
            "theorem_surface": "forward_four_obligation_transfer",
            "forward_only": True,
            "all_executable_checks_passed": len(failed_obligations) == 0,
            "status": "runtime_linked" if not failed_obligations else "contract_violation",
            "obligations": obligations,
            "failed_obligations": failed_obligations,
            "adapter_contract_summary": adapter_summary,
            "declared_assumptions": [
                "Coverage obligation for the observation-consistent state set.",
                "Soundness of the tightened safe-action set.",
                "Repair membership in the tightened safe-action set.",
                "Fallback admissibility when the tightened set is empty.",
            ],
            "declared_only_contract": (
                "Plant-specific next-state soundness of the tightened safe-action "
                "set is supplied by the domain adapter contract; the generic kernel "
                "checks the typed artifacts and certificate semantics, not a new "
                "domain-specific dynamics proof."
            ),
            "scope_note": (
                "This summary closes the forward one-step transfer surface only. "
                "The converse remains a separate structural failure proposition."
            ),
        }


# ---------------------------------------------------------------------------
# Formal Assumption Register  (A1 – A13)
# ---------------------------------------------------------------------------

ASSUMPTION_REGISTER: dict[str, dict[str, str]] = {
    "A1": {
        "tag": "A1",
        "name": "Almost-sure model error bound",
        "formal": ("For all t, |x_{t+1} - f(x_t,a_t)| <= epsilon_model almost surely."),
        "role": (
            "Provides the deterministic one-step slack used by shielded "
            "postconditions and proof surfaces that cannot rely only on "
            "high-probability disturbance tails."
        ),
    },
    "A2": {
        "tag": "A2",
        "name": "Telemetry-state bridge",
        "formal": (
            "There exists a known monotone function g:[0,1]->R_+ with g(1)=0 "
            "such that |hat{x}_t - x_t| <= g(w_t) for all t."
        ),
        "role": (
            "Connects the runtime reliability score to the state discrepancy "
            "used by observation-consistent sets and tightened safe-action "
            "constraints."
        ),
    },
    "A3": {
        "tag": "A3",
        "name": "Feasible safe repair",
        "formal": ("Whenever the runtime invokes repair, the tightened safe-action set is non-empty."),
        "role": (
            "Keeps repair as an active runtime precondition rather than an "
            "unrestricted theorem that arbitrary constraints always admit a "
            "safe action."
        ),
    },
    "A4": {
        "tag": "A4",
        "name": "Known one-step dynamics",
        "formal": ("The one-step increment map Delta(a)=f(x,a)-x is known exactly to the controller."),
        "role": (
            "Allows runtime certificates to check the immediate effect of an "
            "action against domain constraints."
        ),
    },
    "A5": {
        "tag": "A5",
        "name": "Absorbed monotone tightening",
        "formal": (
            "The defended tightening margin m_t^* = m_t + epsilon_model is "
            "monotone non-increasing in w_t and bounded above by a finite constant."
        ),
        "role": (
            "Ensures lower reliability expands conservatism monotonically "
            "without unbounded runtime inflation."
        ),
    },
    "A6": {
        "tag": "A6",
        "name": "Bounded detector lag",
        "formal": (
            "Each supported telemetry fault class is detected within a finite "
            "lag tau_max known to the runtime."
        ),
        "role": ("Bounds the exposure window before OQE degradation, repair, or fallback must respond."),
    },
    "A7": {
        "tag": "A7",
        "name": "Causal certificate rule",
        "formal": ("The certificate state obeys cert_t = h(z_1,...,z_t) for a causal update rule h."),
        "role": ("Prevents certificates from using future telemetry and supports the runtime audit chain."),
    },
    "A8": {
        "tag": "A8",
        "name": "Piecewise certified fallback",
        "formal": (
            "Whenever the latent state is safe, fallback either safely holds "
            "the state in place or applies a boundary-aware recovery action "
            "that moves the state inward; otherwise the runtime fails closed."
        ),
        "role": (
            "Scopes fallback to certified hold-or-recovery behavior instead "
            "of assuming universal fallback feasibility."
        ),
    },
    "A9": {
        "tag": "A9",
        "name": "Sub-Gaussian disturbance law",
        "formal": (
            "The disturbance increments satisfy E[exp(s epsilon_t)] <= exp(s^2 sigma_d^2 / 2) for all real s."
        ),
        "role": (
            "Supplies the high-probability disturbance proxy used by T6 and trajectory-PAC validity surfaces."
        ),
    },
    "A10a": {
        "tag": "A10a",
        "name": "Polynomial mixing telemetry",
        "formal": (
            "The telemetry process is phi-mixing with polynomial decay phi(k)=O(k^{-beta}) for some beta > 1."
        ),
        "role": ("Supports scoped weak-dependence arguments where polynomial mixing is sufficient."),
    },
    "A10b": {
        "tag": "A10b",
        "name": "Geometric mixing telemetry",
        "formal": (
            "The telemetry process is phi-mixing with geometric decay "
            "phi(k) <= C rho^k for C > 0 and rho in (0,1)."
        ),
        "role": ("Supports the current T9 separated-window proof under stronger dependence decay."),
    },
    "A11": {
        "tag": "A11",
        "name": "Arbitrage boundary reachability",
        "formal": (
            "Under the admissible-demand family, the evaluated controller can "
            "be driven to a boundary-sensitive witness state where the safety "
            "margin is tight enough for an OASG/no-free-safety construction."
        ),
        "role": (
            "Keeps T4 as an explicit finite-margin and boundary-reachability "
            "witness rather than a universal claim over all operating states."
        ),
    },
    "A12": {
        "tag": "A12",
        "name": "Controller-fault independence",
        "formal": (
            "Conditioned on the available history, the fault/reliability "
            "process is independent of the controller action stream."
        ),
        "role": (
            "Excludes controller-induced sensing faults and adaptive coupled "
            "fault processes from the defended theorem surface."
        ),
    },
    "A13": {
        "tag": "A13",
        "name": "TV bridge",
        "formal": (
            "For the T10 binary boundary-testing observation laws, TV(P_{0,t}, P_{1,t}) <= w_t for every t."
        ),
        "role": (
            "Links the reliability score to distinguishability in the scoped binary lower-bound construction."
        ),
    },
}


def verify_assumption_coverage(tags: set[str]) -> dict[str, Any]:
    """Check which assumptions from A1-A13 are covered by a given tag set.

    Returns dict with keys: covered (list), missing (list), fraction (float).
    """
    all_tags = set(ASSUMPTION_REGISTER.keys())
    covered = sorted(all_tags & tags)
    missing = sorted(all_tags - tags)
    return {
        "covered": covered,
        "missing": missing,
        "fraction": len(covered) / len(all_tags) if all_tags else 0.0,
    }
