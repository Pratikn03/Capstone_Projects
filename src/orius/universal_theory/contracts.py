"""Typed contracts for the ORIUS degraded-observation safety kernel.

The legacy universal runtime exposed loosely typed ``dict[str, Any]`` payloads.
This module keeps the runtime interoperable with those adapters while making
the theorem-facing objects explicit and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol, Sequence


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
        self.weight = float(min(1.0, max(0.0, float(self.weight))))
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
        self.assumption_tags = _tuple_tags(self.assumption_tags)


@dataclass(slots=True)
class SafetySpec:
    """Safety constraints and fallback semantics for one control step."""

    constraints: dict[str, Any]
    fallback_action: dict[str, float] | None = None
    assumption_tags: tuple[AssumptionTag, ...] = ("A3", "A4", "A8")

    def __post_init__(self) -> None:
        self.constraints = dict(self.constraints)
        self.fallback_action = None if self.fallback_action is None else {
            str(k): float(v) for k, v in self.fallback_action.items()
        }
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
        self.fallback_action = None if self.fallback_action is None else {
            str(k): float(v) for k, v in self.fallback_action.items()
        }
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
    assumptions_checked: tuple[AssumptionTag, ...] = ()
    certificate_horizon_steps: int | None = None
    source_domain: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
    ) -> "SafetyCertificate":
        raw = dict(payload)
        known_keys = {
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
            "assumptions_checked",
            "certificate_horizon_steps",
            "validity_horizon_H_t",
            "tau_t",
            "source_domain",
        }
        horizon = raw.get("certificate_horizon_steps", raw.get("validity_horizon_H_t", raw.get("tau_t")))
        assumption_tags = assumptions_checked or raw.get("assumptions_checked") or ()
        return cls(
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
            assumptions_checked=_tuple_tags(assumption_tags),
            certificate_horizon_steps=None if horizon is None else int(horizon),
            source_domain=str(source_domain or raw.get("source_domain", "")),
            extras={key: value for key, value in raw.items() if key not in known_keys},
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = {
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
    episode_risk_bound: dict[str, float]

    def __post_init__(self) -> None:
        self.safe_action = dict(self.safe_action)
        self.state = dict(self.state)
        self.step_risk_bound = float(self.step_risk_bound)
        self.episode_risk_bound = {
            str(key): float(value) for key, value in self.episode_risk_bound.items()
        }

    def to_mapping(self) -> dict[str, Any]:
        return {
            "certificate": self.certificate,
            "safe_action": dict(self.safe_action),
            "reliability_w": float(self.reliability.weight),
            "reliability_flags": dict(self.reliability.flags),
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

    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        ...

    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        ...

    def build_uncertainty_set(
        self,
        state: Mapping[str, Any],
        reliability_w: float,
        quantile: float,
        *,
        cfg: Mapping[str, Any],
        drift_flag: bool | None = None,
        prev_meta: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        ...

    def tighten_action_set(
        self,
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        *,
        cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...

    def repair_action(
        self,
        candidate_action: Mapping[str, Any],
        tightened_set: Mapping[str, Any],
        *,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        cfg: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        ...

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
    ) -> Mapping[str, Any]:
        ...


class ContractViolation(ValueError):
    """Raised when a domain adapter fails the universal theory contract."""


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
        }
