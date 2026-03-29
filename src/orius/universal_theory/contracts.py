"""Typed contracts for the ORIUS degraded-observation safety kernel.

The legacy universal runtime exposed loosely typed ``dict[str, Any]`` payloads.
This module keeps the runtime interoperable with those adapters while making
the theorem-facing objects explicit and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
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
    episode_risk_bound: dict[str, Any]
    contract_checks: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.safe_action = dict(self.safe_action)
        self.state = dict(self.state)
        self.step_risk_bound = float(self.step_risk_bound)
        self.episode_risk_bound = {
            str(key): float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
            for key, value in self.episode_risk_bound.items()
        }
        self.contract_checks = {
            str(key): dict(value) if isinstance(value, Mapping) else value
            for key, value in self.contract_checks.items()
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
            "contract_checks": dict(self.contract_checks),
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
                prev_meta=None,
            )
            degraded_inflation = _extract_inflation(
                degraded_uncertainty,
                fallback=_extract_inflation(degraded_meta, fallback=float(uncertainty_set.inflation)),
            )
            record(
                "monotone_inflation",
                degraded_inflation is not None and degraded_inflation + 1e-9 >= float(uncertainty_set.inflation),
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
