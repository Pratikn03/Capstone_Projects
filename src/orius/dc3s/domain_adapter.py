"""DomainAdapter abstraction for ORIUS/DC3S pipeline stages.

This module introduces the canonical code-level interface for domain-specific
runtime adapters:

- ingest_telemetry: raw packet -> parsed state vector z_t
- compute_oqe: state history -> reliability score w_t and auxiliary flags
- build_uncertainty_set: (state, w_t, conformal quantile) -> uncertainty object U_t
- tighten_action_set: uncertainty + constraints -> safe action set A_t
- repair_action: candidate action + A_t -> safe action and repair metadata
- emit_certificate: per-step record carrying DC3S fields and hashes

Concrete domains (battery, vehicles, HVAC, etc.) should subclass
DomainAdapter and implement these methods with domain-specific logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence


class DomainAdapter(ABC):
    """Abstract interface for domain-specific ORIUS/DC3S adapters."""

    def capability_profile(self) -> Mapping[str, Any]:
        """Return conservative portability metadata for cross-domain evaluation.

        Concrete adapters may override this to expose a richer domain contract.
        The default keeps optional portability layers disabled unless an adapter
        explicitly opts in.
        """
        return {
            "safety_surface_type": "unknown",
            "repair_mode": "projection",
            "fallback_mode": "unspecified",
            "supports_multi_agent_eval": False,
            "supports_certos_eval": False,
        }

    def true_constraint_violated(self, state: Mapping[str, Any]) -> bool | None:
        """Optional benchmark-facing true-state violation predicate.

        Runtime adapters are not required to participate in ORIUS-Bench
        directly, but exposing the hook keeps the domain contract aligned with
        the canonical thesis-facing benchmark semantics.
        """
        return None

    def observed_constraint_satisfied(self, observed_state: Mapping[str, Any]) -> bool | None:
        """Optional benchmark-facing observed-state satisfiability predicate."""
        return None

    def constraint_margin(self, state: Mapping[str, Any]) -> float | None:
        """Optional scalar safety margin for domain-agnostic reporting."""
        return None

    @abstractmethod
    def ingest_telemetry(self, raw_packet: Mapping[str, Any]) -> Mapping[str, Any]:
        """Parse raw telemetry into a numeric/structured state vector z_t."""

    @abstractmethod
    def compute_oqe(
        self,
        state: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[float, Mapping[str, Any]]:
        """Compute observation quality w_t in [0, 1] and auxiliary flags."""

    @abstractmethod
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
        """Construct an uncertainty object U_t and metadata from state, w_t, and q_t."""

    @abstractmethod
    def tighten_action_set(
        self,
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        *,
        cfg: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return a representation of the tightened safe action set A_t."""

    @abstractmethod
    def repair_action(
        self,
        candidate_action: Mapping[str, Any],
        tightened_set: Mapping[str, Any],
        *,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
        constraints: Mapping[str, Any],
        cfg: Mapping[str, Any],
    ) -> tuple[Mapping[str, float], Mapping[str, Any]]:
        """Project the candidate action into the safe set and return (safe_action, meta)."""

    @abstractmethod
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
        """Emit a per-step certificate payload for persistence and audit."""
