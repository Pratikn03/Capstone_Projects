"""ORIUS-Bench domain adapter interface.

Every benchmark track implements this interface. The battery track delegates
to the existing DC3S/CPSBench code; the navigation track implements a
simple 2D robot.

The seven ORIUS metrics are computed uniformly across domains via
``metrics_engine.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any


class BenchmarkAdapter(ABC):
    """Abstract interface for a benchmark domain track."""

    @abstractmethod
    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        """Reset the environment and return the initial state."""

    @abstractmethod
    def true_state(self) -> Mapping[str, Any]:
        """Return the hidden true state of the system."""

    @abstractmethod
    def observe(
        self, true_state: Mapping[str, Any], fault: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        """Apply optional fault injection and return the observed state."""

    @abstractmethod
    def safe_action_set(self, state: Mapping[str, Any], uncertainty: Mapping[str, Any]) -> Mapping[str, Any]:
        """Compute the safe action set given state and uncertainty."""

    @abstractmethod
    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        """Advance the system by one step, return the new true state."""

    @abstractmethod
    def compute_useful_work(self, trajectory: Sequence[Mapping[str, Any]]) -> float:
        """Compute domain-specific useful work over a trajectory."""

    @abstractmethod
    def check_violation(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Check if the current state violates safety constraints.
        Returns {"violated": bool, "severity": float}.
        """

    def true_constraint_violated(self, state: Mapping[str, Any]) -> bool:
        """Canonical true-state safety predicate for the benchmark harness."""
        return bool(self.check_violation(state).get("violated", False))

    def observed_constraint_satisfied(self, observed_state: Mapping[str, Any]) -> bool | None:
        """Canonical observation-space safety predicate.

        Returns ``None`` when the observed state cannot be evaluated under the
        domain's constraint model, for example under severe telemetry corruption.
        """
        try:
            return not bool(self.check_violation(observed_state).get("violated", False))
        except Exception:
            return None

    def constraint_margin(self, state: Mapping[str, Any]) -> float | None:
        """Optional scalar safety margin used by the domain-agnostic schema."""
        return None

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return the domain identifier string."""
