"""ORIUS-Bench controller API.

Controllers submit to the benchmark by implementing ``ControllerAPI``.
The benchmark runner calls ``propose_action`` each step and logs the
result through the certificate and audit path.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class ControllerAPI(ABC):
    """Abstract controller interface for ORIUS-Bench."""

    @abstractmethod
    def propose_action(
        self,
        observed_state: Mapping[str, Any],
        uncertainty: Mapping[str, Any] | None = None,
        certificate_state: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Propose an action given the observed state and uncertainty."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Controller identifier."""


class NominalController(ControllerAPI):
    """Baseline: uses observation directly, ignores uncertainty."""

    def __init__(self, default_action: Mapping[str, Any] | None = None):
        self._default = dict(default_action or {"charge_mw": 0.0, "discharge_mw": 50.0})

    def propose_action(self, observed_state, uncertainty=None, certificate_state=None):
        return dict(self._default)

    @property
    def name(self):
        return "nominal"


class RobustController(ControllerAPI):
    """Uses uncertainty intervals to conservatively choose actions."""

    def __init__(self, safety_factor: float = 0.5):
        self._safety_factor = safety_factor

    def propose_action(self, observed_state, uncertainty=None, certificate_state=None):
        # Conservative: reduce action by safety factor
        return {"charge_mw": 0.0, "discharge_mw": 50.0 * self._safety_factor}

    @property
    def name(self):
        return "robust"


class DC3SController(ControllerAPI):
    """DC3S-like controller that respects certificate state."""

    def propose_action(self, observed_state, uncertainty=None, certificate_state=None):
        if certificate_state and certificate_state.get("fallback_required"):
            return {"charge_mw": 0.0, "discharge_mw": 0.0}
        return {"charge_mw": 0.0, "discharge_mw": 80.0}

    @property
    def name(self):
        return "dc3s"


class FallbackController(ControllerAPI):
    """Simple ramp-down fallback controller."""

    def __init__(self):
        self._step = 0

    def propose_action(self, observed_state, uncertainty=None, certificate_state=None):
        self._step += 1
        decay = max(0.0, 1.0 - self._step * 0.05)
        return {"charge_mw": 0.0, "discharge_mw": 80.0 * decay}

    @property
    def name(self):
        return "fallback"


class NaiveController(ControllerAPI):
    """Deliberately naive: always max discharge, ignores everything."""

    def __init__(self, max_discharge: float = 200.0):
        self._max = max_discharge

    def propose_action(self, observed_state, uncertainty=None, certificate_state=None):
        return {"charge_mw": 0.0, "discharge_mw": self._max}

    @property
    def name(self):
        return "naive"


def domain_aware_action(
    domain_name: str,
    base_action: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Map battery-style action to domain-specific action."""
    effort = min(1.0, max(0.0, float(base_action.get("discharge_mw", 50)) / 100.0))
    if domain_name == "battery":
        return dict(base_action)
    if domain_name == "navigation":
        return {"ax": 0.3 * effort, "ay": 0.2 * effort}
    if domain_name == "industrial":
        return {"power_setpoint_mw": 400.0 + 80.0 * effort}
    if domain_name == "healthcare":
        return {"alert_level": 0.2 + 0.5 * effort}
    if domain_name == "aerospace":
        return {"throttle": 0.5 + 0.3 * effort, "bank_deg": 2.0}
    if domain_name == "vehicle":
        return {"acceleration_mps2": 0.5 * effort}
    return dict(base_action)


class DomainAwareController(ControllerAPI):
    """Wraps a base controller and maps actions to the track's domain."""

    def __init__(self, base: ControllerAPI, domain_name: str):
        self._base = base
        self._domain = domain_name

    def propose_action(self, observed_state, uncertainty=None, certificate_state=None):
        base_action = self._base.propose_action(observed_state, uncertainty, certificate_state)
        return domain_aware_action(self._domain, base_action)

    @property
    def name(self):
        return self._base.name
