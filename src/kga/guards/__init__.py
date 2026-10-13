"""kga.guards -- Fail-safe guards and circuit breakers for production model adaptation."""

from kga.guards.circuit_breaker import CircuitBreaker, CircuitBreakerStatus, CircuitState
from kga.guards.evidence_support import EvidenceSupportGuard, SupportAssessment
from kga.guards.numerical_health import HealthAssessment, NumericalHealthGuard
from kga.guards.state_backend import DistributedCircuitState, FileStateBackend, InMemoryStateBackend, StateBackend
from kga.guards.streaming_monitor import MonitorStatus, StreamingDriftMonitor

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerStatus",
    "CircuitState",
    "EvidenceSupportGuard",
    "SupportAssessment",
    "NumericalHealthGuard",
    "HealthAssessment",
    "StreamingDriftMonitor",
    "MonitorStatus",
    "StateBackend",
    "InMemoryStateBackend",
    "FileStateBackend",
    "DistributedCircuitState",
]
