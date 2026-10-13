"""kga.guards.circuit_breaker -- Stateful circuit breaker and latch for model adaptation."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Optional

from kga.guards.state_backend import DistributedCircuitState, StateBackend


class CircuitState(str, enum.Enum):
    """Lifecycle states of the adaptation circuit breaker."""
    CLOSED = "closed"        # Normal operational state: adaptation attempts allowed
    OPEN = "open"            # Tripped: all adaptation halted, 100% traffic routed to frozen base
    HALF_OPEN = "half_open"  # Cooldown trial: testing stability with limited traffic


@dataclass
class CircuitBreakerStatus:
    """Snapshot of circuit breaker status."""
    state: CircuitState
    failure_count: int
    trip_count: int
    last_trip_reason: Optional[str]
    last_trip_timestamp: Optional[float]
    time_in_current_state: float


class CircuitBreaker:
    """High-reliability circuit breaker governing test-time model adaptation.
    
    Adheres to the fail-closed principle: when open, live traffic is unconditionally
    routed to the frozen baseline model (f0) with zero adaptation attempts.
    Optionally synchronizes with a StateBackend for multi-worker container clusters.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_cooldown_seconds: float = 60.0,
        half_open_success_threshold: int = 2,
        backend: Optional[StateBackend] = None,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_cooldown_seconds = max(0.0, float(recovery_cooldown_seconds))
        self.half_open_success_threshold = max(1, int(half_open_success_threshold))
        self.backend = backend

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._half_open_success_count: int = 0
        self._trip_count: int = 0
        self._last_trip_reason: Optional[str] = None
        self._last_state_change: float = time.monotonic()

        if self.backend is not None:
            self._sync_from_backend()

    @property
    def state(self) -> CircuitState:
        """Current state, accounting for automatic cooldown expiration."""
        if self.backend is not None:
            self._sync_from_backend()

        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_state_change
            if elapsed >= self.recovery_cooldown_seconds > 0.0:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    def allow_adaptation_attempt(self) -> bool:
        """Query whether the gateway should attempt candidate adaptation."""
        current = self.state
        return current in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def trip(self, reason: str) -> None:
        """Immediately force the circuit breaker into OPEN state."""
        self._last_trip_reason = reason
        self._trip_count += 1
        self._transition_to(CircuitState.OPEN)

    def record_success(self) -> None:
        """Record a successful, certified adaptation execution."""
        current = self.state
        if current == CircuitState.HALF_OPEN:
            self._half_open_success_count += 1
            if self._half_open_success_count >= self.half_open_success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif current == CircuitState.CLOSED:
            self._failure_count = 0
            if self.backend is not None:
                self._sync_to_backend()

    def record_failure(self, reason: str) -> None:
        """Record an anomaly, timeout, or guard trip."""
        current = self.state
        self._failure_count += 1
        if current == CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
            self.trip(f"Failure threshold exceeded ({self._failure_count}/{self.failure_threshold}): {reason}")
        elif self.backend is not None:
            self._sync_to_backend()

    def reset(self) -> None:
        """Manual operator reset back to healthy CLOSED state."""
        self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0
        self._half_open_success_count = 0
        self._last_trip_reason = None
        if self.backend is not None:
            self._sync_to_backend()

    def get_status(self) -> CircuitBreakerStatus:
        """Return full status snapshot."""
        return CircuitBreakerStatus(
            state=self.state,
            failure_count=self._failure_count,
            trip_count=self._trip_count,
            last_trip_reason=self._last_trip_reason,
            last_trip_timestamp=self._last_state_change if self._trip_count > 0 else None,
            time_in_current_state=time.monotonic() - self._last_state_change,
        )

    def _transition_to(self, new_state: CircuitState) -> None:
        self._state = new_state
        self._last_state_change = time.monotonic()
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_success_count = 0

        if self.backend is not None:
            self._sync_to_backend()

    def _sync_from_backend(self) -> None:
        if self.backend is None:
            return
        st = self.backend.load_state()
        if st.state == "open":
            self._state = CircuitState.OPEN
        elif st.state == "half_open":
            self._state = CircuitState.HALF_OPEN
        else:
            self._state = CircuitState.CLOSED
        self._failure_count = st.failure_count
        self._trip_count = st.trip_count
        self._last_trip_reason = st.last_trip_reason

    def _sync_to_backend(self) -> None:
        if self.backend is None:
            return
        st = DistributedCircuitState(
            state=self._state.value,
            failure_count=self._failure_count,
            trip_count=self._trip_count,
            last_trip_reason=self._last_trip_reason,
            last_state_change=time.time(),
            evalue_wealth=1.0,
            evalue_step=0,
        )
        self.backend.save_state(st)
