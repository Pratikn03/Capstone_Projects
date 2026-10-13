"""kga.guards.state_backend -- Distributed state backend for multi-worker circuit breaking."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union


@dataclass
class DistributedCircuitState:
    state: str  # "closed", "open", "half_open"
    failure_count: int
    trip_count: int
    last_trip_reason: Optional[str]
    last_state_change: float
    evalue_wealth: float
    evalue_step: int


class StateBackend(Protocol):
    """Protocol for state synchronization across multi-worker serving instances."""

    def load_state(self) -> DistributedCircuitState:
        ...

    def save_state(self, state: DistributedCircuitState) -> None:
        ...


class InMemoryStateBackend:
    """Thread-safe in-memory state backend for single-process serving."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = DistributedCircuitState(
            state="closed",
            failure_count=0,
            trip_count=0,
            last_trip_reason=None,
            last_state_change=time.monotonic(),
            evalue_wealth=1.0,
            evalue_step=0,
        )

    def load_state(self) -> DistributedCircuitState:
        with self._lock:
            return DistributedCircuitState(
                state=self._state.state,
                failure_count=self._state.failure_count,
                trip_count=self._state.trip_count,
                last_trip_reason=self._state.last_trip_reason,
                last_state_change=self._state.last_state_change,
                evalue_wealth=self._state.evalue_wealth,
                evalue_step=self._state.evalue_step,
            )

    def save_state(self, state: DistributedCircuitState) -> None:
        with self._lock:
            self._state = state


class FileStateBackend:
    """Inter-process file-backed state backend with atomic file locking for multi-pod Kubernetes clusters."""

    def __init__(self, state_file_path: Union[str, Path]) -> None:
        self.state_file = Path(state_file_path)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.state_file.with_suffix(".lock")
        self._thread_lock = threading.Lock()

        if not self.state_file.exists():
            initial = DistributedCircuitState(
                state="closed",
                failure_count=0,
                trip_count=0,
                last_trip_reason=None,
                last_state_change=time.time(),
                evalue_wealth=1.0,
                evalue_step=0,
            )
            self.save_state(initial)

    def load_state(self) -> DistributedCircuitState:
        with self._thread_lock:
            if not self.state_file.exists():
                return DistributedCircuitState(
                    state="closed",
                    failure_count=0,
                    trip_count=0,
                    last_trip_reason=None,
                    last_state_change=time.time(),
                    evalue_wealth=1.0,
                    evalue_step=0,
                )
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return DistributedCircuitState(
                    state=data["state"],
                    failure_count=int(data["failure_count"]),
                    trip_count=int(data["trip_count"]),
                    last_trip_reason=data.get("last_trip_reason"),
                    last_state_change=float(data["last_state_change"]),
                    evalue_wealth=float(data.get("evalue_wealth", 1.0)),
                    evalue_step=int(data.get("evalue_step", 0)),
                )
            except Exception:
                # Fail-closed safe fallback
                return DistributedCircuitState(
                    state="open",
                    failure_count=999,
                    trip_count=1,
                    last_trip_reason="Corrupted state file fallback",
                    last_state_change=time.time(),
                    evalue_wealth=1.0,
                    evalue_step=0,
                )

    def save_state(self, state: DistributedCircuitState) -> None:
        with self._thread_lock:
            temp_file = self.state_file.with_suffix(".tmp")
            data = {
                "state": state.state,
                "failure_count": state.failure_count,
                "trip_count": state.trip_count,
                "last_trip_reason": state.last_trip_reason,
                "last_state_change": state.last_state_change,
                "evalue_wealth": state.evalue_wealth,
                "evalue_step": state.evalue_step,
            }
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.state_file)
