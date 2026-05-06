"""ORIUS-Bench replayable fault-schedule engine.

Given a seed, the engine produces a deterministic sequence of fault events
(sensor bias, blackout, noise spikes) that benchmark controllers must
handle.  The same seed always yields the same fault scenario, enabling
reproducible leaderboard comparisons.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FaultEvent:
    """A single injected fault."""

    step: int
    kind: str  # "bias", "blackout", "noise", "stuck_sensor"
    params: dict[str, Any] = field(default_factory=dict)
    duration: int = 1

    def affects_step(self, t: int) -> bool:
        return self.step <= t < self.step + self.duration


@dataclass
class FaultSchedule:
    """Deterministic fault schedule for a benchmark episode."""

    seed: int
    events: list[FaultEvent] = field(default_factory=list)
    horizon: int = 96

    @property
    def digest(self) -> str:
        """SHA-256 fingerprint for reproducibility audit."""
        payload = f"{self.seed}|{self.horizon}|{len(self.events)}"
        for e in self.events:
            payload += f"|{e.step},{e.kind},{e.duration}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def generate_fault_schedule(
    seed: int,
    horizon: int = 96,
    *,
    fault_rate: float = 0.15,
    blackout_prob: float = 0.05,
    max_blackout_duration: int = 12,
) -> FaultSchedule:
    """Generate a deterministic fault schedule from *seed*.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    horizon : int
        Total number of steps in the episode.
    fault_rate : float
        Probability of any single-step sensor fault per step.
    blackout_prob : float
        Per-step probability of a multi-step blackout starting.
    max_blackout_duration : int
        Maximum duration of a blackout event.

    Returns
    -------
    FaultSchedule
        Deterministic, replayable schedule.
    """
    rng = np.random.default_rng(seed)
    events: list[FaultEvent] = []

    t = 0
    while t < horizon:
        # Check blackout first
        if rng.random() < blackout_prob:
            dur = int(rng.integers(1, max_blackout_duration + 1))
            dur = min(dur, horizon - t)
            events.append(FaultEvent(step=t, kind="blackout", duration=dur))
            t += dur
            continue

        # Single-step faults
        if rng.random() < fault_rate:
            kind = rng.choice(["bias", "noise", "stuck_sensor"])
            params: dict[str, Any] = {}
            if kind == "bias":
                params["magnitude"] = float(rng.normal(0, 5))
            elif kind == "noise":
                params["sigma"] = float(rng.uniform(1, 10))
            elif kind == "stuck_sensor":
                params["frozen_value"] = float(rng.uniform(0.1, 0.9))
            events.append(FaultEvent(step=t, kind=kind, params=params, duration=1))
        t += 1

    return FaultSchedule(seed=seed, events=events, horizon=horizon)


def active_faults(schedule: FaultSchedule, step: int) -> list[FaultEvent]:
    """Return all faults active at *step*."""
    return [e for e in schedule.events if e.affects_step(step)]


def apply_faults(
    true_state: dict[str, float],
    faults: Sequence[FaultEvent],
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Corrupt *true_state* according to active *faults*.

    Returns a new dict (never mutates the original).
    """
    observed = dict(true_state)
    if rng is None:
        rng = np.random.default_rng(42)

    # Build a simple history ring-buffer from params for replay fault

    for f in faults:
        if f.kind == "blackout":
            # During blackout the controller receives stale NaN observations
            observed = {k: float("nan") for k in observed}
            break  # blackout overrides everything
        if f.kind == "bias" and "soc" in observed:
            observed["soc"] = observed["soc"] + f.params.get("magnitude", 0.0)
        elif f.kind == "noise" and "soc" in observed:
            sigma = f.params.get("sigma", 5.0)
            observed["soc"] = observed["soc"] + float(rng.normal(0, sigma))
        elif f.kind == "stuck_sensor" and "soc" in observed:
            observed["soc"] = f.params.get("frozen_value", 0.5)
        elif f.kind == "replay":
            # Return a stale reading from k_steps_ago.
            # The caller must pass a history buffer via f.params["history"]
            # (list of past state dicts, most-recent last).
            k = int(f.params.get("k_steps_ago", 5))
            past: list[dict[str, float]] = list(f.params.get("history", []))
            if past and k <= len(past):
                stale = past[-k]
                for key in observed:
                    if key in stale:
                        observed[key] = float(stale[key])
        elif f.kind == "coordinated_spoof":
            # Apply a small systematic bias <= 10% of normal range per signal.
            # Designed to evade spike detection by staying within plausible bounds.
            spoof_frac = float(f.params.get("spoof_fraction", 0.05))
            normal_range = float(f.params.get("normal_range", 1.0))
            magnitude = spoof_frac * normal_range
            for key in list(observed.keys()):
                val = observed[key]
                with contextlib.suppress(TypeError, ValueError):
                    observed[key] = float(val) + magnitude

    return observed


def replay_schedule(seed: int, horizon: int = 96, **kwargs) -> FaultSchedule:
    """Convenience alias: identical schedule from same seed."""
    return generate_fault_schedule(seed, horizon, **kwargs)
