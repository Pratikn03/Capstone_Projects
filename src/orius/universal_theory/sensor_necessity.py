"""Sensor-necessity helpers under adapter semantics (Tsensor)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from orius.universal_theory.ambiguity import compute_common_safe_core


def sensor_ablation(observation: dict[str, float], removed_keys: list[str]) -> dict[str, float]:
    return {k: v for k, v in observation.items() if k not in set(removed_keys)}


def safe_core_after_sensor_drop(
    states: Iterable[object], safe_action_fn: Callable[[object], set[object]]
) -> set[object]:
    return compute_common_safe_core(states, safe_action_fn)


def critical_sensor_test(states: Iterable[object], safe_action_fn: Callable[[object], set[object]]) -> bool:
    return len(safe_core_after_sensor_drop(states, safe_action_fn)) == 0
