"""Ambiguity-class helpers for no-free-safety theorems (T9)."""

from __future__ import annotations

from collections.abc import Callable, Iterable

State = object
Action = object
Observation = object


def build_ambiguity_class(
    states: Iterable[State], observation_fn: Callable[[State], Observation], observation: Observation
) -> list[State]:
    return [x for x in states if observation_fn(x) == observation]


def compute_common_safe_core(
    ambiguity_class: Iterable[State], safe_action_fn: Callable[[State], set[Action]]
) -> set[Action]:
    states = list(ambiguity_class)
    if not states:
        return set()
    core = set(safe_action_fn(states[0]))
    for x in states[1:]:
        core &= set(safe_action_fn(x))
    return core


def is_empty_safe_core(
    ambiguity_class: Iterable[State], safe_action_fn: Callable[[State], set[Action]]
) -> bool:
    return len(compute_common_safe_core(ambiguity_class, safe_action_fn)) == 0


def find_mandatory_release_counterexample(
    ambiguity_class: Iterable[State],
    safe_action_fn: Callable[[State], set[Action]],
    chosen_action: Action,
) -> State | None:
    for x in ambiguity_class:
        if chosen_action not in safe_action_fn(x):
            return x
    return None
