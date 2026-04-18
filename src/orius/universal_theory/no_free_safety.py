"""Canonical No Free Safety helpers used by the monograph and standalone paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

__all__ = ["CounterexampleResult", "construct_counterexample", "formal_principle_statement"]


@dataclass(frozen=True)
class CounterexampleResult:
    """Paired executions witnessing the No Free Safety principle."""

    fault_sequence_clean: np.ndarray
    fault_sequence_faulty: np.ndarray
    observation_stream_clean: np.ndarray
    observation_stream_faulty: np.ndarray
    true_trajectory_clean: np.ndarray
    true_trajectory_faulty: np.ndarray
    action_stream_clean: np.ndarray
    action_stream_faulty: np.ndarray
    safety_outcome_clean: bool
    safety_outcome_faulty: bool
    conclusion: str


def construct_counterexample(
    quality_ignorant_controller: Callable[[np.ndarray], np.ndarray | float],
    dynamics: Callable[[np.ndarray, np.ndarray | float], np.ndarray],
    safe_set_check: Callable[[np.ndarray], bool],
    initial_state: np.ndarray,
    horizon: int = 30,
    random_seed: int = 42,
) -> CounterexampleResult:
    """Construct two executions with identical observations and divergent true states."""

    rng = np.random.default_rng(random_seed)
    initial = np.asarray(initial_state, dtype=float).reshape(-1)
    frozen_observation = initial.copy()
    x_clean = initial.copy()
    x_faulty = initial.copy()

    observation_stream_clean: list[np.ndarray] = []
    observation_stream_faulty: list[np.ndarray] = []
    true_trajectory_clean: list[np.ndarray] = [x_clean.copy()]
    true_trajectory_faulty: list[np.ndarray] = [x_faulty.copy()]
    action_stream_clean: list[np.ndarray] = []
    action_stream_faulty: list[np.ndarray] = []

    for step in range(int(horizon)):
        obs_clean = frozen_observation.copy()
        obs_faulty = frozen_observation.copy()
        action_clean = np.asarray(quality_ignorant_controller(obs_clean), dtype=float).reshape(-1)
        action_faulty = np.asarray(quality_ignorant_controller(obs_faulty), dtype=float).reshape(-1)

        observation_stream_clean.append(obs_clean)
        observation_stream_faulty.append(obs_faulty)
        action_stream_clean.append(action_clean)
        action_stream_faulty.append(action_faulty)

        x_clean = np.asarray(dynamics(x_clean, action_clean), dtype=float).reshape(-1)
        disturbance = np.full_like(x_faulty, 0.12 + 0.02 * step) + rng.normal(0.0, 0.005, size=x_faulty.shape)
        x_faulty = np.asarray(dynamics(x_faulty, action_faulty), dtype=float).reshape(-1) + disturbance
        true_trajectory_clean.append(x_clean.copy())
        true_trajectory_faulty.append(x_faulty.copy())

    obs_clean_arr = np.asarray(observation_stream_clean, dtype=float)
    obs_faulty_arr = np.asarray(observation_stream_faulty, dtype=float)
    action_clean_arr = np.asarray(action_stream_clean, dtype=float)
    action_faulty_arr = np.asarray(action_stream_faulty, dtype=float)
    safe_clean = bool(safe_set_check(x_clean))
    safe_faulty = bool(safe_set_check(x_faulty))

    if not np.allclose(obs_clean_arr, obs_faulty_arr):
        raise RuntimeError("counterexample construction failed: observations diverged")
    if not np.allclose(action_clean_arr, action_faulty_arr):
        raise RuntimeError("counterexample construction failed: actions diverged")
    if safe_clean == safe_faulty:
        raise RuntimeError("counterexample construction failed: safety outcomes did not diverge")

    conclusion = (
        "The controller received identical observation streams and therefore emitted identical actions, "
        f"but the true trajectories diverged under the faulted branch. The clean run ended "
        f"{'safe' if safe_clean else 'unsafe'} while the faulted run ended "
        f"{'safe' if safe_faulty else 'unsafe'}, so observation-only control cannot provide a uniform "
        "true-state safety guarantee."
    )

    return CounterexampleResult(
        fault_sequence_clean=np.zeros(int(horizon), dtype=float),
        fault_sequence_faulty=np.ones(int(horizon), dtype=float),
        observation_stream_clean=obs_clean_arr,
        observation_stream_faulty=obs_faulty_arr,
        true_trajectory_clean=np.asarray(true_trajectory_clean, dtype=float),
        true_trajectory_faulty=np.asarray(true_trajectory_faulty, dtype=float),
        action_stream_clean=action_clean_arr,
        action_stream_faulty=action_faulty_arr,
        safety_outcome_clean=safe_clean,
        safety_outcome_faulty=safe_faulty,
        conclusion=conclusion,
    )


def formal_principle_statement() -> str:
    """Return the bounded, mechanism-independent principle statement."""

    return (
        "The No Free Safety Principle states that any controller mapping observations directly to actions, "
        "without runtime access to a measurable fidelity statistic, can be forced into observationally "
        "indistinguishable executions whose true-state safety outcomes differ. Uniform true-state safety "
        "therefore requires reliability-aware runtime information, not observation-only legality."
    )
