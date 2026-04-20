from __future__ import annotations

import numpy as np

from orius.universal_theory.no_free_safety import construct_counterexample, formal_principle_statement


def test_construct_counterexample_uses_identical_observations_and_actions() -> None:
    result = construct_counterexample(
        quality_ignorant_controller=lambda obs: np.array([0.0]) if obs[0] < 5.0 else np.array([-0.1]),
        dynamics=lambda state, action: state + np.asarray(action, dtype=float),
        safe_set_check=lambda state: state[0] <= 2.0,
        initial_state=np.array([0.0]),
        horizon=15,
    )

    assert np.allclose(result.observation_stream_clean, result.observation_stream_faulty)
    assert np.allclose(result.action_stream_clean, result.action_stream_faulty)
    assert not np.allclose(
        result.observation_stream_clean,
        np.repeat(result.observation_stream_clean[:1], result.observation_stream_clean.shape[0], axis=0),
    )
    assert not np.allclose(result.true_trajectory_clean, result.true_trajectory_faulty)
    assert result.safety_outcome_clean != result.safety_outcome_faulty
    assert (not result.safety_outcome_clean) or (not result.safety_outcome_faulty)
    assert result.first_unsafe_step_faulty is not None
    assert result.final_state_safe_clean is True


def test_formal_principle_statement_mentions_reliability() -> None:
    statement = formal_principle_statement().lower()
    assert "reliability" in statement
    assert "observation-only" in statement
