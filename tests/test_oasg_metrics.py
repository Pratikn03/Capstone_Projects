from __future__ import annotations

import numpy as np

from orius.orius_bench.oasg_metrics import build_submission_domain_surfaces, compute_oasg_signature


def test_signature_zero_when_no_gap() -> None:
    true_states = np.column_stack([np.linspace(-1.0, 1.0, 1000), np.zeros(1000)])
    observations = true_states.copy()
    reliability = np.ones(1000, dtype=float)

    result = compute_oasg_signature(
        true_states=true_states,
        observations=observations,
        reliability_scores=reliability,
        safe_set_check=lambda state: abs(state[0]) < 5.0,
        distance_to_boundary=lambda state: abs(state[0]) - 5.0,
    )

    assert result.signature == 0.0
    assert result.exposure_rate == 0.0
    assert result.severity == 0.0


def test_signature_equals_exposure_times_severity() -> None:
    true_states = np.array([[4.0], [6.0], [8.0], [3.0]], dtype=float)
    observations = np.array([[4.0], [4.5], [4.5], [3.0]], dtype=float)
    reliability = np.array([1.0, 0.4, 0.2, 0.7], dtype=float)

    result = compute_oasg_signature(
        true_states=true_states,
        observations=observations,
        reliability_scores=reliability,
        safe_set_check=lambda state: state[0] < 5.0,
        distance_to_boundary=lambda state: state[0] - 5.0,
    )

    assert np.isclose(result.signature, result.exposure_rate * result.severity)


def test_severity_excludes_non_degraded_rows() -> None:
    true_states = np.array([[6.0], [7.0], [8.0]], dtype=float)
    observations = np.array([[4.5], [4.5], [4.5]], dtype=float)
    reliability = np.array([1.0, 0.4, 0.4], dtype=float)

    result = compute_oasg_signature(
        true_states=true_states,
        observations=observations,
        reliability_scores=reliability,
        safe_set_check=lambda state: state[0] < 5.0,
        distance_to_boundary=lambda state: state[0] - 5.0,
    )

    assert np.isclose(result.exposure_rate, 2.0 / 3.0)
    assert np.isclose(result.severity, 2.5)
    assert np.isclose(result.signature, (2.0 + 3.0) / 3.0)


def test_submission_domain_surfaces_are_battery_and_av_only() -> None:
    surfaces = build_submission_domain_surfaces()
    assert set(surfaces) == {"Battery", "Autonomous Vehicles"}
