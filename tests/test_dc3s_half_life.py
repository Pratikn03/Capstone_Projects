from __future__ import annotations

import numpy as np

from orius.dc3s.half_life import (
    certificate_half_life,
    conservative_validity_horizon,
    validity_probability,
)


def test_half_life_matches_definition() -> None:
    result = certificate_half_life(initial_margin=10.0, disturbance_std=1.0)
    assert abs(validity_probability(result.half_life_steps, 10.0, 1.0) - 0.5) < 1e-6


def test_half_life_scales_quadratically_with_margin() -> None:
    small = certificate_half_life(initial_margin=1.0, disturbance_std=1.0)
    large = certificate_half_life(initial_margin=2.0, disturbance_std=1.0)
    assert abs(large.half_life_steps / small.half_life_steps - 4.0) < 1e-6


def test_validity_probability_is_monotone_decreasing() -> None:
    times = np.linspace(0.1, 100.0, 50)
    probs = [validity_probability(time, 5.0, 1.0) for time in times]
    assert all(probs[idx] >= probs[idx + 1] for idx in range(len(probs) - 1))


def test_conservative_horizon_is_shorter_than_half_life() -> None:
    half_life = certificate_half_life(initial_margin=6.0, disturbance_std=1.0)
    conservative = conservative_validity_horizon(6.0, 1.0, minimum_validity_probability=0.95)
    assert conservative < half_life.half_life_steps
