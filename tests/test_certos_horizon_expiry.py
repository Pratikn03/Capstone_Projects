from orius.dc3s.temporal_theorems import (
    forward_reachable_tube,
    validate_certificate_horizon,
)


def test_forward_reachable_tube_certifies_safe_prefix() -> None:
    tube = forward_reachable_tube(
        [(4.0, 5.0)],
        [0.0, 0.2, 0.2],
        drift_radius_per_step=0.1,
    )

    result = validate_certificate_horizon(tube, safe_lower=0.0, safe_upper=6.0)

    assert result["valid"] is True
    assert result["horizon"] == 3
    assert result["fails_closed"] is False


def test_release_fails_closed_when_horizon_less_than_one() -> None:
    tube = forward_reachable_tube(
        [(5.9, 6.0)],
        [0.4],
        drift_radius_per_step=0.0,
    )

    result = validate_certificate_horizon(tube, safe_lower=0.0, safe_upper=6.0)

    assert result["horizon"] == 0
    assert result["fails_closed"] is True
