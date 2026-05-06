"""Unit tests for DC3S Page-Hinkley drift detector."""

from __future__ import annotations

from orius.dc3s.drift import PageHinkleyDetector


def test_page_hinkley_triggers_and_cooldown():
    detector = PageHinkleyDetector.from_state(
        None,
        cfg={
            "ph_delta": 0.01,
            "ph_lambda": 0.5,
            "warmup_steps": 3,
            "cooldown_steps": 2,
        },
    )

    # Warmup with small residuals.
    for _ in range(3):
        out = detector.update(0.1)
        assert out["drift"] is False

    # A large residual should trigger drift.
    triggered = detector.update(4.0)
    assert triggered["drift"] is True
    assert triggered["cooldown_remaining"] >= 1

    # During cooldown, drift should not retrigger immediately.
    during_cooldown = detector.update(5.0)
    assert during_cooldown["drift"] is False
