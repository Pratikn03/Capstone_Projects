"""Tests for retraining decision logic with DC3S health triggers."""
from __future__ import annotations

from orius.monitoring.retraining import retraining_decision


def test_retraining_decision_adds_dc3s_reasons() -> None:
    cfg = {"retraining": {"cadence_days": 365}}
    dc3s_health = {
        "triggered": True,
        "triggered_flags": ["intervention_rate", "low_reliability_rate", "drift_flag_rate"],
    }

    decision = retraining_decision(
        cfg,
        data_drift=False,
        model_drift=False,
        last_trained_path=None,
        dc3s_health=dc3s_health,
    )

    assert decision.retrain is True
    assert set(decision.reasons) == {
        "dc3s_intervention_spike",
        "dc3s_reliability_degradation",
        "dc3s_drift_persistence",
    }


def test_retraining_decision_ignores_non_mapped_dc3s_flags() -> None:
    cfg = {"retraining": {"cadence_days": 365}}
    dc3s_health = {
        "triggered": True,
        "triggered_flags": ["inflation_p95"],
    }

    decision = retraining_decision(
        cfg,
        data_drift=False,
        model_drift=False,
        last_trained_path=None,
        dc3s_health=dc3s_health,
    )

    assert decision.retrain is False
    assert decision.reasons == []
