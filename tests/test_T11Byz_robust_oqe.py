import pytest

from orius.dc3s.quality import (
    byzantine_reliability_error_bound,
    compute_reliability_robust,
    trimmed_mean_reliability,
)


def test_trimmed_mean_reliability_removes_byzantine_extremes():
    scores = [0.0, 0.9, 0.91, 0.92, 0.93, 1.0]

    robust = trimmed_mean_reliability(scores, byzantine_budget=1)
    naive = sum(scores) / len(scores)

    assert robust > naive
    assert 0.9 <= robust <= 0.95


def test_trimmed_mean_requires_byzantine_minority():
    with pytest.raises(ValueError, match="requires byzantine_budget < n/2"):
        trimmed_mean_reliability([0.2, 0.8], byzantine_budget=1)


def test_byzantine_error_bound_holds_for_honest_interval():
    honest = [0.88, 0.9, 0.92]
    observed = [0.0, *honest, 1.0]

    result = byzantine_reliability_error_bound(honest, observed, byzantine_budget=1)

    assert result["bound_satisfied"] is True
    assert result["absolute_error"] <= result["rho"]


def test_compute_reliability_robust_flags_spike_without_crashing():
    history = [10.0] * 8 + [30.0]

    w_t, flags = compute_reliability_robust(history, trim_frac=0.1)

    assert 0.05 <= w_t <= 1.0
    assert flags["robust"] is True
    assert flags["spike_detected"] is True
