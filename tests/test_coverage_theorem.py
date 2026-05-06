"""
Tests for DC³S coverage theorem — formal guarantee verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from orius.dc3s.calibration import build_uncertainty_set
from orius.dc3s.coverage_theorem import (
    assert_coverage_guarantee,
    compute_empirical_coverage,
    inflation_lower_bound,
    verify_inflation_geq_one,
)

# ---------------------------------------------------------------------------
# verify_inflation_geq_one
# ---------------------------------------------------------------------------


def test_verify_inflation_geq_one_valid() -> None:
    """Inflation ≥ 1 should pass without error."""
    verify_inflation_geq_one(1.0)
    verify_inflation_geq_one(1.5)
    verify_inflation_geq_one(3.0)


def test_verify_inflation_geq_one_raises_when_below_one() -> None:
    """Inflation < 1 violates the coverage guarantee — must raise."""
    with pytest.raises(ValueError, match="state inflation must be >= 1"):
        verify_inflation_geq_one(0.5)


def test_verify_inflation_geq_one_boundary() -> None:
    """Exactly 1.0 is valid (within tolerance)."""
    verify_inflation_geq_one(1.0 - 1e-12)  # within tol=1e-9, should pass
    with pytest.raises(ValueError):
        verify_inflation_geq_one(1.0 - 1e-6)  # outside tol


# ---------------------------------------------------------------------------
# compute_empirical_coverage
# ---------------------------------------------------------------------------


def test_compute_empirical_coverage_all_covered() -> None:
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 1.5, 2.5])
    hi = np.array([2.0, 2.5, 3.5])
    result = compute_empirical_coverage(y, lo, hi)
    assert result["picp"] == pytest.approx(1.0)
    assert result["n_samples"] == 3


def test_compute_empirical_coverage_none_covered() -> None:
    y = np.array([10.0, 20.0])
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    result = compute_empirical_coverage(y, lo, hi)
    assert result["picp"] == pytest.approx(0.0)


def test_compute_empirical_coverage_partial() -> None:
    y = np.array([1.0, 5.0, 3.0, 7.0])
    lo = np.array([0.5, 0.0, 2.0, 0.0])
    hi = np.array([1.5, 2.0, 4.0, 5.0])
    result = compute_empirical_coverage(y, lo, hi)
    # y[0]=1.0 ∈ [0.5,1.5] ✓; y[1]=5.0 ∉ [0,2] ✗; y[2]=3.0 ∈ [2,4] ✓; y[3]=7.0 ∉ [0,5] ✗
    assert result["picp"] == pytest.approx(0.5)


def test_compute_empirical_coverage_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        compute_empirical_coverage(np.array([1.0, 2.0]), np.array([0.0]), np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# assert_coverage_guarantee
# ---------------------------------------------------------------------------


def test_assert_coverage_guarantee_passes_with_high_coverage() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, size=500)
    # Wide intervals guarantee coverage
    lo = y - 5.0
    hi = y + 5.0
    result = assert_coverage_guarantee(y, lo, hi, alpha=0.10, tolerance=0.02)
    assert result["passed"] is True
    assert result["picp"] == pytest.approx(1.0)


def test_assert_coverage_guarantee_fails_with_low_coverage() -> None:
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, size=500)
    # Intervals far from the true values → essentially 0% coverage
    lo = np.full(500, 100.0)
    hi = np.full(500, 101.0)
    with pytest.raises(AssertionError, match="Empirical coverage="):
        assert_coverage_guarantee(y, lo, hi, alpha=0.10, tolerance=0.02)


# ---------------------------------------------------------------------------
# Core theorem: DC³S inflated intervals are at least as wide as base intervals
# ---------------------------------------------------------------------------


def test_dc3s_inflation_monotonically_widens_intervals() -> None:
    """
    Formal theorem check: for infl ≥ 1, the DC³S interval is a superset of
    the base conformal interval at every sample.
    """
    rng = np.random.default_rng(42)
    yhat = rng.normal(500, 50, size=100)
    q = np.full(100, 30.0)  # base conformal half-width
    w_t_values = rng.uniform(0.05, 1.0, size=100)
    drift_flags = rng.uniform(size=100) < 0.1

    base_lower = yhat - q
    base_upper = yhat + q

    for i in range(100):
        cfg = {"k_quality": 0.8, "k_drift": 0.6, "infl_max": 3.0}
        lo_dc3s, hi_dc3s, meta = build_uncertainty_set(
            yhat=yhat[i],
            q=q[i],
            w_t=w_t_values[i],
            drift_flag=bool(drift_flags[i]),
            cfg=cfg,
        )
        inflation = meta["inflation"]
        # Guarantee: inflation ≥ 1
        verify_inflation_geq_one(inflation)
        # Guarantee: DC³S interval ⊇ base interval
        assert lo_dc3s[0] <= base_lower[i] + 1e-9, (
            f"DC³S lower {lo_dc3s[0]:.4f} > base lower {base_lower[i]:.4f} at i={i}"
        )
        assert hi_dc3s[0] >= base_upper[i] - 1e-9, (
            f"DC³S upper {hi_dc3s[0]:.4f} < base upper {base_upper[i]:.4f} at i={i}"
        )


def test_dc3s_recovers_base_at_perfect_telemetry_no_drift() -> None:
    """Corollary: when w_t=1 and drift=False, infl=1 and DC³S = base interval."""
    np.array([500.0])
    np.array([30.0])
    cfg = {"k_quality": 0.8, "k_drift": 0.6, "infl_max": 3.0}
    lo, hi, meta = build_uncertainty_set(yhat=500.0, q=30.0, w_t=1.0, drift_flag=False, cfg=cfg)
    assert meta["inflation"] == pytest.approx(1.0)
    assert lo[0] == pytest.approx(500.0 - 30.0)
    assert hi[0] == pytest.approx(500.0 + 30.0)


# ---------------------------------------------------------------------------
# inflation_lower_bound
# ---------------------------------------------------------------------------


def test_inflation_lower_bound_is_always_one() -> None:
    """The minimum possible inflation (w_t=1, no drift) is always exactly 1."""
    for k_q in [0.0, 0.5, 0.8, 1.2]:
        for k_d in [0.0, 0.3, 0.9]:
            lb = inflation_lower_bound(k_quality=k_q, k_drift=k_d)
            assert lb == pytest.approx(1.0), f"Expected 1.0 for k_q={k_q}, k_d={k_d}, got {lb}"
