"""Tests for RSS safe-gap model (src/orius/vehicles/rss_safety.py).

Verifies the Shalev-Shwartz et al. 2017 longitudinal formula:
    gap_safe = v_ego·t_resp + v_ego²/(2·a_min) − v_lead²/(2·a_max)
    clamped to ≥ 0.
"""

import math

import numpy as np
import pytest

from orius.vehicles.rss_safety import (
    RssParameters,
    rss_safe_gap,
    rss_safe_gap_vec,
    rss_violation,
    rss_violation_vec,
)

# Default RSS parameters
_T = 0.75
_A_EGO = 4.0
_A_LEAD = 6.0


class TestRssSafeGap:
    """Scalar rss_safe_gap tests."""

    def test_equal_speeds_returns_nonzero(self):
        """When v_ego == v_lead, gap > 0 because of reaction distance."""
        v = 20.0
        gap = rss_safe_gap(v, v)
        # reaction + ego_brake - lead_brake
        expected = v * _T + v**2 / (2 * _A_EGO) - v**2 / (2 * _A_LEAD)
        assert gap == pytest.approx(expected)
        assert gap > 0  # reaction term dominates

    def test_lead_stopped(self):
        """v_lead=0 → gap = reaction_dist + halting_dist."""
        v_ego = 30.0
        gap = rss_safe_gap(v_ego, 0.0)
        expected = v_ego * _T + v_ego**2 / (2 * _A_EGO)
        assert gap == pytest.approx(expected)

    def test_ego_stopped(self):
        """v_ego=0 → gap = 0 (clamped, since −lead_brake² < 0)."""
        gap = rss_safe_gap(0.0, 30.0)
        assert gap == 0.0

    def test_both_stopped(self):
        gap = rss_safe_gap(0.0, 0.0)
        assert gap == 0.0

    def test_monotonic_in_ego_speed(self):
        """Gap should increase as ego goes faster (lead speed fixed)."""
        v_lead = 15.0
        speeds = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        gaps = [rss_safe_gap(v, v_lead) for v in speeds]
        for i in range(len(gaps) - 1):
            assert gaps[i + 1] > gaps[i], f"Not monotonic at {speeds[i+1]}"

    def test_clamp_to_zero(self):
        """When lead is much faster, raw formula goes negative → clamp."""
        gap = rss_safe_gap(1.0, 100.0)
        assert gap == 0.0

    def test_custom_params(self):
        gap = rss_safe_gap(20.0, 10.0, t_resp=1.5, a_min_brake_ego=3.0,
                           a_max_brake_lead=8.0)
        expected = 20.0 * 1.5 + 20.0**2 / 6.0 - 10.0**2 / 16.0
        assert gap == pytest.approx(max(0.0, expected))

    def test_parameters_dataclass(self):
        p = RssParameters(t_resp=1.0, a_min_brake_ego=5.0, a_max_brake_lead=7.0)
        assert p.t_resp == 1.0
        assert p.a_min_brake_ego == 5.0
        assert p.a_max_brake_lead == 7.0


class TestRssViolation:
    def test_violation_when_too_close(self):
        safe = rss_safe_gap(25.0, 20.0)
        assert rss_violation(safe - 1.0, 25.0, 20.0) is True

    def test_no_violation_at_safe_distance(self):
        safe = rss_safe_gap(25.0, 20.0)
        assert rss_violation(safe + 1.0, 25.0, 20.0) is False

    def test_boundary_not_violation(self):
        """Exactly at safe gap is NOT a violation (gap >= safe)."""
        safe = rss_safe_gap(25.0, 20.0)
        assert rss_violation(safe, 25.0, 20.0) is False


class TestVectorised:
    def test_vec_matches_scalar(self):
        v_ego = np.array([0.0, 10.0, 20.0, 30.0])
        v_lead = np.array([5.0, 10.0, 15.0, 0.0])
        vec_gaps = rss_safe_gap_vec(v_ego, v_lead)
        for i in range(len(v_ego)):
            scalar = rss_safe_gap(v_ego[i], v_lead[i])
            assert vec_gaps[i] == pytest.approx(scalar), f"Mismatch at index {i}"

    def test_vec_violation(self):
        v_ego = np.array([20.0, 20.0])
        v_lead = np.array([15.0, 15.0])
        safe = rss_safe_gap_vec(v_ego, v_lead)
        gaps = np.array([safe[0] - 5.0, safe[1] + 5.0])
        viols = rss_violation_vec(gaps, v_ego, v_lead)
        assert viols[0] is np.True_
        assert viols[1] is np.False_

    def test_custom_params_vec(self):
        p = RssParameters(t_resp=1.0, a_min_brake_ego=3.0, a_max_brake_lead=8.0)
        v_ego = np.array([20.0])
        v_lead = np.array([10.0])
        gap = rss_safe_gap_vec(v_ego, v_lead, params=p)
        expected = 20.0 * 1.0 + 400.0 / 6.0 - 100.0 / 16.0
        assert gap[0] == pytest.approx(max(0.0, expected))
