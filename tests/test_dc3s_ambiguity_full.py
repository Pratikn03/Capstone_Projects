"""Comprehensive tests for DC3S ambiguity widening."""
from __future__ import annotations

import numpy as np
import pytest

from orius.dc3s.ambiguity import AmbiguityConfig, widen_bounds


class TestWidenBounds:
    def test_zero_lambda_no_widening(self):
        cfg = AmbiguityConfig(lambda_mw=0.0)
        lo, hi, meta = widen_bounds(np.array([90.0]), np.array([110.0]), 0.5, cfg)
        assert np.isclose(lo[0], 90.0)
        assert np.isclose(hi[0], 110.0)
        assert meta["delta_mw"] == 0.0

    def test_positive_lambda_widens(self):
        cfg = AmbiguityConfig(lambda_mw=10.0, min_w=0.05, max_extra=1.0)
        lo, hi, meta = widen_bounds(np.array([90.0]), np.array([110.0]), 0.5, cfg)
        assert lo[0] < 90.0
        assert hi[0] > 110.0
        assert meta["delta_mw"] > 0.0

    def test_perfect_telemetry_zero_extra(self):
        cfg = AmbiguityConfig(lambda_mw=10.0, min_w=0.05, max_extra=1.0)
        lo, hi, meta = widen_bounds(np.array([90.0]), np.array([110.0]), 1.0, cfg)
        assert np.isclose(lo[0], 90.0)
        assert np.isclose(hi[0], 110.0)

    def test_worst_case_capped_by_max_extra(self):
        cfg = AmbiguityConfig(lambda_mw=100.0, min_w=0.05, max_extra=0.5)
        _, _, meta = widen_bounds(np.array([90.0]), np.array([110.0]), 0.0, cfg)
        assert meta["delta_mw"] == pytest.approx(100.0 * 0.5)

    def test_array_inputs(self):
        cfg = AmbiguityConfig(lambda_mw=5.0, min_w=0.05)
        lo = np.array([80.0, 90.0, 100.0])
        hi = np.array([120.0, 130.0, 140.0])
        lo2, hi2, _ = widen_bounds(lo, hi, 0.5, cfg)
        assert lo2.shape == (3,)
        assert hi2.shape == (3,)
        assert np.all(lo2 < lo)
        assert np.all(hi2 > hi)

    def test_lower_gt_upper_raises(self):
        cfg = AmbiguityConfig(lambda_mw=5.0)
        with pytest.raises(ValueError, match="lower cannot exceed upper"):
            widen_bounds(np.array([110.0]), np.array([90.0]), 0.5, cfg)

    def test_mismatched_lengths_raises(self):
        cfg = AmbiguityConfig(lambda_mw=5.0)
        with pytest.raises(ValueError, match="same length"):
            widen_bounds(np.array([90.0, 100.0]), np.array([110.0]), 0.5, cfg)

    def test_min_w_floor(self):
        cfg = AmbiguityConfig(lambda_mw=10.0, min_w=0.2)
        _, _, meta = widen_bounds(np.array([90.0]), np.array([110.0]), 0.1, cfg)
        assert meta["w_used"] == 0.2
