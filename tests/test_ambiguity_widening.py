from __future__ import annotations

import numpy as np

from orius.dc3s.ambiguity import AmbiguityConfig, widen_bounds


def test_widen_bounds_expands_more_when_reliability_drops() -> None:
    lower = np.array([90.0, 95.0, 100.0], dtype=float)
    upper = np.array([110.0, 115.0, 120.0], dtype=float)
    cfg = AmbiguityConfig(lambda_mw=50.0, min_w=0.05, max_extra=1.0)

    lo_high, hi_high, meta_high = widen_bounds(lower=lower, upper=upper, w_t=0.95, cfg=cfg)
    lo_low, hi_low, meta_low = widen_bounds(lower=lower, upper=upper, w_t=0.20, cfg=cfg)

    assert np.all(lo_high <= lower)
    assert np.all(hi_high >= upper)
    assert np.all(lo_low <= lo_high)
    assert np.all(hi_low >= hi_high)
    assert meta_low["delta_mw"] > meta_high["delta_mw"]
    assert "w_t" in meta_low and "w_used" in meta_low
