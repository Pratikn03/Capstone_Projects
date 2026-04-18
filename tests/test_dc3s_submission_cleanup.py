from __future__ import annotations

from orius.dc3s.calibration import build_uncertainty_set, derived_inflation_factor, effective_sample_size


def test_effective_sample_size_matches_kish_formula() -> None:
    assert effective_sample_size([1.0, 1.0, 1.0, 1.0]) == 4.0
    assert effective_sample_size([1.0, 0.0, 0.0, 0.0]) == 1.0


def test_derived_inflation_is_available_but_non_default() -> None:
    result = derived_inflation_factor(0.4, [1.0] * 128, alpha=0.1, inflation_max=2.0)
    assert result.inflation_factor > 1.0


def test_build_uncertainty_set_defaults_to_heuristic_selector() -> None:
    cfg = {"k_q": 0.8, "k_drift": 0.6, "infl_max": 3.0}
    _, _, meta = build_uncertainty_set(yhat=100.0, q=10.0, w_t=0.5, drift_flag=True, cfg=cfg)
    assert meta["inflation_law_selector"] == "heuristic"


def test_build_uncertainty_set_supports_opt_in_derived_selector() -> None:
    cfg = {
        "inflation_law": "derived",
        "reliability_history": [1.0] * 128,
        "infl_max": 2.0,
        "k_q": 0.8,
        "k_drift": 0.6,
    }
    _, _, meta = build_uncertainty_set(yhat=100.0, q=10.0, w_t=0.4, drift_flag=False, cfg=cfg)
    assert meta["inflation_law_selector"] == "derived"
    assert meta["inflation"] > 1.0
