from __future__ import annotations

from orius.forecasting.uncertainty.conformal import build_runtime_interval
from orius.forecasting.uncertainty.shift_aware import (
    ShiftAwareConfig,
    make_aci_state,
    update_adaptive_quantile,
    SubgroupCoverageTracker,
    compute_validity_score,
)


def test_aci_update_miss_increases_alpha() -> None:
    cfg = ShiftAwareConfig(enabled=True, aci_mode="aci_basic", adaptation_step=0.05)
    st = make_aci_state(cfg)
    before = st.effective_alpha
    update_adaptive_quantile(st, is_miss=True, config=cfg)
    assert st.effective_alpha > before


def test_subgroup_tracker_counts() -> None:
    cfg = ShiftAwareConfig(enabled=True)
    tracker = SubgroupCoverageTracker(config=cfg)
    tracker.update(covered=True, interval_width=2.0, abs_residual=0.2, context={"reliability": 0.9})
    tracker.update(covered=False, interval_width=2.0, abs_residual=1.2, context={"reliability": 0.9})
    rows = tracker.to_rows()
    assert rows
    assert rows[0]["count"] == 2
    assert rows[0]["miss_count"] == 1


def test_validity_bounded() -> None:
    cfg = ShiftAwareConfig(enabled=True)
    v = compute_validity_score(
        reliability_score=0.0,
        drift_magnitude=10.0,
        normalized_residual=10.0,
        under_coverage_gap=10.0,
        adaptation_instability=10.0,
        config=cfg,
    )
    assert 0.0 <= v.validity_score <= 1.0


def test_interval_monotonicity_property() -> None:
    cfg = ShiftAwareConfig(enabled=True, policy_mode="shift_aware_linear")
    seq = [
        (0.95, False, 0.1, 0.0),
        (0.8, False, 0.2, 0.05),
        (0.6, True, 0.6, 0.1),
        (0.4, True, 1.0, 0.2),
    ]
    widths = []
    for r, d, nr, gap in seq:
        dec = build_runtime_interval(
            y_hat=10.0,
            base_half_width=1.0,
            reliability_score=r,
            drift_flag=d,
            residual_features={"normalized_residual": nr, "abs_residual": nr, "y_true": 20.0},
            subgroup_context={"volatility": nr},
            config=cfg,
        )
        widths.append(dec.adjusted_half_width)
    assert all(widths[i] <= widths[i + 1] for i in range(len(widths) - 1))


def test_legacy_mode_no_crash_missing_subgroup_context() -> None:
    cfg = ShiftAwareConfig(enabled=False)
    dec = build_runtime_interval(
        y_hat=1.0,
        base_half_width=0.5,
        reliability_score=0.9,
        drift_flag=False,
        config=cfg,
    )
    assert dec.applied_policy == "legacy_rac_cert"
