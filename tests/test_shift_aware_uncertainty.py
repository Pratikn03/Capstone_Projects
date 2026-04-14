from __future__ import annotations

import numpy as np
import pandas as pd

from orius.dc3s.certificate import make_certificate
from orius.forecasting.uncertainty.conformal import build_runtime_interval
from orius.forecasting.uncertainty.shift_aware import (
    AdaptiveQuantileState,
    ShiftAwareConfig,
    SubgroupCoverageTracker,
    compute_validity_score,
    update_adaptive_quantile,
    write_shift_aware_artifacts,
)


def test_aci_update_correctness() -> None:
    st = AdaptiveQuantileState(mode="aci_basic", base_alpha=0.1, effective_alpha=0.1, learning_rate=0.05)
    update_adaptive_quantile(st, miss=True)
    assert st.effective_alpha > 0.1
    update_adaptive_quantile(st, miss=False)
    assert st.effective_alpha <= 0.15


def test_subgroup_tracking_correctness() -> None:
    tracker = SubgroupCoverageTracker(target_coverage=0.9)
    key = tracker.build_group_key(reliability_score=0.2, volatility=0.9, fault_type="dropout", ts="2026-01-01T03:00:00Z")
    stats = tracker.update(group_key=key, covered=False, interval_width=5.0, abs_residual=2.0)
    assert stats.count == 1
    assert stats.miss_count == 1
    assert stats.under_coverage_gap > 0


def test_validity_score_boundedness() -> None:
    cfg = ShiftAwareConfig(enabled=True)
    validity = compute_validity_score(
        reliability_score=0.0,
        drift_magnitude=1.0,
        normalized_residual=1.0,
        subgroup_under_coverage_gap=1.0,
        adaptation_instability=1.0,
        cfg=cfg,
    )
    assert 0.0 <= validity.validity_score <= 1.0


def test_interval_monotonicity_property() -> None:
    cfg = ShiftAwareConfig(enabled=True, policy_mode="shift_aware_linear", aci_mode="fixed")
    widths = []
    for reliability, drift, resid in [(0.9, False, 0.1), (0.7, False, 0.5), (0.5, True, 1.0), (0.2, True, 2.0)]:
        d = build_runtime_interval(
            y_hat=100.0,
            base_half_width=5.0,
            reliability_score=reliability,
            drift_flag=drift,
            residual_features={"abs_residual": resid, "covered": False, "volatility": resid},
            subgroup_context={},
            fault_context={},
            config=cfg,
        )
        widths.append(d.adjusted_half_width)
    assert all(b >= a for a, b in zip(widths, widths[1:]))


def test_legacy_mode_reproduces_no_extra_widening() -> None:
    d = build_runtime_interval(
        y_hat=10.0,
        base_half_width=2.0,
        reliability_score=0.1,
        drift_flag=True,
        config=ShiftAwareConfig(enabled=False),
    )
    assert d.adjusted_half_width == 2.0
    assert d.applied_policy == "legacy_rac_cert"


def test_certificate_new_fields_present() -> None:
    cert = make_certificate(
        command_id="cmd",
        device_id="d",
        zone_id="z",
        controller="dc3s",
        proposed_action={},
        safe_action={},
        uncertainty={},
        reliability={},
        drift={},
        model_hash="m",
        config_hash="c",
        validity_score=0.3,
        adaptive_quantile=0.2,
        conditional_coverage_gap=0.1,
        runtime_interval_policy="shift_aware_linear",
        coverage_group_key="g",
        shift_alert_flag=True,
    )
    assert cert["validity_score"] == 0.3
    assert cert["runtime_interval_policy"] == "shift_aware_linear"


def test_artifact_schema_consistency(tmp_path) -> None:
    tracker = SubgroupCoverageTracker()
    key = tracker.build_group_key(reliability_score=0.5, volatility=0.1, fault_type="none", ts="")
    tracker.update(group_key=key, covered=True, interval_width=1.0, abs_residual=0.2)
    write_shift_aware_artifacts(tracker=tracker, validity_trace=[{"t": 0, "validity_score": 1.0}], adaptive_trace=[{"t": 0, "effective_alpha": 0.1}], publication_dir=str(tmp_path))
    df = pd.read_csv(tmp_path / "reliability_group_coverage.csv")
    assert {"reliability_bin", "count", "empirical_coverage", "under_coverage_gap"}.issubset(df.columns)


def test_config_aliases_supported() -> None:
    cfg = ShiftAwareConfig.from_mapping(
        {
            "enable": True,
            "adaptation_step_size": 0.02,
            "reliability_bin_count": 7,
            "volatility_bin_count": 6,
            "validity_score_weights": {"residual": 0.4},
        }
    )
    assert cfg.enabled is True
    assert cfg.adaptation_step == 0.02
    assert cfg.reliability_bins == 7
    assert cfg.volatility_bins == 6
    assert cfg.validity_weights["residual"] == 0.4


def test_no_crash_missing_subgroup_context() -> None:
    cfg = ShiftAwareConfig(enabled=True)
    d = build_runtime_interval(
        y_hat=1.0,
        base_half_width=1.0,
        reliability_score=0.8,
        drift_flag=False,
        residual_features=None,
        subgroup_context=None,
        fault_context=None,
        config=cfg,
    )
    assert np.isfinite(d.lower)


def test_dc3s_pipeline_emits_shift_validity_fields() -> None:
    from orius.dc3s.pipeline import run_dc3s_step

    class _Adapter:
        def project_to_safe_set(self, candidate_action, uncertainty_set, state=None):
            return dict(candidate_action), {"repaired": False, "mode": "projection"}

    class _State:
        min_soc_mwh = 0.0
        max_soc_mwh = 100.0
        capacity_mwh = 100.0
        current_soc_mwh = 50.0

    out = run_dc3s_step(
        event={"ts": "2026-01-01T00:00:00Z", "fault_type": "none", "value": 1.0},
        last_event={"ts": "2025-12-31T23:00:00Z", "value": 1.0},
        yhat=10.0,
        q=2.0,
        candidate_action={"charge_mw": 1.0, "discharge_mw": 0.0},
        domain_adapter=_Adapter(),
        state=_State(),
        residual=3.0,
        cfg={
            "reliability": {"min_w": 0.05},
            "shift_aware_uncertainty": {
                "enabled": True,
                "policy_mode": "shift_aware_linear",
                "aci_mode": "aci_basic",
            },
        },
    )
    cert = out["certificate"]
    assert "validity_score" in cert
    assert "adaptive_quantile" in cert
    assert "runtime_interval_policy" in cert
