from __future__ import annotations

import pandas as pd

from orius.forecasting.uncertainty.shift_aware import write_comparison_package
from orius.monitoring.residual_validity import ResidualValidityMonitor


def test_comparison_package_outputs(tmp_path) -> None:
    legacy = pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0],
            "lower": [0.5, 1.5, 2.5],
            "upper": [1.5, 2.5, 3.5],
        }
    )
    shift = pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0],
            "lower": [0.0, 1.0, 2.0],
            "upper": [2.0, 3.0, 4.0],
        }
    )
    legacy_csv = tmp_path / "legacy.csv"
    shift_csv = tmp_path / "shift.csv"
    legacy.to_csv(legacy_csv, index=False)
    shift.to_csv(shift_csv, index=False)

    signoff = write_comparison_package(
        legacy_csv=str(legacy_csv),
        shift_csv=str(shift_csv),
        out_dir=str(tmp_path / "out"),
        target_coverage=0.5,
        max_width_increase=5.0,
    )
    assert (tmp_path / "out" / "legacy_vs_shift_summary.csv").exists()
    assert (tmp_path / "out" / "acceptance_signoff.json").exists()
    assert signoff["all_checks_pass"] is True


def test_residual_monitor_root_cause_and_sustained_state() -> None:
    monitor = ResidualValidityMonitor(sustained_steps=2)
    out1 = monitor.update(
        abs_residual=5.0, covered=False, reliability_score=0.1, telemetry_degraded=True, subgroup_gap=0.2
    )
    out2 = monitor.update(
        abs_residual=5.0, covered=False, reliability_score=0.1, telemetry_degraded=True, subgroup_gap=0.2
    )
    assert out1["state"] in {"watch", "nominal"}
    assert out2["state"] in {"watch", "degraded", "invalid"}
    assert out2["root_cause"] == "telemetry_and_uq"
