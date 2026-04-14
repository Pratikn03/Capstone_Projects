from __future__ import annotations

from pathlib import Path

from orius.dc3s.certificate import make_certificate
from orius.forecasting.uncertainty.shift_aware import write_shift_aware_artifacts


def test_certificate_includes_shift_aware_fields() -> None:
    cert = make_certificate(
        command_id="c1",
        device_id="d",
        zone_id="z",
        controller="dc3s",
        proposed_action={"p": 1},
        safe_action={"p": 1},
        uncertainty={},
        reliability={},
        drift={},
        model_hash="m",
        config_hash="c",
        validity_score=0.4,
        adaptive_quantile=1.2,
        conditional_coverage_gap=0.1,
        runtime_interval_policy="shift_aware_linear",
        coverage_group_key="r:b1",
        shift_alert_flag=True,
    )
    assert cert["validity_score"] == 0.4
    assert cert["runtime_interval_policy"] == "shift_aware_linear"


def test_artifact_schema_consistent(tmp_path: Path) -> None:
    out = write_shift_aware_artifacts(
        reliability_rows=[{
            "group_key": "g",
            "count": 1,
            "covered_count": 1,
            "miss_count": 0,
            "empirical_coverage": 1.0,
            "target_coverage": 0.9,
            "under_coverage_gap": 0.0,
            "avg_interval_width": 2.0,
            "avg_abs_residual": 0.1,
        }],
        volatility_rows=[],
        fault_rows=[],
        validity_trace_rows=[],
        quantile_trace_rows=[],
        out_dir=tmp_path,
    )
    rel = Path(out["reliability_group_coverage"])
    assert rel.exists()
    header = rel.read_text(encoding="utf-8").splitlines()[0]
    assert header == "group_key,count,covered_count,miss_count,empirical_coverage,target_coverage,under_coverage_gap,avg_interval_width,avg_abs_residual"
