"""Smoke test for CPSBench-IoT runner outputs and determinism."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from orius.cpsbench_iot.runner import REQUIRED_OUTPUTS, run_suite


def _assert_outputs(out_dir: Path) -> None:
    for name in REQUIRED_OUTPUTS:
        path = out_dir / name
        assert path.exists(), f"Missing artifact: {path}"
        assert path.stat().st_size > 0, f"Empty artifact: {path}"


def test_cpsbench_runner_smoke_and_determinism(tmp_path):
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    run_suite(scenarios=["nominal"], seeds=[11], out_dir=out_a, horizon=24)
    run_suite(scenarios=["nominal"], seeds=[11], out_dir=out_b, horizon=24)

    _assert_outputs(out_a)
    _assert_outputs(out_b)

    main_a = pd.read_csv(out_a / "dc3s_main_table.csv").sort_values(["scenario", "seed", "controller"]).reset_index(drop=True)
    main_b = pd.read_csv(out_b / "dc3s_main_table.csv").sort_values(["scenario", "seed", "controller"]).reset_index(drop=True)
    required_cols = {
        "true_soc_violation_rate",
        "true_soc_violation_severity_mean",
        "true_soc_violation_severity_p95",
        "unsafe_command_rate",
        "bms_trip_rate",
        "intervention_rate",
        "adaptive_width_mean",
        "adaptive_width_p95",
        "cvar_eta",
        "cvar_cost",
        "cqr_picp_group_low",
        "cqr_picp_group_mid",
        "cqr_picp_group_high",
        "cqr_width_group_low",
        "cqr_width_group_mid",
        "cqr_width_group_high",
        "rac_sensitivity_mean",
        "rac_sensitivity_p95",
        "rac_q_multiplier_mean",
        "rac_q_multiplier_p95",
        "rac_inflation_mean",
    }
    assert required_cols <= set(main_a.columns)
    assert "dc3s_ftit" in set(main_a["controller"].astype(str))
    pd.testing.assert_frame_equal(main_a, main_b, rtol=1e-8, atol=1e-8)
