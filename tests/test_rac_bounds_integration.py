from __future__ import annotations

from pathlib import Path

import pandas as pd

from orius.cpsbench_iot.runner import run_suite


def test_rac_bounds_columns_present_for_all_controllers(tmp_path: Path) -> None:
    out_dir = tmp_path / "rac_suite"
    run_suite(scenarios=["nominal"], seeds=[11], out_dir=out_dir, horizon=24)
    df = pd.read_csv(out_dir / "dc3s_main_table.csv")
    assert set(df["controller"].astype(str).unique()) == {
        "deterministic_lp",
        "robust_fixed_interval",
        "cvar_interval",
        "dc3s_wrapped",
        "dc3s_ftit",
        "aci_conformal",
        "scenario_robust",
        "scenario_mpc",
    }
    required = {
        "rac_sensitivity_mean",
        "rac_sensitivity_p95",
        "rac_q_multiplier_mean",
        "rac_q_multiplier_p95",
        "rac_inflation_mean",
    }
    assert required.issubset(set(df.columns))
