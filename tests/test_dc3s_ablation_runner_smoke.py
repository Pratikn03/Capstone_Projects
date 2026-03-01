from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.run_dc3s_ablations_cpsbench import run_dc3s_ablations


def test_dc3s_ablation_runner_smoke(tmp_path: Path) -> None:
    payload = run_dc3s_ablations(
        scenario="drift_combo",
        seeds=[11],
        horizon=24,
        out_dir=tmp_path,
    )

    csv_path = tmp_path / "dc3s_ablation_table.csv"
    assert payload["csv"] == str(csv_path)
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert not df.empty
    assert {"policy", "picp_90", "mean_interval_width", "intervention_rate"}.issubset(df.columns)
    assert set(df["policy"]) == {"dc3s_no_wt", "dc3s_no_drift", "dc3s_linear", "dc3s_kappa"}
