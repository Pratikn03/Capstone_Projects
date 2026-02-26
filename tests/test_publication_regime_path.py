from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_publication_artifact import _build_cqr_group_coverage


def test_publication_module_no_conformal_runtime_dependency() -> None:
    source = Path("scripts/build_publication_artifact.py").read_text(encoding="utf-8")
    assert "load_conformal" not in source


def test_build_cqr_group_coverage_uses_existing_regime_outputs(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"target": "load_mw", "group": "low", "picp_90": 0.91, "mean_width": 100.0, "sample_count": 10},
            {"target": "load_mw", "group": "mid", "picp_90": 0.89, "mean_width": 120.0, "sample_count": 10},
            {"target": "load_mw", "group": "high", "picp_90": 0.90, "mean_width": 140.0, "sample_count": 10},
        ]
    )
    (tmp_path / "cqr_group_coverage.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    payload = _build_cqr_group_coverage(tmp_path)
    assert payload["rows"] == 3
    assert (tmp_path / "table3_group_coverage.csv").exists()
    assert (tmp_path / "fig_cqr_group_coverage.png").exists()
    out_df = pd.read_csv(tmp_path / "cqr_group_coverage.csv")
    assert set(out_df["group"].astype(str).tolist()) == {"low", "med", "high"}
