"""Regression tests for multi-domain report generation edge cases."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_reports import ReportContext, build_multi_horizon


def test_build_multi_horizon_skips_empty_test_split(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    splits.mkdir(parents=True)
    features = tmp_path / "features.parquet"

    train_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
            "power_mw": [1.0] * 8,
            "feat": list(range(8)),
        }
    )
    empty_df = train_df.iloc[0:0].copy()
    train_df.to_parquet(splits / "train.parquet", index=False)
    empty_df.to_parquet(splits / "test.parquet", index=False)
    train_df.to_parquet(features, index=False)

    ctx = ReportContext(
        repo_root=Path.cwd(),
        features_path=features,
        splits_dir=splits,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        publication_dir=tmp_path / "publication",
        targets=["power_mw"],
    )

    result = build_multi_horizon(ctx)
    assert result is not None
    assert result["status"] == "skipped"
    assert result["reason"] == "empty_split"
