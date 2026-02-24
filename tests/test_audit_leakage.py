"""Tests for leakage audit script."""
from __future__ import annotations

import pickle

import pandas as pd

import scripts.audit_leakage as leakage


def test_leakage_audit_detects_split_overlap_and_forbidden_features(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(leakage, "REPO_ROOT", tmp_path)
    de_splits = tmp_path / "de_splits"
    de_splits.mkdir(parents=True, exist_ok=True)
    features_path = tmp_path / "de_features.parquet"
    models_dir = tmp_path / "de_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Intentional overlap: train and val share one timestamp.
    train = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="h"), "x": [1, 2, 3]})
    val = pd.DataFrame({"timestamp": pd.date_range("2026-01-01 02:00", periods=2, freq="h"), "x": [4, 5]})
    test = pd.DataFrame({"timestamp": pd.date_range("2026-01-01 04:00", periods=2, freq="h"), "x": [6, 7]})
    train.to_parquet(de_splits / "train.parquet", index=False)
    val.to_parquet(de_splits / "val.parquet", index=False)
    test.to_parquet(de_splits / "test.parquet", index=False)

    # Forbidden exact feature appears in feature store.
    pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="h"), "load_mw": [10, 11, 12], "x": [1, 2, 3]}).to_parquet(
        features_path, index=False
    )

    # Forbidden model feature also appears in artifact.
    artifact = {"feature_cols": ["x", "future_load_hint"]}
    with open(models_dir / "gbm_test.pkl", "wb") as f:
        pickle.dump(artifact, f)

    monkeypatch.setattr(
        leakage,
        "DATASETS",
        {
            "DE": {
                "splits_dir": de_splits,
                "features_path": features_path,
                "models_dir": models_dir,
            }
        },
    )

    cfg_path = tmp_path / "publish_audit.yaml"
    cfg_path.write_text(
        """
publish_audit:
  leakage_gates:
    max_overlap_train_val: 0
    max_overlap_train_test: 0
    max_overlap_val_test: 0
    forbidden_feature_exact: [load_mw]
    forbidden_feature_patterns: ["(?i)future"]
""".strip(),
        encoding="utf-8",
    )

    payload = leakage.run_leakage_audit(config_path=cfg_path)
    assert payload["fail"] is True
    types = {v["type"] for v in payload["violations"]}
    assert "overlap_train_val" in types
    assert "features_forbidden" in types
    assert "model_features_forbidden" in types
