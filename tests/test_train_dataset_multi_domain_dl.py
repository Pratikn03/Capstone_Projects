from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scripts.train_dataset as td
from orius.forecasting import train as forecasting_train
from orius.utils.scaler import StandardScaler


def test_multi_domain_configs_enable_full_dl_stack() -> None:
    expected = ["gbm_lightgbm", "lstm", "tcn", "nbeats", "tft", "patchtst"]

    for dataset_key in ("AV", "HEALTHCARE"):
        cfg = td._load_training_cfg(td.DATASET_REGISTRY[dataset_key])
        assert td._configured_model_types(cfg) == expected
        task_cfg = cfg["task"]
        assert int(task_cfg["lookback_steps"]) == 24
        assert len(cfg["seeds"]) >= 4
        assert cfg["models"]["dl_lstm"]["params"]["epochs"] >= 150
        assert cfg["models"]["dl_tcn"]["params"]["epochs"] >= 150
        assert cfg["models"]["dl_nbeats"]["params"]["epochs"] >= 150
        assert cfg["models"]["dl_tft"]["params"]["epochs"] >= 150
        assert cfg["models"]["dl_patchtst"]["params"]["epochs"] >= 150
        tuning_cfg = cfg["tuning"]
        assert tuning_cfg["enabled"] is False
        assert "baseline_gbm" in tuning_cfg["params"]
    healthcare_cfg = td._load_training_cfg(td.DATASET_REGISTRY["HEALTHCARE"])
    assert healthcare_cfg["task"]["targets"] == ["hr_bpm", "spo2_pct", "respiratory_rate"]


def test_sequence_lookback_resolution_prefers_steps_and_never_returns_zero() -> None:
    assert forecasting_train._resolve_task_lookback({"lookback_hours": 0.1}) == 1
    assert forecasting_train._resolve_task_lookback({"lookback_hours": 0.1, "lookback_steps": 24}) == 24


def test_multi_domain_train_splits_are_imputable_and_scalable() -> None:
    cases = (
        ("AV", "speed_mps"),
        ("HEALTHCARE", "hr_bpm"),
    )
    repo_root = Path(td.REPO_ROOT)

    for dataset_key, target in cases:
        registry_cfg = td.DATASET_REGISTRY[dataset_key]
        train_cfg = td._load_training_cfg(registry_cfg)
        targets = td._configured_targets(train_cfg)
        train_df = pd.read_parquet(repo_root / registry_cfg.splits_path / "train.parquet")
        X, y, feat_cols = forecasting_train.make_xy(train_df, target, targets)
        assert feat_cols
        assert X.dtype == np.float32
        fill_values = forecasting_train._fit_feature_fill_values(X)
        X_imputed = forecasting_train._apply_feature_fill_values(X, fill_values)
        assert np.isfinite(X_imputed).all()
        scaler = StandardScaler.fit(X_imputed)
        X_scaled = scaler.transform(X_imputed)
        assert np.isfinite(X_scaled).all()
        assert np.isfinite(y).all()
