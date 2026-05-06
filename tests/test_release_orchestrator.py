"""End-to-end test for the unified release orchestrator on synthetic data.

Skips legacy training (subprocess-heavy) and the publication-table/significance
subprocess steps — those are exercised by their own tests. The focus here is
that splits-carving, the in-process advanced trainer, prediction extraction,
and manifest writing all wire together correctly and produce a single
release_manifest.json with the expected fields.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from orius.release.manifest import ReleaseManifest, collect_environment, write_release_manifest
from orius.release.splits import carve_splits, splits_config_from_yaml


def _make_synthetic_features(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    hour = ts.hour.to_numpy(dtype=float)
    dow = ts.dayofweek.to_numpy(dtype=float)
    temperature = 10 + 8 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.normal(0, 1, n)
    base = 1000 + 200 * np.sin(2 * np.pi * hour / 24) + 80 * np.sin(2 * np.pi * dow / 7)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "hour": hour,
            "day_of_week": dow,
            "temperature": temperature,
            "load_mw": base + 6 * temperature + rng.normal(0, 25, n),
            "wind_mw": 400 + 120 * np.cos(2 * np.pi * np.arange(n) / 168) + rng.normal(0, 30, n),
            "solar_mw": np.maximum(0.0, 600 * np.sin(2 * np.pi * (hour - 6) / 24)) + rng.normal(0, 20, n),
        }
    )


@pytest.fixture()
def synthetic_features(tmp_path: Path) -> tuple[Path, Path]:
    features = _make_synthetic_features(n=24 * 30, seed=42)
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path)
    config = {
        "data": {"timestamp_col": "timestamp"},
        "task": {"horizon_hours": 24, "lookback_hours": 72, "targets": ["load_mw"]},
        "splits": {"train_ratio": 0.70, "val_ratio": 0.10, "calibration_ratio": 0.05, "gap_hours": 0},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return features_path, config_path


def test_carve_splits_is_deterministic(synthetic_features: tuple[Path, Path], tmp_path: Path) -> None:
    features_path, config_path = synthetic_features
    cfg = splits_config_from_yaml(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    out_a = tmp_path / "splits_a"
    out_b = tmp_path / "splits_b"
    carved_a = carve_splits(features_path=features_path, out_dir=out_a, cfg=cfg)
    carved_b = carve_splits(features_path=features_path, out_dir=out_b, cfg=cfg)
    assert carved_a.splits_sha256 == carved_b.splits_sha256
    assert carved_a.features_sha256 == carved_b.features_sha256
    assert carved_a.boundaries == carved_b.boundaries
    train_a = pd.read_parquet(carved_a.train_path)
    train_b = pd.read_parquet(carved_b.train_path)
    pd.testing.assert_frame_equal(train_a.reset_index(drop=True), train_b.reset_index(drop=True))


def test_carve_splits_records_boundaries_match_legacy_indexing(
    synthetic_features: tuple[Path, Path], tmp_path: Path
) -> None:
    features_path, config_path = synthetic_features
    cfg = splits_config_from_yaml(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    carved = carve_splits(features_path=features_path, out_dir=tmp_path / "splits", cfg=cfg)
    n = carved.boundaries["n_rows"]
    expected_train_end = int(n * 0.70)
    expected_cal_start = expected_train_end
    expected_cal_end = expected_cal_start + int(n * 0.05)
    expected_val_start = expected_cal_end
    expected_val_end = expected_val_start + int(n * 0.10)
    assert carved.boundaries["train_end"] == expected_train_end
    assert carved.boundaries["calibration_start"] == expected_cal_start
    assert carved.boundaries["calibration_end"] == expected_cal_end
    assert carved.boundaries["val_start"] == expected_val_start
    assert carved.boundaries["val_end"] == expected_val_end


def test_sha256_changes_when_features_change(synthetic_features: tuple[Path, Path], tmp_path: Path) -> None:
    features_path, config_path = synthetic_features
    cfg = splits_config_from_yaml(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    carved_a = carve_splits(features_path=features_path, out_dir=tmp_path / "a", cfg=cfg)

    df = pd.read_parquet(features_path)
    df.loc[0, "load_mw"] = float(df.loc[0, "load_mw"]) + 1.0
    df.to_parquet(features_path)

    carved_b = carve_splits(features_path=features_path, out_dir=tmp_path / "b", cfg=cfg)
    assert carved_a.features_sha256 != carved_b.features_sha256
    assert carved_a.splits_sha256 != carved_b.splits_sha256


def test_release_manifest_round_trip(tmp_path: Path) -> None:
    manifest = ReleaseManifest(
        region="DE",
        release_id="TEST",
        started_at="2026-05-02T00:00:00+00:00",
        environment=collect_environment(),
        inputs={"splits_sha256": "abc"},
    )
    out = tmp_path / "release_manifest.json"
    write_release_manifest(manifest, out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["region"] == "DE"
    assert payload["release_id"] == "TEST"
    assert payload["inputs"]["splits_sha256"] == "abc"
    assert payload["environment"]["python_version"]
    assert "dep_versions" in payload["environment"]


def test_orchestrator_smoke_end_to_end(synthetic_features: tuple[Path, Path], tmp_path: Path) -> None:
    if importlib.util.find_spec("ngboost") is None:
        pytest.skip("ngboost not installed in this environment")
    features_path, config_path = synthetic_features

    spec = importlib.util.spec_from_file_location(
        "run_release", Path(__file__).resolve().parent.parent / "scripts" / "run_release.py"
    )
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    manifest = module.run_release(
        region="DE",
        release_id="SMOKE_E2E",
        features_path=features_path,
        config_path=config_path,
        out_root=tmp_path,
        targets=("load_mw",),
        seeds=(42,),
        advanced_models=("ngboost",),
        significance_baselines=("ngboost",),
        holiday_country=None,
        legacy_seed=42,
        skip_legacy=True,
        skip_advanced=False,
        skip_table=True,
        skip_significance=True,
        n_resamples=200,
        flaml_time_budget=15,
        legacy_extra_args=[],
    )

    assert manifest.summary["n_failed_steps"] == 0
    assert manifest.inputs["splits_sha256"]
    assert manifest.inputs["features_sha256"]
    assert manifest.inputs["splits_boundaries"]["n_rows"] > 0

    release_root = Path(manifest.artifacts["release_root"])
    assert (release_root / "release_manifest.json").exists()
    assert (release_root / "splits" / "splits_manifest.json").exists()
    assert (release_root / "predictions").exists()
    advanced_runs = release_root / "advanced_baselines"
    npz_files = list(advanced_runs.glob("ngboost_load_mw_seed*.npz"))
    assert len(npz_files) == 1
    staged = list((release_root / "predictions").glob("ngboost_load_mw_seed*.npz"))
    assert len(staged) == 1

    payload = json.loads((release_root / "release_manifest.json").read_text(encoding="utf-8"))
    step_names = [s["name"] for s in payload["steps"]]
    assert "preflight" in step_names
    assert "carve_splits" in step_names
    assert "advanced_trainer" in step_names
    assert "extract_predictions" in step_names
    assert payload["environment"]["dep_versions"]["ngboost"] is not None
