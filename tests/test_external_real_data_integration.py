from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.download_av_datasets as av_builder
import scripts.build_navigation_real_dataset as nav_builder
from scripts._dataset_registry import DATASET_REGISTRY
from orius.data_pipeline.build_features_navigation import build_features as build_navigation_features


def test_waymo_external_fixture_builds_canonical_av_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    external_root = tmp_path / "external"
    source_dir = external_root / "waymo_open_motion" / "train"
    source_dir.mkdir(parents=True)
    raw_dir = tmp_path / "repo_raw"
    raw_dir.mkdir(parents=True)
    monkeypatch.setattr(av_builder, "RAW_DIR", raw_dir)
    pd.DataFrame(
        {
            "track_id": ["veh-1", "veh-1", "veh-1"],
            "frame_index": [0, 1, 2],
            "center_x": [10.0, 10.8, 11.7],
            "velocity_x": [8.0, 8.5, 9.0],
            "speed_limit": [13.4, 13.4, 13.4],
            "lead_distance_m": [25.0, 25.5, 26.0],
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:02Z"],
        }
    ).to_csv(source_dir / "train.csv", index=False)

    out_path = tmp_path / "av_trajectories_orius.csv"
    av_builder.build_real_av_dataset("waymo_motion", out_path, external_root=external_root)

    df = pd.read_csv(out_path)
    assert list(df.columns) == [
        "vehicle_id",
        "step",
        "position_m",
        "speed_mps",
        "speed_limit_mps",
        "lead_position_m",
        "ts_utc",
        "source_split",
    ]
    assert df["vehicle_id"].tolist() == ["veh-1", "veh-1", "veh-1"]
    assert df["source_split"].tolist() == ["train", "train", "train"]
    assert round(float(df["lead_position_m"].iloc[0]), 2) == 35.00

    summary = json.loads((raw_dir / "waymo_motion_build_summary.json").read_text())
    assert summary["source"] == "waymo_motion"
    assert summary["rows"] == 3


def test_navigation_builder_and_feature_pipeline_from_kitti_fixture(tmp_path: Path) -> None:
    external_root = tmp_path / "external"
    poses_dir = external_root / "kitti_odometry" / "dataset" / "poses"
    seq_dir = external_root / "kitti_odometry" / "dataset" / "sequences" / "00"
    poses_dir.mkdir(parents=True)
    seq_dir.mkdir(parents=True)
    (seq_dir / "times.txt").write_text("0.0\n0.1\n0.2\n0.3\n", encoding="utf-8")
    (poses_dir / "00.txt").write_text(
        "\n".join(
            [
                "1 0 0 0.0 0 1 0 0 0 0 1 0.0",
                "1 0 0 0.1 0 1 0 0 0 0 1 0.0",
                "1 0 0 0.2 0 1 0 0 0 0 1 0.1",
                "1 0 0 0.4 0 1 0 0 0 0 1 0.2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_csv = tmp_path / "navigation_orius.csv"
    manifest_out = tmp_path / "navigation_manifest.json"
    nav_builder.build_navigation_dataset(
        out_csv,
        external_root=external_root,
        sequences=["00"],
        manifest_out=manifest_out,
    )

    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["robot_id", "step", "x", "y", "vx", "vy", "ts_utc", "source_sequence"]
    assert df["robot_id"].nunique() == 1
    assert df["source_sequence"].astype(str).str.zfill(2).tolist() == ["00", "00", "00", "00"]

    feature_dir = tmp_path / "navigation_features"
    features_path = build_navigation_features(out_csv, feature_dir)
    assert features_path.exists()
    assert (feature_dir / "splits" / "train.parquet").exists()
    assert (feature_dir / "splits" / "test.parquet").exists()
    manifest = json.loads(manifest_out.read_text())
    assert manifest["dataset"] == "KITTI Odometry"
    assert manifest["rows_per_sequence"]["00"] == 4


def test_dataset_registry_includes_navigation_row() -> None:
    cfg = DATASET_REGISTRY["NAVIGATION"]
    assert cfg.config_file == "configs/train_forecast_navigation.yaml"
    assert cfg.raw_data_path == "data/navigation/processed/navigation_orius.csv"
    assert cfg.feature_module == "orius.data_pipeline.build_features_navigation"
