"""Tests for the nuPlan-to-ORIUS AV replay bridge."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pandas as pd

import orius.av_waymo.training as av_training
from orius.av_nuplan import build_nuplan_replay_surface, resolve_nuplan_train_archives
from orius.av_waymo import build_feature_tables


def _token(prefix: str, index: int) -> bytes:
    return f"{prefix}{index:012d}".encode("ascii")[:16]


def _write_synthetic_nuplan_db(path: Path, *, steps: int = 91, location: str = "sg-one-north") -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE camera (token BLOB);
        CREATE TABLE category (token BLOB, name TEXT, description TEXT);
        CREATE TABLE ego_pose (
            token BLOB, timestamp INTEGER, x REAL, y REAL, z REAL,
            qw REAL, qx REAL, qy REAL, qz REAL,
            vx REAL, vy REAL, vz REAL,
            acceleration_x REAL, acceleration_y REAL, acceleration_z REAL,
            angular_rate_x REAL, angular_rate_y REAL, angular_rate_z REAL,
            epsg INTEGER, log_token BLOB
        );
        CREATE TABLE image (token BLOB);
        CREATE TABLE lidar (token BLOB);
        CREATE TABLE lidar_box (
            token BLOB, lidar_pc_token BLOB, track_token BLOB,
            next_token BLOB, prev_token BLOB,
            x REAL, y REAL, z REAL, width REAL, length REAL, height REAL,
            vx REAL, vy REAL, vz REAL, yaw REAL, confidence REAL
        );
        CREATE TABLE lidar_pc (
            token BLOB, next_token BLOB, prev_token BLOB,
            ego_pose_token BLOB, lidar_token BLOB, scene_token BLOB,
            filename TEXT, timestamp INTEGER
        );
        CREATE TABLE log (
            token BLOB, vehicle_name TEXT, date TEXT, timestamp INTEGER,
            logfile TEXT, location TEXT, map_version TEXT
        );
        CREATE TABLE scenario_tag (token BLOB);
        CREATE TABLE scene (
            token BLOB, log_token BLOB, name TEXT, goal_ego_pose_token BLOB,
            roadblock_ids TEXT
        );
        CREATE TABLE track (token BLOB, category_token BLOB, width REAL, length REAL, height REAL);
        CREATE TABLE traffic_light_status (token BLOB);
        """
    )
    log_token = b"log-token-000001"
    scene_token = b"scene-token-0001"
    lidar_token = b"lidar-token-0001"
    category_token = b"category-vehicle"
    track_token = b"track-lead-00001"
    con.execute(
        "INSERT INTO log VALUES (?, ?, ?, ?, ?, ?, ?)",
        (log_token, "veh-test", "2021-09-29", 1_632_878_000_000_000, "synthetic", location, location),
    )
    con.execute("INSERT INTO scene VALUES (?, ?, ?, ?, ?)", (scene_token, log_token, "scene-0001", None, ""))
    con.execute("INSERT INTO lidar VALUES (?)", (lidar_token,))
    con.execute("INSERT INTO category VALUES (?, ?, ?)", (category_token, "vehicle", "synthetic vehicle"))
    con.execute("INSERT INTO track VALUES (?, ?, ?, ?, ?)", (track_token, category_token, 2.0, 4.8, 1.8))
    base_ts = 1_632_878_371_000_000
    for index in range(steps):
        ego_token = _token("ego", index)
        lidar_pc_token = _token("pc", index)
        box_token = _token("box", index)
        timestamp = base_ts + index * 100_000
        ego_x = float(index)
        con.execute(
            "INSERT INTO ego_pose VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ego_token,
                timestamp,
                ego_x,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                10.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                32648,
                log_token,
            ),
        )
        con.execute(
            "INSERT INTO lidar_pc VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                lidar_pc_token,
                _token("pc", index + 1) if index < steps - 1 else None,
                _token("pc", index - 1) if index > 0 else None,
                ego_token,
                lidar_token,
                scene_token,
                f"synthetic/MergedPointCloud/{index:04d}.pcd",
                timestamp,
            ),
        )
        con.execute(
            "INSERT INTO lidar_box VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                box_token,
                lidar_pc_token,
                track_token,
                None,
                None,
                ego_x + 30.0,
                0.2,
                0.0,
                2.0,
                4.8,
                1.8,
                8.0,
                0.0,
                0.0,
                0.0,
                0.95,
            ),
        )
    con.commit()
    con.close()


def test_build_nuplan_replay_surface_and_features(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_nuplan.db"
    _write_synthetic_nuplan_db(db_path)

    train_zip = tmp_path / "nuplan-v1.1_train_singapore.zip"
    maps_zip = tmp_path / "nuplan-maps-v1.0.zip"
    with zipfile.ZipFile(train_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(db_path, "data/cache/train_singapore/synthetic_nuplan.db")
    with zipfile.ZipFile(maps_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("maps/nuplan-maps-v1.0.json", "{}")
        archive.writestr("maps/sg-one-north/9.17.1964/map.gpkg", b"synthetic")

    out_dir = tmp_path / "processed_nuplan"
    report = build_nuplan_replay_surface(
        train_zip=train_zip,
        maps_zip=maps_zip,
        out_dir=out_dir,
        max_dbs=1,
        max_scenarios=1,
    )

    assert report["source_dataset"] == "nuplan_singapore"
    assert report["source_datasets"] == ["nuplan_singapore"]
    assert report["row_count"] == 91
    assert report["scenario_count"] == 1
    replay = pd.read_parquet(out_dir / "replay_windows.parquet")
    assert not (out_dir / "replay_windows.parquet.tmp").exists()
    assert set(replay["source_dataset"].unique()) == {"nuplan_singapore"}
    assert replay["scenario_id"].nunique() == 1
    assert replay["neighbor_count"].max() == 1
    assert replay["lead_track_id"].notna().all()
    assert replay["min_gap_m"].min() > 20.0

    feature_report = build_feature_tables(
        replay_windows_path=out_dir / "replay_windows.parquet",
        out_dir=out_dir,
    )
    assert feature_report["row_count"] > 0
    assert (out_dir / "step_features.parquet").exists()


def test_build_nuplan_replay_surface_from_multiple_zips_skips_incomplete(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "synthetic_nuplan.db"
    _write_synthetic_nuplan_db(db_path)

    maps_zip = tmp_path / "nuplan-maps-v1.0.zip"
    with zipfile.ZipFile(maps_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("maps/nuplan-maps-v1.0.json", "{}")

    for archive_index in range(4):
        city = "pittsburgh" if archive_index == 3 else "singapore"
        train_zip = tmp_path / f"nuplan-v1.1_train_{city}_part{archive_index}.zip"
        with zipfile.ZipFile(train_zip, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(db_path, f"data/cache/train_{city}/same_member_name.db")
    (tmp_path / "nuplan-v1.1_train_extra.zip.crdownload").write_bytes(b"incomplete")

    archives, skipped = resolve_nuplan_train_archives(
        train_dirs=[tmp_path],
        train_glob="*.zip*",
        skip_incomplete=True,
    )
    assert len(archives) == 4
    assert {row["reason"] for row in skipped} >= {"incomplete_download", "maps_archive"}

    out_dir = tmp_path / "processed_multi_nuplan"
    report = build_nuplan_replay_surface(
        train_dirs=[tmp_path],
        train_glob="*.zip*",
        maps_zip=maps_zip,
        out_dir=out_dir,
        max_dbs=4,
        max_scenarios=4,
    )

    assert report["scenario_count"] == 4
    assert report["source_dataset"] == "nuplan_multi_city"
    assert set(report["source_datasets"]) == {"nuplan_singapore", "nuplan_pittsburgh"}
    assert len(report["inputs"]["train_archives"]) == 4
    assert (out_dir / "nuplan_source_manifest.json").exists()
    replay = pd.read_parquet(out_dir / "replay_windows.parquet")
    assert set(replay["source_dataset"].unique()) == {"nuplan_singapore", "nuplan_pittsburgh"}
    assert replay["scenario_id"].nunique() == 4
    assert replay["shard_id"].nunique() == 4

    monkeypatch.setattr(av_training, "IN_MEMORY_FEATURE_ROW_LIMIT", 0)
    feature_report = av_training.build_feature_tables(
        replay_windows_path=out_dir / "replay_windows.parquet",
        out_dir=out_dir,
        split_strategy="balanced",
    )
    assert feature_report["streaming"] is True
    assert feature_report["split_counts"] == {"calibration": 1, "test": 1, "train": 1, "val": 1}


def test_grouped_archive_db_city_split_has_no_db_leakage(tmp_path: Path, monkeypatch) -> None:
    maps_zip = tmp_path / "nuplan-maps-v1.0.zip"
    with zipfile.ZipFile(maps_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("maps/nuplan-maps-v1.0.json", "{}")

    city_locations = {
        "singapore": "sg-one-north",
        "boston": "us-ma-boston",
    }
    for city, location in city_locations.items():
        for index in range(10):
            db_path = tmp_path / f"synthetic_{city}_{index}.db"
            _write_synthetic_nuplan_db(db_path, location=location)
            train_zip = tmp_path / f"nuplan-v1.1_train_{city}_part{index}.zip"
            with zipfile.ZipFile(train_zip, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.write(db_path, f"data/cache/train_{city}/log_{index}.db")

    out_dir = tmp_path / "processed_grouped_nuplan"
    report = build_nuplan_replay_surface(
        train_dirs=[tmp_path],
        train_glob="nuplan-v*.zip",
        maps_zip=maps_zip,
        out_dir=out_dir,
        max_dbs_per_archive=1,
        max_scenarios_per_archive=1,
    )
    assert report["scenario_count"] == 20

    monkeypatch.setattr(av_training, "IN_MEMORY_FEATURE_ROW_LIMIT", 0)
    feature_report = av_training.build_feature_tables(
        replay_windows_path=out_dir / "replay_windows.parquet",
        out_dir=out_dir,
        split_strategy="grouped_archive_db_city",
    )

    assert feature_report["split_strategy"] == "grouped_archive_db_city"
    assert feature_report["split_group_columns"] == ["db_entry", "source_archive_id"]
    assert feature_report["split_group_count"] == 20
    assert feature_report["split_counts"] == {"calibration": 2, "test": 2, "train": 14, "val": 2}
    assert feature_report["split_city_group_counts"] == {
        "sg-one-north": {"calibration": 1, "test": 1, "train": 7, "val": 1},
        "us-ma-boston": {"calibration": 1, "test": 1, "train": 7, "val": 1},
    }

    scenario_index = pd.read_parquet(
        out_dir / "scenario_index.parquet",
        columns=["scenario_id", "source_archive_id", "db_entry"],
    )
    anchors = pd.read_parquet(out_dir / "anchor_features.parquet", columns=["scenario_id", "split"])
    joined = scenario_index.merge(anchors, on="scenario_id", how="inner")
    assert joined.groupby(["source_archive_id", "db_entry"])["split"].nunique().max() == 1


def test_build_nuplan_replay_surface_can_bound_each_archive(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_nuplan.db"
    _write_synthetic_nuplan_db(db_path)

    maps_zip = tmp_path / "nuplan-maps-v1.0.zip"
    with zipfile.ZipFile(maps_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("maps/nuplan-maps-v1.0.json", "{}")

    for city in ("boston", "pittsburgh", "singapore"):
        train_zip = tmp_path / f"nuplan-v1.1_train_{city}.zip"
        with zipfile.ZipFile(train_zip, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(db_path, f"data/cache/train_{city}/same_member_name.db")

    out_dir = tmp_path / "processed_per_archive"
    report = build_nuplan_replay_surface(
        train_dirs=[tmp_path],
        train_glob="nuplan-v*.zip",
        maps_zip=maps_zip,
        out_dir=out_dir,
        max_dbs_per_archive=1,
        max_scenarios_per_archive=1,
    )

    assert report["scenario_count"] == 3
    assert report["db_count"] == 3
    assert report["source_dataset"] == "nuplan_multi_city"
    assert set(report["source_datasets"]) == {"nuplan_boston", "nuplan_pittsburgh", "nuplan_singapore"}
    assert report["bounds"]["max_dbs_per_archive"] == 1
    assert report["bounds"]["max_scenarios_per_archive"] == 1
    replay = pd.read_parquet(out_dir / "replay_windows.parquet")
    assert replay["scenario_id"].nunique() == 3
    assert set(replay["source_dataset"].unique()) == {
        "nuplan_boston",
        "nuplan_pittsburgh",
        "nuplan_singapore",
    }


def test_feature_tables_can_mark_holdout_surface_as_all_test(tmp_path: Path) -> None:
    db_path = tmp_path / "synthetic_nuplan.db"
    _write_synthetic_nuplan_db(db_path)

    train_zip = tmp_path / "nuplan-v1.0_val.zip"
    maps_zip = tmp_path / "nuplan-maps-v1.0.zip"
    with zipfile.ZipFile(train_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(db_path, "data/cache/public_set_val/synthetic_nuplan.db")
    with zipfile.ZipFile(maps_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("maps/nuplan-maps-v1.0.json", "{}")

    out_dir = tmp_path / "processed_val_holdout"
    report = build_nuplan_replay_surface(
        train_zip=train_zip,
        maps_zip=maps_zip,
        out_dir=out_dir,
        max_dbs=1,
        max_scenarios=1,
    )
    assert report["scenario_count"] == 1

    feature_report = build_feature_tables(
        replay_windows_path=out_dir / "replay_windows.parquet",
        out_dir=out_dir,
        split_strategy="all_test",
    )
    assert feature_report["split_strategy"] == "all_test"
    assert feature_report["split_counts"] == {"test": 1}
