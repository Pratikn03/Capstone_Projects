from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.download_av_datasets as av_builder
import scripts.download_aerospace_datasets as aerospace_builder
import scripts.build_aerospace_public_adsb_runtime as aerospace_public_builder
import scripts.build_aerospace_real_flight_dataset as aerospace_real_builder
import scripts.build_navigation_real_dataset as nav_builder
import scripts.verify_real_data_preflight as real_data_preflight
from scripts._dataset_registry import DATASET_REGISTRY
from orius.data_pipeline.build_features_navigation import build_features as build_navigation_features
from orius.data_pipeline import real_data_contract
from orius.orius_bench import real_data_loader


def test_waymo_repo_local_fixture_preferred_over_external(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_raw = tmp_path / "repo_raw"
    repo_waymo = repo_raw / "waymo_open_motion" / "train"
    repo_waymo.mkdir(parents=True)
    external_root = tmp_path / "external"
    external_waymo = external_root / "waymo_open_motion" / "train"
    external_waymo.mkdir(parents=True)

    monkeypatch.setattr(av_builder, "RAW_DIR", repo_raw)
    pd.DataFrame(
        {
            "track_id": ["repo-veh", "repo-veh"],
            "frame_index": [0, 1],
            "center_x": [1.0, 2.0],
            "velocity_x": [5.0, 5.5],
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"],
        }
    ).to_csv(repo_waymo / "repo.csv", index=False)
    pd.DataFrame(
        {
            "track_id": ["external-veh", "external-veh"],
            "frame_index": [0, 1],
            "center_x": [100.0, 101.0],
            "velocity_x": [9.0, 9.5],
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"],
        }
    ).to_csv(external_waymo / "external.csv", index=False)

    out_path = tmp_path / "av_repo_local.csv"
    av_builder.build_real_av_dataset("waymo_motion", out_path, external_root=external_root)

    df = pd.read_csv(out_path)
    assert df["vehicle_id"].tolist() == ["repo-veh", "repo-veh"]
    manifest = json.loads((repo_raw / "waymo_open_motion_provenance.json").read_text())
    assert manifest["source_kind"] == "repo_local"


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
    manifest = json.loads((raw_dir / "waymo_open_motion_provenance.json").read_text())
    assert manifest["source_kind"] == "external"
    assert manifest["canonical_source"] is True


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
    assert manifest["canonical_source"] is True
    assert manifest["source_kind"] == "external"
    assert manifest["used_fallback"] is True


def test_aerospace_cmapss_fixture_builds_real_processed_surface(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    sample = "\n".join(
        [
            "1 1 0.1 0.2 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 100.00 39.06 23.4190",
            "1 2 0.2 0.3 100.0 518.67 642.15 1591.82 1403.14 14.62 21.61 553.75 2388.04 9044.07 1.30 47.49 522.28 2388.07 8131.49 8.4318 0.03 392 2388 100.00 39.00 23.4236",
        ]
    ) + "\n"
    for filename in aerospace_builder.CMAPSS_TRAIN_FILES:
        (raw_dir / filename).write_text(sample, encoding="utf-8")

    monkeypatch.setattr(aerospace_builder, "RAW_DIR", raw_dir)
    monkeypatch.setattr(aerospace_builder, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(aerospace_builder, "PROVENANCE_PATH", raw_dir / "cmapss_provenance.json")

    out_path = processed_dir / "aerospace_orius.csv"
    aerospace_builder.convert_cmapss_to_orius(out_path)

    df = pd.read_csv(out_path)
    assert set(["flight_id", "step", "altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct", "ts_utc"]).issubset(df.columns)
    assert len(df) == 8
    manifest = json.loads((raw_dir / "cmapss_provenance.json").read_text())
    assert manifest["canonical_source"] is True
    assert manifest["used_fallback"] is False


def test_aerospace_public_adsb_builder_emits_proxy_runtime_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_root = tmp_path / "external"
    local_dir = external_root / "aerospace_flight_telemetry" / aerospace_public_builder.HF_LOCAL_DIRNAME
    local_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "aircraft_id": ["ac1"] * 8 + ["ac2"] * 8,
            "year": [2026] * 16,
            "month": [1] * 16,
            "day": [2] * 16,
            "hour": [3] * 16,
            "minute": [4] * 16,
            "second": [float(i) for i in range(8)] + [float(i) for i in range(8)],
            "altitude_ft": [3000 + 25 * i for i in range(16)],
            "ground_speed_kts": [180 + i for i in range(16)],
            "heading_deg": [5 * i for i in range(16)],
            "latitude": [40.0] * 16,
            "longitude": [-88.0] * 16,
            "aircraft_tail": ["N1"] * 8 + ["N2"] * 8,
            "data_age_sec": [0.5] * 16,
            "range_nm": [12.0] * 16,
            "bearing_deg": [45.0] * 16,
            "altitude_is_gnss": [True] * 16,
        }
    ).to_csv(local_dir / "tartanaviation_adsb_19k_clean.csv", index=False)

    raw_dir = tmp_path / "repo_raw"
    processed_dir = tmp_path / "processed"
    publication_dir = tmp_path / "publication"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    publication_dir.mkdir(parents=True)

    monkeypatch.setattr(aerospace_public_builder, "RAW_DIR", raw_dir)
    monkeypatch.setattr(aerospace_public_builder, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(aerospace_public_builder, "PUBLICATION_DIR", publication_dir)
    monkeypatch.setattr(aerospace_public_builder, "PROCESSED_CSV", processed_dir / "aerospace_public_adsb_runtime.csv")
    monkeypatch.setattr(aerospace_public_builder, "PROVENANCE_PATH", raw_dir / "public_adsb_proxy_provenance.json")
    monkeypatch.setattr(aerospace_public_builder, "SUMMARY_JSON", publication_dir / "aerospace_public_flight_runtime_summary.json")
    monkeypatch.setattr(aerospace_public_builder, "SUMMARY_CSV", publication_dir / "aerospace_public_flight_runtime_summary.csv")
    monkeypatch.setattr(aerospace_public_builder, "SUMMARY_MD", publication_dir / "aerospace_public_flight_runtime_summary.md")
    monkeypatch.setattr(aerospace_public_builder, "GOVERNANCE_CSV", publication_dir / "aerospace_public_flight_governance_matrix.csv")
    monkeypatch.setattr(aerospace_public_builder, "CALIBRATION_CSV", publication_dir / "aerospace_public_flight_calibration_diagnostics.csv")
    monkeypatch.setattr(aerospace_public_builder, "CANDIDATE_PARITY_CSV", publication_dir / "aerospace_public_flight_candidate_parity.csv")

    out_path = aerospace_public_builder.build_public_runtime(
        external_root=external_root,
        out_csv=aerospace_public_builder.PROCESSED_CSV,
        download=False,
    )

    built = pd.read_csv(out_path)
    assert {"flight_id", "step", "altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct", "ts_utc"}.issubset(built.columns)
    assert built["speed_source"].eq("groundspeed_proxy").all()
    provenance = json.loads((raw_dir / "public_adsb_proxy_provenance.json").read_text())
    assert provenance["canonical_source"] is False
    assert provenance["proxy_fields"]["airspeed_kt"] == "populated_from_ground_speed_kts"
    summary = json.loads((publication_dir / "aerospace_public_flight_runtime_summary.json").read_text())
    assert summary["lane_type"] == "bounded_public_adsb_runtime"
    assert (publication_dir / "aerospace_public_flight_governance_matrix.csv").exists()
    assert (publication_dir / "aerospace_public_flight_calibration_diagnostics.csv").exists()
    assert (publication_dir / "aerospace_public_flight_candidate_parity.csv").exists()


def test_aerospace_public_adsb_builder_defaults_to_external_root_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_root = tmp_path / "external"
    local_dir = external_root / "aerospace_flight_telemetry" / aerospace_public_builder.HF_LOCAL_DIRNAME
    local_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "aircraft_id": ["ac1"] * 8,
            "year": [2026] * 8,
            "month": [1] * 8,
            "day": [2] * 8,
            "hour": [3] * 8,
            "minute": [4] * 8,
            "second": [float(i) for i in range(8)],
            "altitude_ft": [3000 + 10 * i for i in range(8)],
            "ground_speed_kts": [180 + i for i in range(8)],
            "heading_deg": [5 * i for i in range(8)],
            "latitude": [40.0] * 8,
            "longitude": [-88.0] * 8,
            "aircraft_tail": ["N1"] * 8,
            "data_age_sec": [0.5] * 8,
        }
    ).to_csv(local_dir / "tartanaviation_adsb_19k_clean.csv", index=False)

    raw_dir = tmp_path / "repo_raw"
    processed_dir = tmp_path / "processed"
    publication_dir = tmp_path / "publication"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    publication_dir.mkdir(parents=True)

    monkeypatch.setenv(aerospace_public_builder.EXTERNAL_DATA_ROOT_ENV, str(external_root))
    monkeypatch.setattr(aerospace_public_builder, "RAW_DIR", raw_dir)
    monkeypatch.setattr(aerospace_public_builder, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(aerospace_public_builder, "PUBLICATION_DIR", publication_dir)
    monkeypatch.setattr(aerospace_public_builder, "PROCESSED_CSV", processed_dir / "aerospace_public_adsb_runtime.csv")
    monkeypatch.setattr(aerospace_public_builder, "PROVENANCE_PATH", raw_dir / "public_adsb_proxy_provenance.json")
    monkeypatch.setattr(aerospace_public_builder, "SUMMARY_JSON", publication_dir / "aerospace_public_flight_runtime_summary.json")
    monkeypatch.setattr(aerospace_public_builder, "SUMMARY_CSV", publication_dir / "aerospace_public_flight_runtime_summary.csv")
    monkeypatch.setattr(aerospace_public_builder, "SUMMARY_MD", publication_dir / "aerospace_public_flight_runtime_summary.md")
    monkeypatch.setattr(aerospace_public_builder, "GOVERNANCE_CSV", publication_dir / "aerospace_public_flight_governance_matrix.csv")
    monkeypatch.setattr(aerospace_public_builder, "CALIBRATION_CSV", publication_dir / "aerospace_public_flight_calibration_diagnostics.csv")
    monkeypatch.setattr(aerospace_public_builder, "CANDIDATE_PARITY_CSV", publication_dir / "aerospace_public_flight_candidate_parity.csv")

    out_path = aerospace_public_builder.build_public_runtime(
        external_root=None,
        out_csv=aerospace_public_builder.PROCESSED_CSV,
        download=False,
    )

    assert out_path.exists()
    assert (raw_dir / "public_adsb_proxy_provenance.json").exists()
    summary = json.loads((publication_dir / "aerospace_public_flight_runtime_summary.json").read_text(encoding="utf-8"))
    assert summary["lane_type"] == "bounded_public_adsb_runtime"


def test_dataset_registry_includes_navigation_row() -> None:
    cfg = DATASET_REGISTRY["NAVIGATION"]
    assert cfg.config_file == "configs/train_forecast_navigation.yaml"
    assert cfg.raw_data_path == "data/navigation/processed/navigation_orius.csv"
    assert cfg.feature_module == "orius.data_pipeline.build_features_navigation"
    assert cfg.provenance_path == "data/navigation/raw/kitti_odometry_provenance.json"
    assert cfg.maturity_tier == "shadow_synthetic"
    assert cfg.exact_blocker == "navigation_kitti_runtime_missing"


def test_real_data_loader_uses_repo_local_phase1_paths() -> None:
    assert real_data_loader.CCPP_PATH == Path("data/industrial/raw/CCPP.csv").resolve()
    assert real_data_loader.BIDMC_PATH == Path("data/healthcare/raw/bidmc_csv").resolve()
    assert real_data_loader.INDUSTRIAL_RUNTIME_PATH == Path("data/industrial/processed/industrial_orius.csv").resolve()
    assert real_data_loader.HEALTHCARE_RUNTIME_PATH == Path("data/healthcare/processed/healthcare_orius.csv").resolve()
    assert real_data_loader.AEROSPACE_RUNTIME_PATH == Path("data/aerospace/processed/aerospace_public_adsb_runtime.csv").resolve()
    assert real_data_loader.AEROSPACE_REALFLIGHT_PATH == Path("data/aerospace/processed/aerospace_realflight_runtime.csv").resolve()


def test_real_data_preflight_checks_repo_local_ccpp_file() -> None:
    assert real_data_preflight.INDUSTRIAL_PATHS == [Path("data/industrial/raw/CCPP.csv").resolve()]


def test_phase1_runtime_docs_do_not_reference_dead_legacy_paths() -> None:
    for rel_path in ("README.md", "DATA.md", "data/DATASET_DOWNLOAD_GUIDE.md", "scripts/prepare_datasets.py"):
        text = (Path(__file__).resolve().parents[1] / rel_path).read_text(encoding="utf-8")
        assert "data/ccpp" not in text
        assert "data/bidmc" not in text


def test_tool_status_finds_project_venv_tools_without_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "repo"
    module_path = repo_root / "src" / "orius" / "data_pipeline" / "real_data_contract.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# synthetic test path\n", encoding="utf-8")
    tool_dir = repo_root / ".venv" / "bin"
    tool_dir.mkdir(parents=True)
    tool_path = tool_dir / "hf"
    tool_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tool_path.chmod(0o755)

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(real_data_contract, "__file__", str(module_path))
    monkeypatch.setattr(real_data_contract.sys, "executable", str(repo_root / "python-bin" / "python"))

    status = real_data_contract.tool_status(("hf", "kaggle"))
    assert status["hf"] == str(tool_path)
    assert status["kaggle"] is None


def test_real_data_preflight_prefers_external_root_for_nonempty_dataset(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo" / "data" / "navigation" / "raw" / "kitti_odometry"
    repo_dir.mkdir(parents=True)
    external_root = tmp_path / "external"
    external_dir = external_root / "kitti_odometry"
    external_dir.mkdir(parents=True)
    (external_dir / "poses.txt").write_text("0 0 0\n", encoding="utf-8")

    check = real_data_preflight._external_dataset_check(
        repo_dir=repo_dir,
        external_dataset_key="kitti_odometry",
        explicit_root=external_root,
    )

    assert check["exists"] is True
    assert check["has_files"] is True
    assert check["source_kind"] == "external"
    assert check["path"] == str(external_dir)


def test_aerospace_real_flight_builder_emits_official_runtime_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_root = tmp_path / "external"
    provider_dir = external_root / "aerospace_flight_telemetry" / "provider_batch_01"
    provider_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "flight_id": ["f1", "f1", "f2", "f2"],
            "step": [0, 1, 0, 1],
            "altitude_m": [1000.0, 1010.0, 1200.0, 1215.0],
            "airspeed_kt": [180.0, 182.0, 190.0, 192.0],
            "bank_angle_deg": [5.0, 6.0, 3.0, 4.0],
            "fuel_remaining_pct": [90.0, 89.0, 88.0, 87.0],
            "ts_utc": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:01Z",
                "2026-01-01T00:01:00Z",
                "2026-01-01T00:01:01Z",
            ],
        }
    ).to_csv(provider_dir / "provider_runtime.csv", index=False)

    raw_dir = tmp_path / "repo_raw"
    processed_dir = tmp_path / "processed"
    publication_dir = tmp_path / "publication"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    publication_dir.mkdir(parents=True)

    monkeypatch.setattr(aerospace_real_builder, "RAW_DIR", raw_dir)
    monkeypatch.setattr(aerospace_real_builder, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(aerospace_real_builder, "PUBLICATION_DIR", publication_dir)
    monkeypatch.setattr(aerospace_real_builder, "PROCESSED_CSV", processed_dir / "aerospace_realflight_runtime.csv")
    monkeypatch.setattr(aerospace_real_builder, "PROVENANCE_PATH", raw_dir / "aerospace_realflight_provenance.json")
    monkeypatch.setattr(aerospace_real_builder, "SUMMARY_JSON", publication_dir / "aerospace_realflight_runtime_summary.json")
    monkeypatch.setattr(aerospace_real_builder, "SUMMARY_CSV", publication_dir / "aerospace_realflight_runtime_summary.csv")
    monkeypatch.setattr(aerospace_real_builder, "SUMMARY_MD", publication_dir / "aerospace_realflight_runtime_summary.md")

    out_path = aerospace_real_builder.build_real_flight_runtime(
        external_root=external_root,
        out_csv=aerospace_real_builder.PROCESSED_CSV,
    )

    built = pd.read_csv(out_path)
    assert {"flight_id", "step", "altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct", "ts_utc"}.issubset(built.columns)
    provenance = json.loads((raw_dir / "aerospace_realflight_provenance.json").read_text(encoding="utf-8"))
    assert provenance["canonical_source"] is True
    assert provenance["used_fallback"] is False
    summary = json.loads((publication_dir / "aerospace_realflight_runtime_summary.json").read_text(encoding="utf-8"))
    assert summary["lane_type"] == "official_provider_real_flight_runtime"
    assert summary["flight_count"] == 2
