from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from orius.orius_bench import real_data_loader
import scripts.refresh_real_data_manifests as refresh
from scripts._dataset_registry import (
    DATASET_REGISTRY,
    get_runtime_dataset_config,
    get_runtime_dataset_path,
    get_runtime_source_label,
    iter_trainable_dataset_keys,
    repo_path,
    runtime_domain_configs,
)


def test_dataset_registry_runtime_helpers_match_phase1_truth() -> None:
    configs = runtime_domain_configs()

    assert configs["battery"].maturity_tier == "reference"
    assert configs["vehicle"].maturity_tier == "proof_validated"
    assert configs["industrial"].maturity_tier == "proof_validated"
    assert configs["healthcare"].maturity_tier == "proof_validated"
    assert configs["navigation"].maturity_tier == "shadow_synthetic"
    assert configs["aerospace"].maturity_tier == "experimental"

    navigation = get_runtime_dataset_config("navigation")
    aerospace = get_runtime_dataset_config("aerospace")
    assert navigation.exact_blocker == "navigation_kitti_runtime_missing"
    assert aerospace.exact_blocker == "aerospace_realflight_runtime_missing"

    assert get_runtime_dataset_path("navigation") is None
    assert get_runtime_source_label("navigation") == "missing"
    assert get_runtime_dataset_path("aerospace", allow_support_tier=True) is not None
    assert get_runtime_source_label("aerospace", allow_support_tier=True) == "support"


def test_dataset_registry_iterators_and_error_paths_cover_phase1_runtime_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = iter_trainable_dataset_keys()

    assert len(keys) == len(set(keys))
    assert "DE" in keys
    assert "AEROSPACE" in keys
    assert repo_path(None) is None

    with pytest.raises(KeyError, match="No runtime dataset config registered"):
        get_runtime_dataset_config("unknown-domain")

    canonical_path = tmp_path / "navigation_runtime.csv"
    support_path = tmp_path / "navigation_support.csv"
    canonical_path.write_text("x\n", encoding="utf-8")
    support_path.write_text("x\n", encoding="utf-8")

    navigation_cfg = DATASET_REGISTRY["NAVIGATION"]
    monkeypatch.setattr(navigation_cfg, "canonical_runtime_path", str(canonical_path))
    monkeypatch.setattr(navigation_cfg, "support_runtime_path", str(support_path))

    assert get_runtime_dataset_path("navigation") == canonical_path
    assert get_runtime_source_label("navigation") == "canonical"


def test_real_data_loader_covers_synthetic_fallbacks_and_validation_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clipped = real_data_loader._ou_series(0.5, 0.25, 0.2, n=8, seed=7, clip_lo=0.0, clip_hi=1.0)
    assert len(clipped) == 8
    assert all(0.0 <= value <= 1.0 for value in clipped)

    with pytest.raises(ValueError, match="is not configured"):
        real_data_loader._repo_registry_path("INDUSTRIAL", "support_runtime_path")

    missing_ccpp = tmp_path / "missing_ccpp.csv"
    missing_bidmc = tmp_path / "missing_bidmc"
    monkeypatch.setattr(real_data_loader, "CCPP_PATH", missing_ccpp)
    monkeypatch.setattr(real_data_loader, "BIDMC_PATH", missing_bidmc)

    assert len(real_data_loader.get_ccpp_rows(seed=3)) == 9568
    assert len(real_data_loader.get_bidmc_rows(seed=3)) == 4000


def test_real_data_loader_skips_invalid_rows_across_repo_local_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ccpp_path = tmp_path / "CCPP.csv"
    bidmc_file = tmp_path / "bidmc.csv"
    bidmc_dir = tmp_path / "bidmc_csv"
    bidmc_dir.mkdir(parents=True)
    bidmc_synth = bidmc_dir / "_synthetic_bidmc_vitals.csv"
    navigation_runtime = tmp_path / "navigation_orius.csv"
    aerospace_runtime = tmp_path / "aerospace_public_adsb_runtime.csv"
    av_runtime = tmp_path / "av_trajectories_orius.csv"
    industrial_runtime = tmp_path / "industrial_orius.csv"
    healthcare_runtime = tmp_path / "healthcare_orius.csv"

    ccpp_path.write_text("AT,V,AP,RH,PE\n1,2,3,4,5\nbad,2,3,4,5\n", encoding="utf-8")
    bidmc_file.write_text(
        "HR,SpO2,RR\n94,97,25\n95,98,\n96,nan,22\nbad,97,25\n",
        encoding="utf-8",
    )
    bidmc_synth.write_text("HR,SpO2,RR\n88,96,18\n", encoding="utf-8")
    navigation_runtime.write_text(
        "robot_id,step,x,y,vx,vy,ts_utc,source_sequence\n"
        "robot-1,0,1,2,0.1,0.2,2026-01-01T00:00:00Z,00\n"
        "robot-2,1,,2,0.1,0.2,2026-01-01T00:00:01Z,00\n",
        encoding="utf-8",
    )
    aerospace_runtime.write_text(
        "flight_id,step,altitude_m,airspeed_kt,bank_angle_deg,fuel_remaining_pct,ts_utc\n"
        "flight-1,0,3000,180,5,80,2026-01-01T00:00:00Z\n"
        "flight-2,1,bad,181,5,79,2026-01-01T00:00:01Z\n",
        encoding="utf-8",
    )
    av_runtime.write_text(
        "vehicle_id,step,position_m,speed_mps,speed_limit_mps,lead_position_m,ts_utc\n"
        "veh-1,0,10,8,13.4,35,2026-01-01T00:00:00Z\n"
        "veh-2,1,11,8,,36,2026-01-01T00:00:01Z\n",
        encoding="utf-8",
    )
    industrial_runtime.write_text(
        "sensor_id,step,temp_c,vacuum_cmhg,pressure_mbar,humidity_pct,power_mw,ts_utc\n"
        "sensor-1,0,20,40,1010,50,450,2026-01-01T00:00:00Z\n"
        "sensor-2,1,21,41,1011,51,bad,2026-01-01T00:00:01Z\n",
        encoding="utf-8",
    )
    healthcare_runtime.write_text(
        "patient_id,step,hr_bpm,spo2_pct,respiratory_rate,ts_utc\n"
        "patient-1,0,72,97,14,2026-01-01T00:00:00Z\n"
        "patient-2,1,73,bad,15,2026-01-01T00:00:01Z\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(real_data_loader, "CCPP_PATH", ccpp_path)
    monkeypatch.setattr(real_data_loader, "BIDMC_PATH", bidmc_dir)
    monkeypatch.setattr(real_data_loader, "BIDMC_SYNTHETIC_PATH", bidmc_synth)
    monkeypatch.setattr(real_data_loader, "NAVIGATION_PATH", navigation_runtime)
    monkeypatch.setattr(real_data_loader, "AEROSPACE_RUNTIME_PATH", aerospace_runtime)
    monkeypatch.setattr(real_data_loader, "AV_PATH", av_runtime)
    monkeypatch.setattr(real_data_loader, "INDUSTRIAL_RUNTIME_PATH", industrial_runtime)
    monkeypatch.setattr(real_data_loader, "HEALTHCARE_RUNTIME_PATH", healthcare_runtime)

    assert real_data_loader.load_ccpp_rows(ccpp_path) == [{"AT": 1.0, "V": 2.0, "AP": 3.0, "RH": 4.0, "PE": 5.0}]
    assert real_data_loader.load_bidmc_rows(bidmc_file) == [{"HR": 94.0, "SpO2": 97.0, "RR": 25.0}]
    assert real_data_loader.load_bidmc_rows(bidmc_dir) == [{"HR": 88.0, "SpO2": 96.0, "RR": 18.0}]
    assert len(real_data_loader.load_navigation_rows()) == 1
    assert len(real_data_loader.load_aerospace_runtime_rows()) == 1
    assert len(real_data_loader.load_vehicle_rows()) == 1
    assert len(real_data_loader.load_industrial_runtime_rows()) == 1
    assert len(real_data_loader.load_healthcare_runtime_rows()) == 1


def test_real_data_loader_loads_ccpp_and_bidmc_repo_local_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ccpp_path = tmp_path / "industrial" / "raw" / "CCPP.csv"
    bidmc_dir = tmp_path / "healthcare" / "raw" / "bidmc_csv"
    bidmc_synth = bidmc_dir / "_synthetic_bidmc_vitals.csv"
    ccpp_path.parent.mkdir(parents=True)
    bidmc_dir.mkdir(parents=True)

    ccpp_path.write_text("AT,V,AP,RH,PE\n1,2,3,4,5\n", encoding="utf-8")
    (bidmc_dir / "bidmc_01_Numerics.csv").write_text(
        "Time [s], HR, PULSE, RESP, SpO2\n0,94,93,25,97\n1,95,94,26,98\n",
        encoding="utf-8",
    )
    bidmc_synth.write_text("HR,SpO2,RR\n88,96,18\n", encoding="utf-8")

    monkeypatch.setattr(real_data_loader, "CCPP_PATH", ccpp_path)
    monkeypatch.setattr(real_data_loader, "BIDMC_PATH", bidmc_dir)
    monkeypatch.setattr(real_data_loader, "BIDMC_SYNTHETIC_PATH", bidmc_synth)

    ccpp_rows = real_data_loader.load_ccpp_rows(ccpp_path)
    bidmc_rows = real_data_loader.load_bidmc_rows(bidmc_dir)

    assert ccpp_rows == [{"AT": 1.0, "V": 2.0, "AP": 3.0, "RH": 4.0, "PE": 5.0}]
    assert bidmc_rows == [
        {"HR": 94.0, "SpO2": 97.0, "RR": 25.0},
        {"HR": 95.0, "SpO2": 98.0, "RR": 26.0},
    ]
    assert real_data_loader.get_ccpp_rows() == ccpp_rows
    assert real_data_loader.get_bidmc_rows() == bidmc_rows


def test_real_data_loader_runtime_row_loaders_and_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    av_path = tmp_path / "av.csv"
    industrial_runtime = tmp_path / "industrial_orius.csv"
    healthcare_runtime = tmp_path / "healthcare_orius.csv"
    navigation_runtime = tmp_path / "navigation_orius.csv"
    aerospace_support = tmp_path / "aerospace_public_adsb_runtime.csv"
    aerospace_realflight = tmp_path / "aerospace_realflight_runtime.csv"
    ccpp_path = tmp_path / "CCPP.csv"
    bidmc_dir = tmp_path / "bidmc_csv"
    bidmc_dir.mkdir(parents=True)

    av_path.write_text(
        "vehicle_id,step,position_m,speed_mps,speed_limit_mps,lead_position_m,ts_utc\n"
        "veh-1,0,10,8,13.4,35,2026-01-01T00:00:00Z\n",
        encoding="utf-8",
    )
    industrial_runtime.write_text(
        "sensor_id,step,temp_c,vacuum_cmhg,pressure_mbar,humidity_pct,power_mw,ts_utc\n"
        "sensor-1,0,20,40,1010,50,450,2026-01-01T00:00:00Z\n",
        encoding="utf-8",
    )
    healthcare_runtime.write_text(
        "patient_id,step,hr_bpm,spo2_pct,respiratory_rate,ts_utc\n"
        "patient-1,0,72,97,14,2026-01-01T00:00:00Z\n",
        encoding="utf-8",
    )
    navigation_runtime.write_text(
        "robot_id,step,x,y,vx,vy,ts_utc,source_sequence\n"
        "robot-1,0,1,2,0.1,0.2,2026-01-01T00:00:00Z,00\n",
        encoding="utf-8",
    )
    aerospace_support.write_text(
        "flight_id,step,altitude_m,airspeed_kt,bank_angle_deg,fuel_remaining_pct,ts_utc\n"
        "flight-1,0,3000,180,5,80,2026-01-01T00:00:00Z\n",
        encoding="utf-8",
    )
    aerospace_realflight.write_text(
        "flight_id,step,altitude_m,airspeed_kt,bank_angle_deg,fuel_remaining_pct,ts_utc\n"
        "flight-2,0,3100,185,4,82,2026-01-01T00:00:00Z\n",
        encoding="utf-8",
    )
    ccpp_path.write_text("AT,V,AP,RH,PE\n1,2,3,4,5\n", encoding="utf-8")
    (bidmc_dir / "bidmc_01_Numerics.csv").write_text(
        "Time [s], HR, PULSE, RESP, SpO2\n0,94,93,25,97\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(real_data_loader, "AV_PATH", av_path)
    monkeypatch.setattr(real_data_loader, "INDUSTRIAL_RUNTIME_PATH", industrial_runtime)
    monkeypatch.setattr(real_data_loader, "HEALTHCARE_RUNTIME_PATH", healthcare_runtime)
    monkeypatch.setattr(real_data_loader, "NAVIGATION_PATH", navigation_runtime)
    monkeypatch.setattr(real_data_loader, "AEROSPACE_RUNTIME_PATH", aerospace_support)
    monkeypatch.setattr(real_data_loader, "AEROSPACE_REALFLIGHT_PATH", aerospace_realflight)
    monkeypatch.setattr(real_data_loader, "CCPP_PATH", ccpp_path)
    monkeypatch.setattr(real_data_loader, "BIDMC_PATH", bidmc_dir)
    monkeypatch.setattr(real_data_loader, "BIDMC_SYNTHETIC_PATH", bidmc_dir / "_synthetic_bidmc_vitals.csv")

    assert real_data_loader.load_vehicle_rows()[0]["vehicle_id"] == "veh-1"
    assert real_data_loader.load_industrial_runtime_rows()[0]["power_mw"] == 450.0
    assert real_data_loader.load_healthcare_runtime_rows()[0]["spo2_pct"] == 97.0
    assert real_data_loader.load_navigation_rows()[0]["source_sequence"] == "00"
    assert real_data_loader.load_aerospace_runtime_rows()[0]["airspeed_kt"] == 180.0

    status = real_data_loader.dataset_status()
    assert status["ccpp"]["real_data"] is True
    assert status["bidmc"]["real_data"] is True
    assert status["navigation"]["real_data"] is True
    assert status["aerospace_runtime"]["real_data"] is True
    assert status["aerospace_support_runtime"]["real_data"] is True


def test_refresh_battery_manifest_records_reference_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    opsd_dir = raw_dir / "opsd-time_series-2020-10-06"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    opsd_dir.mkdir(parents=True)
    (opsd_dir / "time_series.csv").write_text("utc_timestamp,load_mw\n2020-01-01T00:00:00Z,1\n", encoding="utf-8")
    pd.DataFrame({"x": [1]}).to_parquet(processed_dir / "features.parquet", index=False)

    monkeypatch.setattr(refresh, "BATTERY_RAW_DIR", raw_dir)
    monkeypatch.setattr(refresh, "BATTERY_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(refresh, "BATTERY_MANIFEST_PATH", raw_dir / "opsd_germany_provenance.json")

    result = refresh.refresh_battery_manifest()
    manifest = json.loads((raw_dir / "opsd_germany_provenance.json").read_text(encoding="utf-8"))

    assert result["status"] == "refreshed"
    assert manifest["domain"] == "battery"
    assert manifest["canonical_source"] is True


def test_refresh_healthcare_manifest_records_bidmc_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthcare_dir = tmp_path / "data" / "healthcare"
    raw_dir = healthcare_dir / "raw" / "bidmc_csv"
    processed_dir = healthcare_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (raw_dir / "bidmc_01_Numerics.csv").write_text(
        "Time [s], HR, PULSE, RESP, SpO2\n0,94,93,25,97\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "patient_id": ["01"],
            "step": [0],
            "hr_bpm": [94.0],
            "spo2_pct": [97.0],
            "respiratory_rate": [25.0],
            "ts_utc": ["2026-01-01T00:00:00Z"],
        }
    ).to_csv(processed_dir / "healthcare_orius.csv", index=False)

    monkeypatch.setattr(refresh, "HEALTHCARE_DATA_DIR", healthcare_dir)

    result = refresh.refresh_healthcare_manifest()
    manifest = json.loads((healthcare_dir / "raw" / "bidmc_provenance.json").read_text(encoding="utf-8"))

    assert result["status"] == "refreshed"
    assert manifest["domain"] == "healthcare"
    assert manifest["canonical_source"] is True


def test_refresh_navigation_manifest_refreshes_when_kitti_surface_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    navigation_dir = tmp_path / "data" / "navigation"
    raw_dir = navigation_dir / "raw" / "kitti_odometry" / "dataset"
    poses_dir = raw_dir / "poses"
    seq_dir = raw_dir / "sequences" / "00"
    processed_dir = navigation_dir / "processed"
    poses_dir.mkdir(parents=True)
    seq_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (poses_dir / "00.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")
    (seq_dir / "times.txt").write_text("0.0\n", encoding="utf-8")
    pd.DataFrame(
        {
            "robot_id": ["robot-1"],
            "step": [0],
            "x": [1.0],
            "y": [2.0],
            "vx": [0.1],
            "vy": [0.2],
            "ts_utc": ["2026-01-01T00:00:00Z"],
            "source_sequence": ["00"],
        }
    ).to_csv(processed_dir / "navigation_orius.csv", index=False)

    monkeypatch.setattr(refresh, "NAVIGATION_DATA_DIR", navigation_dir)

    result = refresh.refresh_navigation_manifest()
    manifest = json.loads((navigation_dir / "raw" / "kitti_odometry_provenance.json").read_text(encoding="utf-8"))

    assert result["status"] == "refreshed"
    assert manifest["domain"] == "navigation"
    assert manifest["canonical_source"] is True


def test_refresh_av_manifest_refreshes_waymo_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    av_dir = tmp_path / "data" / "av"
    raw_dir = av_dir / "raw" / "waymo_open_motion" / "train"
    processed_dir = av_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (raw_dir / "sample.csv").write_text("track_id,frame_index,center_x,velocity_x,timestamp\nveh-1,0,1,5,2026-01-01T00:00:00Z\n", encoding="utf-8")
    pd.DataFrame(
        {
            "vehicle_id": ["veh-1"],
            "step": [0],
            "position_m": [10.0],
            "speed_mps": [8.0],
            "speed_limit_mps": [13.4],
            "lead_position_m": [35.0],
            "ts_utc": ["2026-01-01T00:00:00Z"],
            "source_split": ["train"],
        }
    ).to_csv(processed_dir / "av_trajectories_orius.csv", index=False)

    monkeypatch.setattr(refresh, "AV_DATA_DIR", av_dir)

    result = refresh.refresh_av_manifest()
    manifest = json.loads((av_dir / "raw" / "waymo_open_motion_provenance.json").read_text(encoding="utf-8"))

    assert result["status"] == "refreshed"
    assert manifest["domain"] == "av"
    assert manifest["canonical_source"] is True


def test_refresh_helpers_and_blocked_rows_cover_phase1_failure_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_parquet = tmp_path / "features.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(summary_parquet, index=False)

    assert refresh._tabular_summary(summary_parquet)["rows"] == 2
    assert refresh._kitti_raw_ready(None) is False
    assert refresh._looks_like_public_adsb_proxy(
        refresh.ResolvedRawSource(
            path=tmp_path / "missing_adsb_root",
            source_kind="repo_local",
            checked_locations=(str(tmp_path / "missing_adsb_root"),),
        )
    ) is False

    adsb_root = tmp_path / "adsb_root"
    adsb_root.mkdir(parents=True)
    (adsb_root / "tartanaviation_adsb_19k_clean.csv").write_text("flight_id\nf1\n", encoding="utf-8")
    assert refresh._looks_like_public_adsb_proxy(
        refresh.ResolvedRawSource(
            path=adsb_root,
            source_kind="repo_local",
            checked_locations=(str(adsb_root),),
        )
    ) is True

    industrial_dir = tmp_path / "data" / "industrial"
    healthcare_dir = tmp_path / "data" / "healthcare"
    navigation_dir = tmp_path / "data" / "navigation"
    aerospace_dir = tmp_path / "data" / "aerospace"
    monkeypatch.setattr(refresh, "INDUSTRIAL_DATA_DIR", industrial_dir)
    monkeypatch.setattr(refresh, "HEALTHCARE_DATA_DIR", healthcare_dir)
    monkeypatch.setattr(refresh, "NAVIGATION_DATA_DIR", navigation_dir)
    monkeypatch.setattr(refresh, "AEROSPACE_DATA_DIR", aerospace_dir)

    assert refresh.refresh_industrial_manifest()["status"] == "blocked"
    assert refresh.refresh_healthcare_manifest()["status"] == "blocked"
    assert refresh.refresh_navigation_manifest()["blocker"] == "navigation_kitti_runtime_missing"
    assert refresh.refresh_aerospace_manifest()["blocker"] == "aerospace_trainable_surface_missing"


def test_refresh_healthcare_manifest_tolerates_missing_patient_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthcare_dir = tmp_path / "data" / "healthcare"
    raw_dir = healthcare_dir / "raw" / "bidmc_csv"
    processed_dir = healthcare_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (raw_dir / "bidmc_01_Numerics.csv").write_text(
        "Time [s], HR, PULSE, RESP, SpO2\n0,94,93,25,97\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "step": [0],
            "hr_bpm": [94.0],
            "spo2_pct": [97.0],
            "respiratory_rate": [25.0],
            "ts_utc": ["2026-01-01T00:00:00Z"],
        }
    ).to_csv(processed_dir / "healthcare_orius.csv", index=False)

    monkeypatch.setattr(refresh, "HEALTHCARE_DATA_DIR", healthcare_dir)

    result = refresh.refresh_healthcare_manifest()
    manifest = json.loads((healthcare_dir / "raw" / "bidmc_provenance.json").read_text(encoding="utf-8"))

    assert result["status"] == "refreshed"
    assert manifest["patient_count"] is None


def test_refresh_av_manifest_blocks_when_no_canonical_or_legacy_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    av_dir = tmp_path / "data" / "av"
    raw_dir = av_dir / "raw"
    processed_dir = av_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "vehicle_id": ["veh-1"],
            "step": [0],
            "position_m": [10.0],
            "speed_mps": [8.0],
            "speed_limit_mps": [13.4],
            "lead_position_m": [35.0],
            "ts_utc": ["2026-01-01T00:00:00Z"],
            "source_split": ["train"],
        }
    ).to_csv(processed_dir / "av_trajectories_orius.csv", index=False)

    monkeypatch.setattr(refresh, "AV_DATA_DIR", av_dir)

    result = refresh.refresh_av_manifest()

    assert result["status"] == "blocked"
    assert result["blocker"] == "canonical_waymo_raw_missing"


def test_refresh_aerospace_manifest_derives_support_checked_locations_without_runtime_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aerospace_dir = tmp_path / "data" / "aerospace"
    raw_dir = aerospace_dir / "raw"
    processed_dir = aerospace_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    for name in ("train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt"):
        (raw_dir / name).write_text("sample", encoding="utf-8")
    pd.DataFrame(
        {
            "flight_id": ["fd001_unit_001"],
            "step": [0],
            "altitude_m": [3000.0],
            "airspeed_kt": [180.0],
            "bank_angle_deg": [5.0],
            "fuel_remaining_pct": [80.0],
            "ts_utc": ["2026-01-01T00:00:00Z"],
        }
    ).to_csv(processed_dir / "aerospace_orius.csv", index=False)
    pd.DataFrame(
        {
            "flight_id": ["pub-1"],
            "step": [0],
            "altitude_m": [3200.0],
            "airspeed_kt": [175.0],
            "bank_angle_deg": [4.0],
            "fuel_remaining_pct": [81.0],
            "ts_utc": ["2026-01-01T00:00:00Z"],
        }
    ).to_csv(processed_dir / "aerospace_public_adsb_runtime.csv", index=False)
    (raw_dir / "public_adsb_proxy_provenance.json").write_text('{"lane_type":"bounded_public_adsb_runtime"}', encoding="utf-8")

    monkeypatch.setattr(refresh, "AEROSPACE_DATA_DIR", aerospace_dir)
    monkeypatch.setattr(refresh, "resolve_repo_or_external_raw_dir", lambda *args, **kwargs: None)

    result = refresh.refresh_aerospace_manifest()
    runtime_contract = json.loads((raw_dir / "multi_flight_runtime_contract.json").read_text(encoding="utf-8"))

    assert result["status"] == "trainable_plus_public_support"
    assert runtime_contract["checked_locations"] == [str(processed_dir / "aerospace_public_adsb_runtime.csv")]


def test_refresh_aerospace_manifest_derives_provider_checked_locations_without_runtime_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aerospace_dir = tmp_path / "data" / "aerospace"
    raw_dir = aerospace_dir / "raw"
    processed_dir = aerospace_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    for name in ("train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt"):
        (raw_dir / name).write_text("sample", encoding="utf-8")
    pd.DataFrame(
        {
            "flight_id": ["fd001_unit_001"],
            "step": [0],
            "altitude_m": [3000.0],
            "airspeed_kt": [180.0],
            "bank_angle_deg": [5.0],
            "fuel_remaining_pct": [80.0],
            "ts_utc": ["2026-01-01T00:00:00Z"],
        }
    ).to_csv(processed_dir / "aerospace_orius.csv", index=False)
    pd.DataFrame(
        {
            "flight_id": ["rf-1"],
            "step": [0],
            "altitude_m": [3300.0],
            "airspeed_kt": [185.0],
            "bank_angle_deg": [5.0],
            "fuel_remaining_pct": [82.0],
            "ts_utc": ["2026-01-01T00:00:00Z"],
        }
    ).to_csv(processed_dir / "aerospace_realflight_runtime.csv", index=False)
    (raw_dir / "aerospace_realflight_provenance.json").write_text(
        '{"lane_type":"official_provider_real_flight_runtime"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(refresh, "AEROSPACE_DATA_DIR", aerospace_dir)
    monkeypatch.setattr(refresh, "resolve_repo_or_external_raw_dir", lambda *args, **kwargs: None)

    result = refresh.refresh_aerospace_manifest()
    runtime_contract = json.loads((raw_dir / "multi_flight_runtime_contract.json").read_text(encoding="utf-8"))

    assert result["status"] == "refreshed"
    assert runtime_contract["checked_locations"] == [str(processed_dir / "aerospace_realflight_runtime.csv")]
    assert runtime_contract["raw_root"] == str(processed_dir)


def test_refresh_manifest_main_filters_domains_and_reports_ready_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_path = tmp_path / "status.json"

    monkeypatch.setitem(
        refresh.REFRESHERS,
        "battery",
        lambda: refresh._report_row(
            domain="battery",
            status="blocked",
            canonical_source_present=False,
            processed_output_present=False,
            blocker="battery_missing",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_real_data_manifests.py",
            "--battery-only",
            "--out",
            str(out_path),
            "--require-bounded-universal-ready",
        ],
    )

    assert refresh.main() == 1
    report = json.loads(out_path.read_text(encoding="utf-8"))
    captured = capsys.readouterr()
    assert report["blocked_domains"] == ["battery"]
    assert "battery: blocked blocker=battery_missing" in captured.out

    monkeypatch.setitem(
        refresh.REFRESHERS,
        "battery",
        lambda: refresh._report_row(
            domain="battery",
            status="refreshed",
            canonical_source_present=True,
            processed_output_present=True,
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refresh_real_data_manifests.py",
            "--domain",
            "battery",
            "--out",
            str(out_path),
        ],
    )

    assert refresh.main() == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["refreshed_domains"] == ["battery"]
