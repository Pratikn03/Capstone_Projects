from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import scripts.refresh_real_data_manifests as refresh


def test_refresh_industrial_manifest_writes_primary_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    industrial_dir = tmp_path / "data" / "industrial"
    raw_dir = industrial_dir / "raw"
    processed_dir = industrial_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (raw_dir / "CCPP.csv").write_text("AT,V,AP,RH,PE\n1,2,3,4,5\n", encoding="utf-8")
    pd.DataFrame(
        {
            "sensor_id": [0],
            "step": [0],
            "temp_c": [20.0],
            "vacuum_cmhg": [40.0],
            "pressure_mbar": [1010.0],
            "humidity_pct": [50.0],
            "power_mw": [450.0],
            "ts_utc": ["2026-01-01T00:00:00Z"],
        }
    ).to_csv(processed_dir / "industrial_orius.csv", index=False)

    monkeypatch.setattr(refresh, "INDUSTRIAL_DATA_DIR", industrial_dir)

    result = refresh.refresh_industrial_manifest()
    manifest = json.loads((raw_dir / "ccpp_provenance.json").read_text(encoding="utf-8"))

    assert result["status"] == "refreshed"
    assert manifest["domain"] == "industrial"
    assert manifest["canonical_source"] is True


def test_refresh_av_manifest_records_legacy_only_when_waymo_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    av_dir = tmp_path / "data" / "av"
    raw_dir = av_dir / "raw" / "hee_dataset"
    processed_dir = av_dir / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (raw_dir / "README.md").write_text("legacy fixture", encoding="utf-8")
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
    manifest = json.loads((av_dir / "raw" / "hee_legacy_provenance.json").read_text(encoding="utf-8"))

    assert result["status"] == "legacy_only"
    assert result["blocker"] == "canonical_waymo_raw_missing"
    assert manifest["domain"] == "av"
    assert manifest["canonical_source"] is False


def test_refresh_aerospace_manifest_writes_runtime_contract_when_runtime_surface_missing(
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

    monkeypatch.setattr(refresh, "AEROSPACE_DATA_DIR", aerospace_dir)

    result = refresh.refresh_aerospace_manifest()
    runtime_contract = json.loads((raw_dir / "multi_flight_runtime_contract.json").read_text(encoding="utf-8"))

    assert result["status"] == "trainable_only"
    assert result["blocker"] == "aerospace_real_multi_flight_runtime_missing"
    assert runtime_contract["present"] is False


def test_refresh_aerospace_manifest_records_public_adsb_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aerospace_dir = tmp_path / "data" / "aerospace"
    raw_dir = aerospace_dir / "raw"
    processed_dir = aerospace_dir / "processed"
    runtime_dir = tmp_path / "external" / "aerospace_flight_telemetry"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)

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
    (runtime_dir / "adsb.csv").write_text("flight_id,step\nf1,0\n", encoding="utf-8")
    (raw_dir / "public_adsb_proxy_provenance.json").write_text('{"lane_type":"bounded_public_adsb_runtime"}', encoding="utf-8")

    monkeypatch.setattr(refresh, "AEROSPACE_DATA_DIR", aerospace_dir)
    monkeypatch.setattr(
        refresh,
        "resolve_repo_or_external_raw_dir",
        lambda *args, **kwargs: refresh.ResolvedRawSource(
            path=runtime_dir,
            source_kind="external",
            checked_locations=(str(runtime_dir),),
        ),
    )

    result = refresh.refresh_aerospace_manifest()
    runtime_contract = json.loads((raw_dir / "multi_flight_runtime_contract.json").read_text(encoding="utf-8"))

    assert result["status"] == "trainable_plus_public_support"
    assert result["blocker"] == "aerospace_real_multi_flight_runtime_missing"
    assert runtime_contract["present"] is True
    assert runtime_contract["surface_status"] == "public_adsb_proxy_only"
    assert runtime_contract["public_adsb_proxy_manifest"] is not None


def test_refresh_aerospace_manifest_transitions_public_support_to_refreshed_when_provider_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aerospace_dir = tmp_path / "data" / "aerospace"
    raw_dir = aerospace_dir / "raw"
    processed_dir = aerospace_dir / "processed"
    runtime_dir = tmp_path / "external" / "aerospace_flight_telemetry"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)

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
    (runtime_dir / "provider_trajectory.csv").write_text("flight_id,ts_utc,altitude_ft\n", encoding="utf-8")

    monkeypatch.setattr(refresh, "AEROSPACE_DATA_DIR", aerospace_dir)
    monkeypatch.setattr(
        refresh,
        "resolve_repo_or_external_raw_dir",
        lambda *args, **kwargs: refresh.ResolvedRawSource(
            path=runtime_dir,
            source_kind="external",
            checked_locations=(str(runtime_dir),),
        ),
    )

    result = refresh.refresh_aerospace_manifest()
    runtime_contract = json.loads((raw_dir / "multi_flight_runtime_contract.json").read_text(encoding="utf-8"))

    assert result["status"] == "refreshed"
    assert result["blocker"] == ""
    assert runtime_contract["surface_status"] == "provider_approved_ready"
    assert runtime_contract["public_adsb_proxy_manifest"] is None
