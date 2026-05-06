from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from orius.data_pipeline.build_features_healthcare import (
    build_features,
    build_max_input_features,
    build_promoted_features,
)


def _write_bridge_csv(path: Path, patient_id: str, source_offset: int) -> None:
    rows = []
    for step in range(40):
        rows.append(
            {
                "timestamp": f"{patient_id}_t{step}",
                "target": 95.0 + (step % 3),
                "forecast": 95.5 + (step % 3),
                "reliability": 0.7 + 0.005 * step,
                "hr": 70.0 + source_offset + step * 0.1,
                "pulse": 69.0 + source_offset + step * 0.1,
                "resp": 14.0 + (step % 5),
                "patient_id": patient_id,
                "domain_label": "healthcare",
                "is_critical": step % 9 == 0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_max_input_features_merges_bidmc_and_mimic_sources(tmp_path: Path) -> None:
    bidmc_path = tmp_path / "healthcare_bidmc_orius.csv"
    mimic_path = tmp_path / "mimic3_healthcare_orius.csv"
    _write_bridge_csv(bidmc_path, "01", 0)
    _write_bridge_csv(mimic_path, "p0001", 5)

    out_dir = tmp_path / "max_input"
    features_path = build_max_input_features([bidmc_path, mimic_path], out_dir)

    features = pd.read_parquet(features_path)
    assert {"bidmc", "mimic3"} == set(features["source_dataset"].unique())
    assert features["patient_id"].nunique() == 2
    assert {
        "shock_index",
        "reliability",
        "forecast_spo2_pct",
        "hr_bpm_lag24",
        "spo2_pct_roll24_mean",
    }.issubset(features.columns)

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_patients"] == 2
    assert manifest["source_row_counts"]["bidmc"] == 40
    assert manifest["source_row_counts"]["mimic3"] == 40

    exported = pd.read_csv(out_dir / "healthcare_max_input_orius.csv")
    assert set(exported["is_critical"].unique()) == {False, True}


def test_build_promoted_features_accepts_bridge_schema_and_writes_splits(tmp_path: Path) -> None:
    mimic_path = tmp_path / "mimic3_healthcare_orius.csv"
    rows = []
    for patient_offset, patient_id in enumerate(("p0001", "p0002", "p0003", "p0004", "p0005")):
        for step in range(40):
            rows.append(
                {
                    "timestamp": f"{patient_id}_t{step}",
                    "target": 95.0 + (step % 3),
                    "forecast": 95.5 + (step % 3),
                    "reliability": 0.7 + 0.005 * step,
                    "hr": 70.0 + patient_offset + step * 0.1,
                    "pulse": 69.0 + patient_offset + step * 0.1,
                    "resp": 14.0 + (step % 5),
                    "patient_id": patient_id,
                    "domain_label": "healthcare",
                    "is_critical": step % 9 == 0,
                }
            )
    pd.DataFrame(rows).to_csv(mimic_path, index=False)

    out_dir = tmp_path / "processed"
    features_path = build_promoted_features(mimic_path, out_dir)

    features = pd.read_parquet(features_path)
    assert {"source_dataset", "patient_id", "spo2_pct", "forecast_spo2_pct", "reliability"}.issubset(
        features.columns
    )
    assert set(features["source_dataset"].unique()) == {"mimic3"}

    split_dir = out_dir / "splits"
    assert (split_dir / "train.parquet").exists()
    assert (split_dir / "calibration.parquet").exists()
    assert (split_dir / "val.parquet").exists()
    assert (split_dir / "test.parquet").exists()
    summary = (split_dir / "SPLIT_SUMMARY.md").read_text(encoding="utf-8")
    assert "contiguous_patient_blocks_by_earliest_timestamp" in summary
    assert "| train |" in summary


def test_build_features_legacy_contract_still_works(tmp_path: Path) -> None:
    input_path = tmp_path / "healthcare_orius.csv"
    rows = []
    for step in range(50):
        rows.append(
            {
                "patient_id": 1,
                "step": step,
                "hr_bpm": 80 + step * 0.1,
                "spo2_pct": 97 - (step % 4) * 0.1,
                "respiratory_rate": 16 + (step % 3),
                "ts_utc": f"2026-01-01T00:00:{step:02d}Z",
            }
        )
    pd.DataFrame(rows).to_csv(input_path, index=False)

    out_dir = tmp_path / "legacy"
    features_path = build_features(input_path, out_dir)

    features = pd.read_parquet(features_path)
    assert {"timestamp", "hr_bpm_lag24", "spo2_pct_lag24", "hour", "minute"}.issubset(features.columns)
    assert len(features) > 0


def test_build_features_writes_patient_disjoint_splits_and_live_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "healthcare_orius.csv"
    rows = []
    patient_lengths = {
        "p080901": 36,
        "p080942": 38,
        "p082100": 40,
        "p084200": 41,
        "p086383": 42,
        "p088500": 43,
        "p092816": 44,
        "p095000": 45,
    }
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    for patient_index, (patient_id, length) in enumerate(patient_lengths.items()):
        patient_start = start + pd.Timedelta(days=patient_index)
        for step in range(length):
            rows.append(
                {
                    "patient_id": patient_id,
                    "step": step,
                    "hr_bpm": 75 + patient_index + step * 0.1,
                    "spo2_pct": 96 - (step % 4) * 0.1,
                    "respiratory_rate": 14 + (step % 3),
                    "ts_utc": (patient_start + pd.Timedelta(seconds=step)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
    pd.DataFrame(rows).to_csv(input_path, index=False)

    out_dir = tmp_path / "legacy"
    build_features(input_path, out_dir)

    split_dir = out_dir / "splits"
    splits = {
        name: pd.read_parquet(split_dir / f"{name}.parquet")
        for name in ("train", "calibration", "val", "test")
    }
    summary = (split_dir / "SPLIT_SUMMARY.md").read_text(encoding="utf-8")

    patient_sets = {name: set(frame["patient_id"].astype(str)) for name, frame in splits.items()}
    for left, right in (
        ("train", "calibration"),
        ("train", "val"),
        ("train", "test"),
        ("calibration", "val"),
        ("calibration", "test"),
        ("val", "test"),
    ):
        assert patient_sets[left].isdisjoint(patient_sets[right])

    timestamps = {name: pd.to_datetime(frame["timestamp"], utc=True) for name, frame in splits.items()}
    assert timestamps["train"].max() < timestamps["calibration"].min()
    assert timestamps["calibration"].max() < timestamps["val"].min()
    assert timestamps["val"].max() < timestamps["test"].min()

    for name, frame in splits.items():
        assert f"| {name} | {len(frame)} |" in summary
