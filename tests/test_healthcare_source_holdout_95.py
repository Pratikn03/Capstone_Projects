from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_healthcare_source_holdout_surface import build_healthcare_source_holdout_surface
from scripts.run_healthcare_heldout_runtime_replay import run_healthcare_heldout_runtime_replay


def _write_healthcare_csv(
    path: Path, source_prefix: str, start: str, patients: int = 6, rows_per_patient: int = 6
) -> None:
    rows = []
    base = pd.Timestamp(start, tz="UTC")
    for patient in range(patients):
        for step in range(rows_per_patient):
            rows.append(
                {
                    "patient_id": f"{source_prefix}{patient}",
                    "ts_utc": (base + pd.Timedelta(minutes=patient * rows_per_patient + step)).isoformat(),
                    "hr_bpm": 80 + patient,
                    "spo2_pct": 96 if step % 5 else 88,
                    "respiratory_rate": 16,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_healthcare_source_holdout_marks_eicu_not_staged_and_preserves_boundaries(tmp_path: Path) -> None:
    bidmc = tmp_path / "bidmc.csv"
    mimic = tmp_path / "mimic.csv"
    out_dir = tmp_path / "heldout"
    _write_healthcare_csv(bidmc, "b", "2026-01-01T00:00:00Z")
    _write_healthcare_csv(mimic, "m", "2026-02-01T00:00:00Z")

    manifest = build_healthcare_source_holdout_surface(
        bidmc=bidmc,
        mimic=mimic,
        eicu_root=tmp_path / "missing_eicu",
        out_dir=out_dir,
        dev_sources=("bidmc",),
        holdout_sources=("mimic3", "eicu"),
        time_forward=True,
        patient_disjoint=True,
    )

    assert manifest["eicu_status"] == "not_staged"
    assert manifest["patient_disjoint"] is True
    assert manifest["time_forward"] is True
    assert set(manifest["holdout_sources_available"]) == {"mimic3"}
    assert "not live clinical deployment" in manifest["claim_boundary"]
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "test.parquet").exists()


def test_healthcare_heldout_replay_emits_required_comparators(tmp_path: Path) -> None:
    bidmc = tmp_path / "bidmc.csv"
    mimic = tmp_path / "mimic.csv"
    splits_dir = tmp_path / "heldout"
    out_dir = tmp_path / "runtime"
    _write_healthcare_csv(bidmc, "b", "2026-01-01T00:00:00Z")
    _write_healthcare_csv(mimic, "m", "2026-02-01T00:00:00Z")
    build_healthcare_source_holdout_surface(
        bidmc=bidmc,
        mimic=mimic,
        eicu_root=tmp_path / "missing_eicu",
        out_dir=splits_dir,
        dev_sources=("bidmc",),
        holdout_sources=("mimic3", "eicu"),
        time_forward=True,
        patient_disjoint=True,
    )

    manifest = run_healthcare_heldout_runtime_replay(
        splits_dir=splits_dir,
        out_dir=out_dir,
        comparators=("news2", "mews", "predictor_only", "conformal_alert_only", "fixed_conservative_alert"),
        require_source_holdout=True,
        require_time_forward=True,
    )

    summary = pd.read_csv(out_dir / "heldout_runtime_summary.csv")
    assert manifest["eicu_status"] == "not_staged"
    assert {
        "orius",
        "news2",
        "mews",
        "predictor_only",
        "conformal_alert_only",
        "fixed_conservative_alert",
    } == set(summary["controller"])
    assert float(summary.loc[summary["controller"] == "orius", "tsvr"].iloc[0]) == 0.0
    assert json.loads((out_dir / "heldout_runtime_manifest.json").read_text())["source_holdout"] is True
