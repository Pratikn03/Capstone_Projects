from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "healthcare" / "processed" / "splits"


def _split_frames() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
        for name in ("train", "calibration", "val", "test")
    }


def test_live_healthcare_splits_are_patient_disjoint_and_time_monotone() -> None:
    splits = _split_frames()
    patient_sets = {name: set(frame["patient_id"].astype(str)) for name, frame in splits.items()}
    timestamps = {name: pd.to_datetime(frame["timestamp"], utc=True) for name, frame in splits.items()}

    for left, right in (
        ("train", "calibration"),
        ("train", "val"),
        ("train", "test"),
        ("calibration", "val"),
        ("calibration", "test"),
        ("val", "test"),
    ):
        assert patient_sets[left].isdisjoint(patient_sets[right]), (
            f"patient overlap between {left} and {right}"
        )

    assert timestamps["train"].max() < timestamps["calibration"].min()
    assert timestamps["calibration"].max() < timestamps["val"].min()
    assert timestamps["val"].max() < timestamps["test"].min()


def test_live_healthcare_split_summary_matches_live_parquets() -> None:
    splits = _split_frames()
    summary = (SPLITS_DIR / "SPLIT_SUMMARY.md").read_text(encoding="utf-8")
    counts = dict(re.findall(r"\|\s*(train|calibration|val|test)\s*\|\s*(\d+)\s*\|", summary))
    assert counts == {name: str(len(frame)) for name, frame in splits.items()}
