#!/usr/bin/env python3
"""Build a dataset-integrity audit for the staged ORIUS data surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import itertools

from orius.data_pipeline.real_data_contract import file_sha256, utc_now_iso, write_json
from scripts._dataset_registry import DATASET_REGISTRY, repo_path

DEFAULT_JSON_OUT = REPO_ROOT / "reports" / "audit" / "data_integrity_audit.json"
DEFAULT_MD_OUT = REPO_ROOT / "reports" / "audit" / "data_integrity_audit.md"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_integrity(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return {"state": "missing", "problems": ["manifest_missing"]}

    problems: list[str] = []
    allow_missing_surfaces = str(manifest.get("status", "")).lower() == "blocked"
    raw_root = manifest.get("raw_root")
    if raw_root and not Path(str(raw_root)).exists() and not allow_missing_surfaces:
        problems.append("raw_root_missing")
    processed_output = manifest.get("processed_output")
    if processed_output and not Path(str(processed_output)).exists() and not allow_missing_surfaces:
        problems.append("processed_output_missing")
    checked_locations = [str(value) for value in manifest.get("checked_locations", [])]
    if any("pytest-" in value or "/private/var/folders/" in value for value in checked_locations):
        problems.append("ephemeral_checked_location")

    return {
        "state": "ok" if not problems else "stale",
        "problems": problems,
    }


def _actual_usage_label(dataset_key: str) -> str:
    cfg = DATASET_REGISTRY[dataset_key]
    canonical_runtime = repo_path(cfg.canonical_runtime_path)
    support_runtime = repo_path(cfg.support_runtime_path)
    features_path = repo_path(cfg.features_path)

    if canonical_runtime is not None and canonical_runtime.exists():
        return "yes"
    if support_runtime is not None and support_runtime.exists():
        return "partial"
    if features_path is not None and features_path.exists():
        return "yes"
    return "no"


def _parquet_rows_and_range(path: Path, *, timestamp_col: str = "timestamp") -> dict[str, Any]:
    frame = pd.read_parquet(path, columns=[timestamp_col])
    return {
        "rows": int(len(frame)),
        "date_start": str(frame[timestamp_col].iloc[0]),
        "date_end": str(frame[timestamp_col].iloc[-1]),
    }


def _csv_rows_and_range(path: Path, *, timestamp_col: str) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=[timestamp_col])
    return {
        "rows": int(len(frame)),
        "date_start": str(frame[timestamp_col].iloc[0]),
        "date_end": str(frame[timestamp_col].iloc[-1]),
    }


def _split_range(path: Path, *, timestamp_col: str = "timestamp") -> tuple[pd.Timestamp, pd.Timestamp]:
    frame = pd.read_parquet(path, columns=[timestamp_col])
    return pd.Timestamp(frame[timestamp_col].iloc[0]), pd.Timestamp(frame[timestamp_col].iloc[-1])


def _split_gap_hours(paths: list[Path], *, timestamp_col: str = "timestamp") -> dict[str, float]:
    ranges = [
        (path.name, *_split_range(path, timestamp_col=timestamp_col)) for path in paths if path.exists()
    ]
    gaps: dict[str, float] = {}
    for (left_name, _, left_end), (right_name, right_start, _) in itertools.pairwise(ranges):
        key = f"{left_name}->{right_name}"
        gaps[key] = round((right_start - left_end).total_seconds() / 3600.0, 2)
    return gaps


def _opsd_row() -> dict[str, Any]:
    cfg = DATASET_REGISTRY["DE"]
    manifest_path = repo_path(cfg.provenance_path)
    manifest = _load_json(manifest_path) if manifest_path is not None else None
    raw_csv = REPO_ROOT / "data" / "raw" / "time_series_60min_singleindex.csv"
    features_path = repo_path(cfg.features_path)
    splits_dir = repo_path(cfg.splits_path)
    raw_stats = _csv_rows_and_range(raw_csv, timestamp_col="utc_timestamp") if raw_csv.exists() else None
    processed_stats = (
        _parquet_rows_and_range(features_path)
        if features_path is not None and features_path.exists()
        else None
    )
    split_paths = (
        [
            splits_dir / "train.parquet",
            splits_dir / "calibration.parquet",
            splits_dir / "val.parquet",
            splits_dir / "test.parquet",
        ]
        if splits_dir is not None
        else []
    )
    split_gaps = _split_gap_hours(split_paths)
    claimed_rows = None if manifest is None else manifest.get("output_summary", {}).get("rows")
    claim_match = "couldn't verify"
    if claimed_rows is not None and processed_stats is not None:
        claim_match = "yes" if int(claimed_rows) == int(processed_stats["rows"]) else "no"
    return {
        "dataset": "OPSD Germany",
        "dataset_key": "opsd_germany",
        "raw_data_present": "yes"
        if raw_csv.exists()
        else "manifest-only"
        if manifest is not None
        else "missing",
        "claimed_rows_records_match": claim_match,
        "hash_matches": "no hash in manifest",
        "actually_used_by_corresponding_adapter": _actual_usage_label("DE"),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_integrity": _manifest_integrity(manifest),
        "maturity_tier": cfg.maturity_tier,
        "raw_rows": None if raw_stats is None else raw_stats["rows"],
        "raw_date_start_utc": None if raw_stats is None else raw_stats["date_start"],
        "raw_date_end_utc": None if raw_stats is None else raw_stats["date_end"],
        "defended_rows": None if processed_stats is None else processed_stats["rows"],
        "defended_date_start_utc": None if processed_stats is None else processed_stats["date_start"],
        "defended_date_end_utc": None if processed_stats is None else processed_stats["date_end"],
        "raw_sha256": file_sha256(raw_csv) if raw_csv.exists() else None,
        "split_gap_hours": split_gaps,
        "split_gap_policy_holds": all(value >= 24.0 for value in split_gaps.values()),
        "notes": [
            "The defended battery row is the processed 17,377-row surface, not the full 50,401-row OPSD raw archive.",
        ],
    }


def _eia_row(dataset_key: str) -> dict[str, Any]:
    cfg = DATASET_REGISTRY[dataset_key]
    manifest_path = repo_path(cfg.provenance_path)
    manifest = _load_json(manifest_path) if manifest_path is not None else None
    raw_root = REPO_ROOT / "data" / "raw" / "us_eia930"
    features_path = repo_path(cfg.features_path)
    processed_stats = (
        _parquet_rows_and_range(features_path)
        if features_path is not None and features_path.exists()
        else None
    )
    claim_match = "couldn't verify"
    if manifest is not None and processed_stats is not None:
        claim_match = "yes" if int(manifest["rows"]) == int(processed_stats["rows"]) else "no"
    return {
        "dataset": f"EIA-930 {cfg.ba_code}",
        "dataset_key": dataset_key.lower(),
        "raw_data_present": "yes"
        if raw_root.exists()
        else "manifest-only"
        if manifest is not None
        else "missing",
        "claimed_rows_records_match": claim_match,
        "hash_matches": "no hash in manifest",
        "actually_used_by_corresponding_adapter": _actual_usage_label(dataset_key),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_integrity": _manifest_integrity(manifest),
        "maturity_tier": "reference_support",
        "defended_rows": None if processed_stats is None else processed_stats["rows"],
        "defended_date_start_utc": None if processed_stats is None else processed_stats["date_start"],
        "defended_date_end_utc": None if processed_stats is None else processed_stats["date_end"],
        "notes": [
            "The staged defended surface ends on 2026-01-20/2026-01-20T12:00:00Z, not 2026-01-31.",
        ],
    }


def _navigation_row() -> dict[str, Any]:
    cfg = DATASET_REGISTRY["NAVIGATION"]
    manifest_path = repo_path(cfg.provenance_path)
    manifest = _load_json(manifest_path) if manifest_path is not None else None
    raw_root = repo_path(cfg.canonical_raw_source_path)
    runtime_path = repo_path(cfg.canonical_runtime_path)
    return {
        "dataset": "KITTI Odometry",
        "dataset_key": "kitti_odometry",
        "raw_data_present": "yes"
        if raw_root is not None and raw_root.exists()
        else "manifest-only"
        if manifest is not None
        else "missing",
        "claimed_rows_records_match": "couldn't verify",
        "hash_matches": "no hash in manifest",
        "actually_used_by_corresponding_adapter": _actual_usage_label("NAVIGATION"),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_integrity": _manifest_integrity(manifest),
        "maturity_tier": cfg.maturity_tier,
        "defended_runtime_present": bool(runtime_path is not None and runtime_path.exists()),
        "exact_blocker": cfg.exact_blocker,
        "notes": [
            "Navigation remains blocked. The repo does not stage the defended KITTI raw/runtime surface.",
        ],
    }


def _cmapss_row() -> dict[str, Any]:
    cfg = DATASET_REGISTRY["AEROSPACE"]
    manifest_path = repo_path(cfg.provenance_path)
    manifest = _load_json(manifest_path) if manifest_path is not None else None
    raw_dir = REPO_ROOT / "data" / "aerospace" / "raw"
    train_fd001 = raw_dir / "train_FD001.txt"
    test_fd001 = raw_dir / "test_FD001.txt"
    rul_fd001 = raw_dir / "RUL_FD001.txt"
    raw_present = all(path.exists() for path in (train_fd001, test_fd001, rul_fd001))

    train_frame = (
        pd.read_csv(train_fd001, sep=r"\s+", header=None, engine="python") if train_fd001.exists() else None
    )
    test_frame = (
        pd.read_csv(test_fd001, sep=r"\s+", header=None, engine="python") if test_fd001.exists() else None
    )
    rul_rows = sum(1 for _ in rul_fd001.open(encoding="utf-8")) if rul_fd001.exists() else 0

    claim_match = (
        "yes" if raw_present and train_frame is not None and test_frame is not None else "couldn't verify"
    )
    if raw_present and train_frame is not None and test_frame is not None:
        if int(train_frame[0].nunique()) != 100 or int(test_frame[0].nunique()) != 100 or rul_rows != 100:
            claim_match = "no"

    support_runtime = repo_path(cfg.support_runtime_path)
    canonical_runtime = repo_path(cfg.canonical_runtime_path)
    return {
        "dataset": "NASA C-MAPSS FD001",
        "dataset_key": "cmapss_fd001",
        "raw_data_present": "yes" if raw_present else "manifest-only" if manifest is not None else "missing",
        "claimed_rows_records_match": claim_match,
        "hash_matches": "no hash in manifest",
        "actually_used_by_corresponding_adapter": _actual_usage_label("AEROSPACE"),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_integrity": _manifest_integrity(manifest),
        "maturity_tier": cfg.maturity_tier,
        "train_units": None if train_frame is None else int(train_frame[0].nunique()),
        "test_units": None if test_frame is None else int(test_frame[0].nunique()),
        "train_rows": None if train_frame is None else int(len(train_frame)),
        "test_rows": None if test_frame is None else int(len(test_frame)),
        "sensor_channels": None if train_frame is None else int(train_frame.shape[1] - 5),
        "train_fd001_sha256": file_sha256(train_fd001) if train_fd001.exists() else None,
        "test_fd001_sha256": file_sha256(test_fd001) if test_fd001.exists() else None,
        "support_runtime_present": bool(support_runtime is not None and support_runtime.exists()),
        "canonical_runtime_present": bool(canonical_runtime is not None and canonical_runtime.exists()),
        "notes": [
            "C-MAPSS is the trainable aerospace companion surface only.",
            "The defended runtime row is still blocked on the canonical real-flight replay surface.",
        ],
    }


def _ccpp_row() -> dict[str, Any]:
    cfg = DATASET_REGISTRY["INDUSTRIAL"]
    manifest_path = repo_path(cfg.provenance_path)
    manifest = _load_json(manifest_path) if manifest_path is not None else None
    raw_csv = repo_path(cfg.canonical_raw_source_path)
    runtime_path = repo_path(cfg.canonical_runtime_path)

    raw_frame = pd.read_csv(raw_csv) if raw_csv is not None and raw_csv.exists() else None
    runtime_rows = None
    if runtime_path is not None and runtime_path.exists():
        runtime_rows = int(len(pd.read_csv(runtime_path)))

    claim_match = "couldn't verify"
    if raw_frame is not None and manifest is not None:
        manifest_rows = manifest.get("output_summary", {}).get("rows")
        claim_match = "yes" if runtime_rows is not None and int(manifest_rows) == runtime_rows else "no"

    return {
        "dataset": "UCI CCPP",
        "dataset_key": "ccpp",
        "raw_data_present": "yes"
        if raw_frame is not None
        else "manifest-only"
        if manifest is not None
        else "missing",
        "claimed_rows_records_match": claim_match,
        "hash_matches": "no hash in manifest",
        "actually_used_by_corresponding_adapter": _actual_usage_label("INDUSTRIAL"),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_integrity": _manifest_integrity(manifest),
        "maturity_tier": cfg.maturity_tier,
        "raw_rows": None if raw_frame is None else int(len(raw_frame)),
        "raw_columns": None if raw_frame is None else list(map(str, raw_frame.columns)),
        "raw_sha256": file_sha256(raw_csv) if raw_csv is not None and raw_csv.exists() else None,
        "runtime_rows": runtime_rows,
        "notes": [
            "The canonical industrial raw source is the repo-local CCPP.csv file. The legacy workbook bundle is companion provenance only.",
        ],
    }


def _healthcare_row() -> dict[str, Any]:
    cfg = DATASET_REGISTRY["HEALTHCARE"]
    manifest_path = repo_path(cfg.provenance_path)
    manifest = _load_json(manifest_path) if manifest_path is not None else None
    raw_root = repo_path(cfg.canonical_raw_source_path)
    runtime_path = repo_path(cfg.canonical_runtime_path)
    runtime_frame = pd.read_csv(runtime_path) if runtime_path is not None and runtime_path.exists() else None
    patient_count = None if runtime_frame is None else int(runtime_frame["patient_id"].nunique())
    mimic_candidates = [
        REPO_ROOT / "data" / "healthcare" / "raw" / "mimic_iii",
        REPO_ROOT / "data" / "healthcare" / "raw" / "mimiciii",
        REPO_ROOT / "data" / "healthcare" / "raw" / "mimic",
    ]
    mimic_present = any(path.exists() for path in mimic_candidates)
    claim_match = "couldn't verify"
    if runtime_frame is not None and manifest is not None:
        manifest_rows = manifest.get("output_summary", {}).get("rows")
        claim_match = "yes" if int(manifest_rows) == int(len(runtime_frame)) else "no"

    return {
        "dataset": "BIDMC runtime / MIMIC-III boundary",
        "dataset_key": "bidmc_runtime_boundary",
        "raw_data_present": "yes"
        if raw_root is not None and raw_root.exists()
        else "manifest-only"
        if manifest is not None
        else "missing",
        "claimed_rows_records_match": claim_match,
        "hash_matches": "no hash in manifest",
        "actually_used_by_corresponding_adapter": _actual_usage_label("HEALTHCARE"),
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_integrity": _manifest_integrity(manifest),
        "maturity_tier": cfg.maturity_tier,
        "runtime_rows": None if runtime_frame is None else int(len(runtime_frame)),
        "patient_count": patient_count,
        "mimic_iii_staged": mimic_present,
        "notes": [
            "BIDMC is the active healthcare runtime source in this repo.",
            "MIMIC-III is not staged here and does not back the executable healthcare row.",
        ],
    }


def build_report() -> dict[str, Any]:
    rows = [
        _opsd_row(),
        _eia_row("US_MISO"),
        _eia_row("US_PJM"),
        _eia_row("US_ERCOT"),
        _navigation_row(),
        _cmapss_row(),
        _ccpp_row(),
        _healthcare_row(),
    ]
    return {
        "generated_at_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "datasets": rows,
        "blocked_datasets": [
            row["dataset"] for row in rows if row.get("actually_used_by_corresponding_adapter") == "no"
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Data Integrity Audit",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
    ]
    for row in report["datasets"]:
        lines.extend(
            [
                f"## {row['dataset']}",
                f"- Raw data present: {row['raw_data_present']}",
                f"- Claimed rows/records match: {row['claimed_rows_records_match']}",
                f"- Hash matches: {row['hash_matches']}",
                f"- Actually used by corresponding adapter: {row['actually_used_by_corresponding_adapter']}",
                f"- Manifest integrity: {row['manifest_integrity']['state']}",
            ]
        )
        if row["manifest_integrity"]["problems"]:
            lines.append(f"- Manifest problems: {', '.join(row['manifest_integrity']['problems'])}")
        for note in row.get("notes", []):
            lines.append(f"- Note: {note}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a dataset-integrity audit for the staged ORIUS surfaces"
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT, help="Write the JSON audit here")
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT, help="Write the Markdown summary here")
    args = parser.parse_args()

    report = build_report()
    write_json(args.json_out, report)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(_markdown(report), encoding="utf-8")
    print(f"json -> {args.json_out}")
    print(f"markdown -> {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
