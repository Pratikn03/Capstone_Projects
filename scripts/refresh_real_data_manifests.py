#!/usr/bin/env python3
"""Refresh repo-truthful provenance manifests for the active 3-domain ORIUS program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from orius.data_pipeline.real_data_contract import (
    ResolvedRawSource,
    build_provenance_manifest,
    resolve_repo_or_external_raw_dir,
    resolved_source_has_files,
    summarize_csv_output,
    summarize_files,
    utc_now_iso,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_REPORT_PATH = REPO_ROOT / "reports" / "real_data_contract_status.json"

BATTERY_RAW_DIR = REPO_ROOT / "data" / "raw"
BATTERY_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
BATTERY_MANIFEST_PATH = BATTERY_RAW_DIR / "opsd_germany_provenance.json"

AV_DATA_DIR = REPO_ROOT / "data" / "av"
HEALTHCARE_DATA_DIR = REPO_ROOT / "data" / "healthcare"


def _parquet_summary(path: Path) -> dict[str, object]:
    df = pd.read_parquet(path)
    return {
        "processed_output": str(path),
        "rows": int(len(df)),
        "columns": list(map(str, df.columns)),
        "generated_at_utc": utc_now_iso(),
    }


def _tabular_summary(path: Path) -> dict[str, Any]:
    if path.suffix == ".parquet":
        return _parquet_summary(path)
    return summarize_csv_output(path)


def _report_row(
    *,
    domain: str,
    status: str,
    manifest_path: Path | None = None,
    canonical_source_present: bool,
    processed_output_present: bool,
    blocker: str | None = None,
    notes: list[str] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_text = None
    if manifest_path is not None:
        try:
            manifest_text = str(manifest_path.relative_to(REPO_ROOT))
        except ValueError:
            manifest_text = str(manifest_path)
    payload: dict[str, Any] = {
        "domain": domain,
        "status": status,
        "manifest_path": manifest_text,
        "canonical_source_present": bool(canonical_source_present),
        "processed_output_present": bool(processed_output_present),
        "blocker": blocker or "",
        "notes": list(notes or []),
    }
    if extras:
        payload.update(extras)
    return payload


def _has_raw_files(raw_source: ResolvedRawSource | None) -> bool:
    return resolved_source_has_files(raw_source)


def refresh_battery_manifest() -> dict[str, Any]:
    raw_source = ResolvedRawSource(
        path=BATTERY_RAW_DIR,
        source_kind="repo_local",
        checked_locations=(str(BATTERY_RAW_DIR),),
    )
    features_path = BATTERY_PROCESSED_DIR / "features.parquet"
    manifest = build_provenance_manifest(
        domain="battery",
        dataset_key="opsd_germany",
        provider="Open Power System Data + SMARD",
        version="2020-10-06",
        raw_source=raw_source,
        processed_output=features_path,
        output_summary=_parquet_summary(features_path),
        raw_inventory=summarize_files(BATTERY_RAW_DIR / "opsd-time_series-2020-10-06"),
        source_urls=[
            "https://open-power-system-data.org/",
            "https://www.smard.de/",
            "https://open-meteo.com/",
        ],
        license_notes="Follow OPSD, SMARD, and Open-Meteo provider terms.",
        access_notes="Battery remains the theorem-grade witness row.",
        canonical_source=True,
        used_fallback=False,
        notes=["battery remains the deepest witness row"],
    )
    path = write_json(BATTERY_MANIFEST_PATH, manifest)
    return _report_row(
        domain="battery",
        status="refreshed",
        manifest_path=path,
        canonical_source_present=True,
        processed_output_present=features_path.exists(),
        notes=["battery remains the canonical witness row"],
    )


def refresh_av_manifest() -> dict[str, Any]:
    raw_dir = AV_DATA_DIR / "raw"
    processed_path = AV_DATA_DIR / "processed" / "av_trajectories_orius.csv"
    waymo_source = resolve_repo_or_external_raw_dir(
        raw_dir / "waymo_open_motion",
        external_dataset_key="waymo_open_motion",
        required=False,
    )
    waymo_ready = processed_path.exists() and _has_raw_files(waymo_source)
    if not waymo_ready or waymo_source is None:
        return _report_row(
            domain="av",
            status="blocked",
            manifest_path=raw_dir / "waymo_open_motion_provenance.json",
            canonical_source_present=bool(waymo_source and _has_raw_files(waymo_source)),
            processed_output_present=processed_path.exists(),
            blocker="canonical_waymo_raw_missing",
            notes=["Waymo Open Motion is the only canonical AV raw-data contract on the current branch."],
        )

    manifest = build_provenance_manifest(
        domain="av",
        dataset_key="waymo_open_motion",
        provider="Waymo Open Dataset",
        version="motion",
        raw_source=waymo_source,
        processed_output=processed_path,
        output_summary=_tabular_summary(processed_path),
        raw_inventory=summarize_files(waymo_source.path),
        source_urls=[
            "https://waymo.com/open/data/motion/",
            "https://waymo.com/open/faq/",
        ],
        license_notes="Waymo Open Dataset license applies; raw payloads remain untracked.",
        access_notes="AV is bounded to the TTC plus predictive-entry-barrier contract.",
        canonical_source=True,
        used_fallback=waymo_source.source_kind != "repo_local",
        notes=["av_trajectories_orius.csv is the canonical AV closure surface"],
    )
    path = write_json(raw_dir / "waymo_open_motion_provenance.json", manifest)
    return _report_row(
        domain="av",
        status="refreshed",
        manifest_path=path,
        canonical_source_present=True,
        processed_output_present=True,
        notes=["AV remains a promoted bounded row under the current contract"],
        extras={"source_kind": waymo_source.source_kind},
    )


def refresh_healthcare_manifest() -> dict[str, Any]:
    raw_dir = HEALTHCARE_DATA_DIR / "raw"
    mimic_dir = HEALTHCARE_DATA_DIR / "mimic3" / "processed"
    processed_path = mimic_dir / "mimic3_healthcare_orius.csv"
    manifest_path = mimic_dir / "mimic3_manifest.json"
    if not (processed_path.exists() and manifest_path.exists()):
        return _report_row(
            domain="healthcare",
            status="blocked",
            manifest_path=manifest_path,
            canonical_source_present=manifest_path.exists(),
            processed_output_present=processed_path.exists(),
            blocker="healthcare_primary_surface_missing",
            notes=["Need the promoted MIMIC-III manifest plus mimic3_healthcare_orius.csv."],
        )

    raw_root = raw_dir / "mimic3"
    raw_source = ResolvedRawSource(
        path=raw_root,
        source_kind="repo_local",
        checked_locations=(str(raw_root),),
    )
    try:
        patient_count = int(pd.read_csv(processed_path, usecols=["patient_id"])["patient_id"].nunique())
    except Exception:
        patient_count = None
    manifest = build_provenance_manifest(
        domain="healthcare",
        dataset_key="mimic3",
        provider="PhysioNet MIMIC-III Waveform Database Matched Subset",
        version="matched_subset_bridge",
        raw_source=raw_source,
        processed_output=processed_path,
        output_summary=_tabular_summary(processed_path),
        raw_inventory=summarize_files(raw_source.path) if raw_source.path.exists() else {},
        source_urls=[
            "https://physionet.org/content/mimic3wdb-matched/1.0/",
            "https://physionet.org/content/mimiciii/1.4/",
        ],
        license_notes="Follow PhysioNet credential, access, and citation requirements.",
        access_notes="Healthcare is promoted only through the bounded MIMIC monitoring row.",
        canonical_source=True,
        used_fallback=False,
        notes=[
            "mimic3_healthcare_orius.csv is the promoted healthcare runtime and evaluation surface",
            "BIDMC remains supplemental and is not canonical on this branch",
        ],
        extras={"patient_count": patient_count},
    )
    path = write_json(manifest_path, manifest)
    return _report_row(
        domain="healthcare",
        status="refreshed",
        manifest_path=path,
        canonical_source_present=True,
        processed_output_present=True,
        notes=["healthcare remains a promoted bounded row under MIMIC monitoring semantics"],
    )


REFRESHERS = {
    "battery": refresh_battery_manifest,
    "av": refresh_av_manifest,
    "healthcare": refresh_healthcare_manifest,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh ORIUS real-data manifests for the active 3-domain program"
    )
    parser.add_argument("--domain", dest="domains", action="append", choices=sorted(REFRESHERS.keys()))
    args = parser.parse_args()

    domains = args.domains or ["battery", "av", "healthcare"]
    rows = [REFRESHERS[domain]() for domain in domains]
    report = {
        "generated_at_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "domains": rows,
        "submission_scope": "battery_av_healthcare",
    }
    write_json(STATUS_REPORT_PATH, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
