#!/usr/bin/env python3
"""Refresh and audit real-data provenance manifests without rebuilding datasets."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from orius.data_pipeline.real_data_contract import (
    ResolvedRawSource,
    build_provenance_manifest,
    resolve_repo_or_external_raw_dir,
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
INDUSTRIAL_DATA_DIR = REPO_ROOT / "data" / "industrial"
HEALTHCARE_DATA_DIR = REPO_ROOT / "data" / "healthcare"
NAVIGATION_DATA_DIR = REPO_ROOT / "data" / "navigation"
AEROSPACE_DATA_DIR = REPO_ROOT / "data" / "aerospace"


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
    payload: dict[str, Any] = {
        "domain": domain,
        "status": status,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "canonical_source_present": bool(canonical_source_present),
        "processed_output_present": bool(processed_output_present),
        "blocker": blocker or "",
        "notes": list(notes or []),
    }
    if extras:
        payload.update(extras)
    return payload


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
        access_notes="Battery ingestion remains unchanged; this manifest only records provenance.",
        canonical_source=True,
        used_fallback=False,
        notes=[
            "battery remains the reference witness row",
            "companion weather, holiday, and carbon files stay in data/raw/",
        ],
        extras={
            "companion_files": [
                str(BATTERY_RAW_DIR / "time_series_60min_singleindex.csv"),
                str(BATTERY_RAW_DIR / "weather_berlin_hourly.csv"),
                str(BATTERY_RAW_DIR / "holidays_de.csv"),
                str(BATTERY_RAW_DIR / "carbon_signals.csv"),
            ]
        },
    )
    path = write_json(BATTERY_MANIFEST_PATH, manifest)
    return _report_row(
        domain="battery",
        status="refreshed",
        manifest_path=path,
        canonical_source_present=True,
        processed_output_present=features_path.exists(),
        notes=["battery remains the deepest witness row"],
    )


def refresh_industrial_manifest() -> dict[str, Any]:
    raw_dir = INDUSTRIAL_DATA_DIR / "raw"
    processed_path = INDUSTRIAL_DATA_DIR / "processed" / "industrial_orius.csv"
    raw_root = raw_dir / "ccpp" if (raw_dir / "ccpp").exists() else raw_dir
    canonical_source_present = raw_root.exists() and processed_path.exists()
    if not canonical_source_present:
        return _report_row(
            domain="industrial",
            status="blocked",
            manifest_path=raw_dir / "ccpp_provenance.json",
            canonical_source_present=raw_root.exists(),
            processed_output_present=processed_path.exists(),
            blocker="industrial_primary_surface_missing",
            notes=["Need primary CCPP raw source plus processed industrial_orius.csv."],
        )

    raw_source = ResolvedRawSource(
        path=raw_root,
        source_kind="repo_local",
        checked_locations=(str(raw_root),),
    )
    manifest = build_provenance_manifest(
        domain="industrial",
        dataset_key="ccpp",
        provider="UCI Combined Cycle Power Plant",
        version="Folds5x2 primary surface",
        raw_source=raw_source,
        processed_output=processed_path,
        output_summary=_tabular_summary(processed_path),
        raw_inventory=summarize_files(raw_source.path),
        source_urls=[
            "https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant",
            "https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems",
        ],
        license_notes="Follow UCI dataset terms for CCPP and any companion ZeMA assets.",
        access_notes="CCPP is the canonical industrial row. ZeMA remains a companion robustness corpus.",
        canonical_source=True,
        used_fallback=False,
        notes=[
            "industrial_orius.csv is the canonical defended industrial processed surface",
            "ZeMA hydraulic files remain companion raw evidence and are not the primary trainable row",
        ],
        extras={
            "companion_sources": {
                "zema_hydraulic_present": bool((raw_dir / "zema_hydraulic").exists()),
                "zema_hydraulic_dir": str(raw_dir / "zema_hydraulic"),
            }
        },
    )
    path = write_json(raw_dir / "ccpp_provenance.json", manifest)
    return _report_row(
        domain="industrial",
        status="refreshed",
        manifest_path=path,
        canonical_source_present=True,
        processed_output_present=True,
        notes=["bounded industrial row remains governed by the current plant family and replay protocol"],
    )


def refresh_healthcare_manifest() -> dict[str, Any]:
    raw_dir = HEALTHCARE_DATA_DIR / "raw"
    processed_path = HEALTHCARE_DATA_DIR / "processed" / "healthcare_orius.csv"
    raw_root = raw_dir / "bidmc_csv" if (raw_dir / "bidmc_csv").exists() else raw_dir / "bidmc"
    canonical_source_present = raw_root.exists() and processed_path.exists()
    if not canonical_source_present:
        return _report_row(
            domain="healthcare",
            status="blocked",
            manifest_path=raw_dir / "bidmc_provenance.json",
            canonical_source_present=raw_root.exists(),
            processed_output_present=processed_path.exists(),
            blocker="healthcare_primary_surface_missing",
            notes=["Need BIDMC raw corpus plus healthcare_orius.csv."],
        )

    raw_source = ResolvedRawSource(
        path=raw_root,
        source_kind="repo_local",
        checked_locations=(str(raw_root),),
    )
    patient_count = None
    try:
        patient_count = int(pd.read_csv(processed_path, usecols=["patient_id"])["patient_id"].nunique())
    except Exception:
        patient_count = None
    manifest = build_provenance_manifest(
        domain="healthcare",
        dataset_key="bidmc",
        provider="PhysioNet BIDMC PPG and Respiration Dataset",
        version="1.0.0",
        raw_source=raw_source,
        processed_output=processed_path,
        output_summary=_tabular_summary(processed_path),
        raw_inventory=summarize_files(raw_source.path),
        source_urls=["https://physionet.org/content/bidmc/1.0.0/"],
        license_notes="Follow PhysioNet credential, access, and citation requirements.",
        access_notes="Repo-local BIDMC CSV storage is the canonical healthcare raw contract.",
        canonical_source=True,
        used_fallback=False,
        notes=[
            "healthcare_orius.csv is built from the BIDMC numerics surface",
            "signals and breaths files remain companion raw evidence when present",
        ],
        extras={"patient_count": patient_count},
    )
    path = write_json(raw_dir / "bidmc_provenance.json", manifest)
    return _report_row(
        domain="healthcare",
        status="refreshed",
        manifest_path=path,
        canonical_source_present=True,
        processed_output_present=True,
        notes=["bounded healthcare row remains governed by the current monitoring and intervention contract"],
    )


def refresh_av_manifest() -> dict[str, Any]:
    raw_dir = AV_DATA_DIR / "raw"
    processed_path = AV_DATA_DIR / "processed" / "av_trajectories_orius.csv"
    waymo_source = resolve_repo_or_external_raw_dir(
        raw_dir / "waymo_open_motion",
        external_dataset_key="waymo_open_motion",
        required=False,
    )
    if processed_path.exists() and waymo_source is not None:
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
            access_notes="Waymo Open Motion is the canonical AV closure surface. Argoverse remains companion-only.",
            canonical_source=True,
            used_fallback=waymo_source.source_kind != "repo_local",
            notes=[
                "av_trajectories_orius.csv is governed by the canonical AV real-data contract",
                "Argoverse 2 remains a companion robustness surface outside the primary defended row",
            ],
        )
        path = write_json(raw_dir / "waymo_open_motion_provenance.json", manifest)
        return _report_row(
            domain="av",
            status="refreshed",
            manifest_path=path,
            canonical_source_present=True,
            processed_output_present=True,
            notes=["AV is bounded to the TTC plus predictive-entry-barrier contract in the current defended row"],
            extras={"source_kind": waymo_source.source_kind},
        )

    legacy_root = raw_dir / "hee_dataset"
    legacy_manifest_path = None
    notes = [
        "Waymo Open Motion remains the canonical AV closure target.",
        "Current repo-local HEE data is legacy compatibility material only.",
    ]
    if processed_path.exists() and legacy_root.exists():
        legacy_source = ResolvedRawSource(
            path=legacy_root,
            source_kind="repo_local",
            checked_locations=(str(legacy_root),),
        )
        legacy_manifest = build_provenance_manifest(
            domain="av",
            dataset_key="hee_legacy",
            provider="HEE legacy compatibility corpus",
            version="legacy_repo_fixture",
            raw_source=legacy_source,
            processed_output=processed_path,
            output_summary=_tabular_summary(processed_path),
            raw_inventory=summarize_files(legacy_root),
            source_urls=[],
            license_notes="Legacy compatibility corpus retained for local testing; not the canonical AV closure source.",
            access_notes="This manifest records legacy lineage only and does not satisfy the bounded-universal AV closure target.",
            canonical_source=False,
            used_fallback=True,
            notes=notes,
        )
        legacy_manifest_path = write_json(raw_dir / "hee_legacy_provenance.json", legacy_manifest)

    return _report_row(
        domain="av",
        status="legacy_only" if legacy_manifest_path is not None else "blocked",
        manifest_path=legacy_manifest_path,
        canonical_source_present=False,
        processed_output_present=processed_path.exists(),
        blocker="canonical_waymo_raw_missing",
        notes=notes,
    )


def refresh_navigation_manifest() -> dict[str, Any]:
    raw_dir = NAVIGATION_DATA_DIR / "raw"
    processed_path = NAVIGATION_DATA_DIR / "processed" / "navigation_orius.csv"
    manifest_path = raw_dir / "kitti_odometry_provenance.json"
    raw_source = resolve_repo_or_external_raw_dir(
        raw_dir / "kitti_odometry",
        external_dataset_key="kitti_odometry",
        required=False,
    )
    if raw_source is None or not processed_path.exists():
        return _report_row(
            domain="navigation",
            status="blocked",
            manifest_path=manifest_path if manifest_path.exists() else None,
            canonical_source_present=raw_source is not None,
            processed_output_present=processed_path.exists(),
            blocker="navigation_real_data_chain_incomplete",
            notes=[
                "Navigation remains blocked until the KITTI-backed processed surface and replay chain are both present.",
            ],
            extras={"source_kind": raw_source.source_kind if raw_source is not None else None},
        )

    manifest = build_provenance_manifest(
        domain="navigation",
        dataset_key="kitti_odometry",
        provider="KITTI Odometry",
        version="odometry benchmark",
        raw_source=raw_source,
        processed_output=processed_path,
        output_summary=_tabular_summary(processed_path),
        raw_inventory=summarize_files(raw_source.path),
        source_urls=["https://www.cvlibs.net/datasets/kitti/eval_odometry.php"],
        license_notes="Follow KITTI usage terms; raw payloads remain untracked.",
        access_notes="KITTI Odometry is the canonical navigation closure source.",
        canonical_source=True,
        used_fallback=raw_source.source_kind != "repo_local",
        notes=[
            "navigation_orius.csv is the required processed surface for defended navigation closure",
            "synthetic navigation traces are not sufficient for the defended bounded row",
        ],
    )
    path = write_json(manifest_path, manifest)
    return _report_row(
        domain="navigation",
        status="refreshed",
        manifest_path=path,
        canonical_source_present=True,
        processed_output_present=True,
        notes=["Navigation still requires replay and artifact refresh beyond raw-data/provenance closure."],
        extras={"source_kind": raw_source.source_kind},
    )


def refresh_aerospace_manifest() -> dict[str, Any]:
    raw_dir = AEROSPACE_DATA_DIR / "raw"
    processed_path = AEROSPACE_DATA_DIR / "processed" / "aerospace_orius.csv"
    manifest_path = raw_dir / "cmapss_provenance.json"
    train_files = [raw_dir / name for name in ("train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt")]
    train_surface_present = all(path.exists() for path in train_files)
    runtime_surface = resolve_repo_or_external_raw_dir(
        raw_dir / "aerospace_flight_telemetry",
        external_dataset_key="aerospace_flight_telemetry",
        required=False,
    )

    runtime_contract = {
        "generated_at_utc": utc_now_iso(),
        "domain": "aerospace",
        "contract": "runtime_replay_surface",
        "canonical_source": "provider_approved_multi_flight_telemetry",
        "present": runtime_surface is not None,
        "checked_locations": [] if runtime_surface is None else list(runtime_surface.checked_locations),
        "raw_root": None if runtime_surface is None else str(runtime_surface.path),
        "notes": [
            "C-MAPSS is the trainable degradation companion surface only.",
            "Bounded-universal aerospace closure requires a separate multi-flight runtime replay surface.",
        ],
    }
    runtime_contract_path = write_json(raw_dir / "multi_flight_runtime_contract.json", runtime_contract)

    if not (train_surface_present and processed_path.exists()):
        return _report_row(
            domain="aerospace",
            status="blocked",
            manifest_path=manifest_path if manifest_path.exists() else None,
            canonical_source_present=train_surface_present,
            processed_output_present=processed_path.exists(),
            blocker="aerospace_trainable_surface_missing",
            notes=[
                "Need C-MAPSS train files plus aerospace_orius.csv for the trainable companion surface.",
                "Real multi-flight telemetry is still required for defended runtime closure.",
            ],
            extras={"runtime_contract_path": str(runtime_contract_path)},
        )

    raw_source = ResolvedRawSource(
        path=raw_dir,
        source_kind="repo_local",
        checked_locations=(str(raw_dir),),
    )
    manifest = build_provenance_manifest(
        domain="aerospace",
        dataset_key="cmapss",
        provider="NASA C-MAPSS",
        version="FD001-FD004 train corpora",
        raw_source=raw_source,
        processed_output=processed_path,
        output_summary=_tabular_summary(processed_path),
        raw_inventory=summarize_files(raw_source.path),
        source_urls=["https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data"],
        license_notes="Follow NASA dataset terms; raw corpora remain untracked.",
        access_notes="C-MAPSS is the trainable aerospace surface only. Multi-flight telemetry governs defended runtime closure.",
        canonical_source=True,
        used_fallback=False,
        notes=[
            "aerospace_orius.csv records the trainable degradation companion surface",
            "real multi-flight telemetry remains a separate required runtime replay contract",
        ],
        extras={
            "runtime_contract_path": str(runtime_contract_path),
            "runtime_replay_surface_present": runtime_surface is not None,
        },
    )
    path = write_json(manifest_path, manifest)
    status = "refreshed" if runtime_surface is not None else "trainable_only"
    blocker = "" if runtime_surface is not None else "aerospace_real_multi_flight_runtime_missing"
    return _report_row(
        domain="aerospace",
        status=status,
        manifest_path=path,
        canonical_source_present=True,
        processed_output_present=True,
        blocker=blocker,
        notes=[
            "Aerospace remains blocked on the defended runtime replay surface until multi-flight telemetry is staged.",
        ],
        extras={"runtime_contract_path": str(runtime_contract_path)},
    )


REFRESHERS = {
    "battery": refresh_battery_manifest,
    "av": refresh_av_manifest,
    "industrial": refresh_industrial_manifest,
    "healthcare": refresh_healthcare_manifest,
    "navigation": refresh_navigation_manifest,
    "aerospace": refresh_aerospace_manifest,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh and audit real-data manifests")
    parser.add_argument("--battery-only", action="store_true", help="Refresh battery manifests only")
    parser.add_argument(
        "--domain",
        action="append",
        choices=sorted(REFRESHERS.keys()),
        help="Refresh only one or more selected domains.",
    )
    parser.add_argument("--out", type=Path, default=STATUS_REPORT_PATH, help="Write a status report to this JSON path")
    parser.add_argument(
        "--require-bounded-universal-ready",
        action="store_true",
        help="Exit non-zero unless every domain has a canonical source, processed output, and no recorded blocker.",
    )
    args = parser.parse_args()

    if args.battery_only:
        domains = ["battery"]
    elif args.domain:
        domains = args.domain
    else:
        domains = ["battery", "av", "industrial", "healthcare", "navigation", "aerospace"]

    rows = [REFRESHERS[domain]() for domain in domains]
    report = {
        "generated_at_utc": utc_now_iso(),
        "domains": rows,
        "refreshed_domains": [row["domain"] for row in rows if row["status"] == "refreshed"],
        "blocked_domains": [row["domain"] for row in rows if row["blocker"]],
        "bounded_universal_raw_contract_ready": all(
            row["canonical_source_present"] and row["processed_output_present"] and not row["blocker"]
            for row in rows
        ),
    }
    write_json(args.out, report)
    for row in rows:
        status = row["status"]
        blocker = f" blocker={row['blocker']}" if row.get("blocker") else ""
        print(f"{row['domain']}: {status}{blocker}")
        if row.get("manifest_path"):
            print(f"  manifest -> {row['manifest_path']}")
    print(f"status report -> {args.out}")

    if args.require_bounded_universal_ready and not report["bounded_universal_raw_contract_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
