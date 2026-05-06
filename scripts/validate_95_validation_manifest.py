#!/usr/bin/env python3
"""Validate ORIUS 95+ validation manifest claim boundaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    REPO_ROOT / "reports" / "predeployment_external_validation" / "orius_95_validation_manifest.json"
)


def _resolve_artifact(value: object) -> Path:
    if value in (None, ""):
        return Path("")
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def _read_first_csv_row(path: Path) -> dict[str, str]:
    if not str(path) or not path.exists() or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return next(csv.DictReader(handle), {})


def _count_csv_rows(path: Path) -> int:
    if not str(path) or not path.exists() or not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _intish(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def validate_manifest(path: Path = DEFAULT_MANIFEST) -> list[str]:
    findings: list[str] = []
    if not path.exists():
        return [f"Missing 95 validation manifest: {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    domains = payload.get("domains", {})
    carla = domains.get("av_carla", {})
    healthcare = domains.get("healthcare", {})
    battery = domains.get("battery", {})

    if carla.get("completed") and carla.get("status") != "completed_closed_loop":
        findings.append("CARLA cannot be completed unless status=completed_closed_loop.")
    if carla.get("completed"):
        missing_closed_loop_artifact = False
        for key in ("manifest", "summary", "traces"):
            raw_target = carla.get(key, "")
            target = _resolve_artifact(raw_target)
            if not raw_target or not target.exists():
                findings.append(f"CARLA completed claim points to missing artifact: {target}")
                if key in {"summary", "traces"}:
                    missing_closed_loop_artifact = True
        if missing_closed_loop_artifact or not carla.get("summary") or not carla.get("traces"):
            findings.append("CARLA completed claim is missing closed-loop summary/traces artifacts.")
        summary_path = _resolve_artifact(carla.get("summary", ""))
        traces_path = _resolve_artifact(carla.get("traces", ""))
        summary = _read_first_csv_row(summary_path)
        trace_rows = _count_csv_rows(traces_path)
        if summary and str(summary.get("status", "")) != "completed_closed_loop":
            findings.append("CARLA completed summary does not have status=completed_closed_loop.")
        if summary and _intish(summary.get("episodes_completed")) <= 0:
            findings.append("CARLA completed summary has no completed episodes.")
        if summary and _intish(summary.get("steps_completed")) <= 0:
            findings.append("CARLA completed summary has no completed steps.")
        if traces_path.exists() and trace_rows <= 0:
            findings.append("CARLA completed traces contain no closed-loop rows.")
    if healthcare.get("eicu_completed") and healthcare.get("eicu_status") != "staged":
        findings.append("eICU cannot be completed unless eicu_status=staged.")
    if (
        "eicu" in str(healthcare.get("claim_boundary", "")).lower()
        and healthcare.get("eicu_status") != "staged"
    ):
        if "not" not in str(healthcare.get("claim_boundary", "")).lower():
            findings.append(
                "Healthcare claim boundary mentions eICU without marking it uncompleted/not staged."
            )
    if battery.get("physical_hil_completed"):
        findings.append("Physical HIL is out of scope for the software-HIL run.")
    if battery.get("evidence_type") != "software_hil":
        findings.append("Battery evidence type must remain software_hil for this run.")
    if payload.get("full_95_external_validation_complete") and (
        not carla.get("completed")
        or healthcare.get("eicu_status") != "staged"
        or not battery.get("physical_hil_completed")
    ):
        findings.append(
            "full_95_external_validation_complete is true without CARLA+eICU+physical-HIL completion."
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    findings = validate_manifest(args.manifest)
    if findings:
        print("[validate_95_validation_manifest] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_95_validation_manifest] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
