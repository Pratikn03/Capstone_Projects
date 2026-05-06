#!/usr/bin/env python3
"""Aggregate ORIUS 95+ validation tier outputs into one fail-closed manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = (
    REPO_ROOT / "reports" / "predeployment_external_validation" / "orius_95_validation_manifest.json"
)


def _path_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_controller_row(path: Path, controller: str = "orius") -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("controller") or "").strip() == controller:
                return row
    return {}


def _nuplan_status(runtime_dir: Path) -> dict[str, Any]:
    summary = runtime_dir / "runtime_summary.csv"
    row = _read_controller_row(summary)
    passed = bool(row)
    return {
        "status": "completed_bounded_runtime_replay" if passed else "missing",
        "completed": passed,
        "runtime_summary": _path_ref(summary) if summary.exists() else str(summary),
        "orius_tsvr": float(row.get("tsvr", 0.0)) if row else None,
        "claim_boundary": "nuPlan replay/surrogate evidence; not road deployment.",
    }


def _carla_status(carla_dir: Path) -> dict[str, Any]:
    manifest = _load_json(carla_dir / "carla_closed_loop_manifest.json")
    completed = bool(manifest.get("carla_completed"))
    summary_path = carla_dir / "carla_closed_loop_summary.csv"
    traces_path = carla_dir / "carla_closed_loop_traces.csv"
    return {
        "status": manifest.get("status", "missing"),
        "completed": completed,
        "manifest": _path_ref(carla_dir / "carla_closed_loop_manifest.json")
        if (carla_dir / "carla_closed_loop_manifest.json").exists()
        else str(carla_dir / "carla_closed_loop_manifest.json"),
        "summary": _path_ref(summary_path) if summary_path.exists() else str(summary_path),
        "traces": _path_ref(traces_path) if traces_path.exists() else str(traces_path),
        "episodes_completed": manifest.get("episodes_completed"),
        "steps_completed": manifest.get("steps_completed"),
        "safety_violations": manifest.get("safety_violations"),
        "certificate_valid_rate": manifest.get("certificate_valid_rate"),
        "blocked_reason": manifest.get("preflight", {}).get("blocked_reason")
        or manifest.get("blocked_reason", ""),
        "claim_boundary": manifest.get(
            "claim_boundary", "CARLA is not completed unless closed-loop traces exist."
        ),
    }


def _healthcare_status(healthcare_dir: Path) -> dict[str, Any]:
    manifest = _load_json(healthcare_dir / "heldout_runtime_manifest.json")
    row = _read_controller_row(healthcare_dir / "heldout_runtime_summary.csv")
    completed = bool(manifest and row)
    return {
        "status": manifest.get("status", "missing"),
        "completed": completed,
        "manifest": _path_ref(healthcare_dir / "heldout_runtime_manifest.json")
        if (healthcare_dir / "heldout_runtime_manifest.json").exists()
        else str(healthcare_dir / "heldout_runtime_manifest.json"),
        "orius_tsvr": float(row.get("tsvr", 0.0)) if row else None,
        "eicu_status": manifest.get("eicu_status", "unknown"),
        "source_holdout": bool(manifest.get("source_holdout", False)),
        "time_forward": bool(manifest.get("time_forward", False)),
        "claim_boundary": manifest.get("claim_boundary", "Healthcare heldout replay not completed."),
    }


def _battery_status(battery_hil_dir: Path) -> dict[str, Any]:
    summary = _load_json(battery_hil_dir / "hil_summary.json")
    completed = bool(summary)
    return {
        "status": "completed_software_hil" if completed else "missing",
        "completed": completed,
        "evidence_type": "software_hil",
        "physical_hil_completed": False,
        "summary": _path_ref(battery_hil_dir / "hil_summary.json")
        if (battery_hil_dir / "hil_summary.json").exists()
        else str(battery_hil_dir / "hil_summary.json"),
        "total_violations": summary.get("total_violations") if summary else None,
        "claim_boundary": "Software HIL/simulator evidence only; not physical bench or field deployment.",
    }


def build_95_validation_manifest(
    *,
    nuplan_runtime: Path,
    carla_dir: Path,
    healthcare_dir: Path,
    battery_hil_dir: Path,
    out: Path = DEFAULT_OUT,
) -> dict[str, Any]:
    domains = {
        "av_nuplan": _nuplan_status(nuplan_runtime),
        "av_carla": _carla_status(carla_dir),
        "healthcare": _healthcare_status(healthcare_dir),
        "battery": _battery_status(battery_hil_dir),
    }
    bounded_package_passed = bool(
        domains["av_nuplan"]["completed"]
        and domains["healthcare"]["completed"]
        and domains["battery"]["completed"]
    )
    full_95_external_validation_complete = bool(
        bounded_package_passed
        and domains["av_carla"]["completed"]
        and domains["healthcare"].get("eicu_status") == "staged"
        and domains["battery"].get("physical_hil_completed")
    )
    manifest = {
        "status": "bounded_predeployment_complete" if bounded_package_passed else "incomplete",
        "bounded_package_passed": bounded_package_passed,
        "full_95_external_validation_complete": full_95_external_validation_complete,
        "domains": domains,
        "claim_boundary": (
            "Bounded predeployment validation only. CARLA, eICU, and physical HIL are "
            "completed claims only when their domain flags are true."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nuplan-runtime", type=Path, required=True)
    parser.add_argument("--carla-dir", type=Path, required=True)
    parser.add_argument("--healthcare-dir", type=Path, required=True)
    parser.add_argument("--battery-hil-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    manifest = build_95_validation_manifest(
        nuplan_runtime=args.nuplan_runtime,
        carla_dir=args.carla_dir,
        healthcare_dir=args.healthcare_dir,
        battery_hil_dir=args.battery_hil_dir,
        out=args.out,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "full_95_external_validation_complete": manifest["full_95_external_validation_complete"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
