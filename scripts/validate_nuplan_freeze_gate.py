#!/usr/bin/env python3
"""Validate bounded nuPlan replay evidence for a full AV predeployment freeze."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = (
    REPO_ROOT / "reports" / "predeployment_external_validation" / "nuplan_closed_loop_summary.csv"
)
DEFAULT_TRACES = REPO_ROOT / "reports" / "predeployment_external_validation" / "nuplan_closed_loop_traces.csv"
DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT / "reports" / "predeployment_external_validation" / "nuplan_closed_loop_manifest.json"
)
NUPLAN_SURFACE = "nuplan_allzip_grouped_runtime_replay_surrogate"
NUPLAN_STATUS = "completed_bounded_replay_not_carla"
MAX_NUPLAN_RUNTIME_TSVR = 1e-3
MIN_NUPLAN_PASS_RATE = 1.0 - MAX_NUPLAN_RUNTIME_TSVR


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _floatish(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_summary(path: Path) -> tuple[dict[str, str], list[str]]:
    findings: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 1:
        findings.append(f"nuPlan freeze summary must contain exactly one row; found {len(rows)}")
    return (rows[0] if rows else {}), findings


def _trace_stats(path: Path) -> dict[str, Any]:
    rows = 0
    surfaces: set[str] = set()
    scenarios: set[str] = set()
    controllers: set[str] = set()
    orius_rows = 0
    orius_true_violations = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            surface = str(row.get("validation_surface", "")).strip()
            controller = str(row.get("controller", "")).strip()
            if surface:
                surfaces.add(surface)
            if row.get("scenario_id"):
                scenarios.add(str(row["scenario_id"]))
            if controller:
                controllers.add(controller)
            if controller == "orius":
                orius_rows += 1
                if _boolish(row.get("true_constraint_violated")):
                    orius_true_violations += 1
    return {
        "trace_rows_scanned": rows,
        "validation_surfaces": sorted(surfaces),
        "scenario_count_scanned": len(scenarios),
        "controllers_scanned": sorted(controllers),
        "orius_trace_rows_scanned": orius_rows,
        "orius_true_constraint_violations": orius_true_violations,
    }


def validate(
    *,
    summary_path: Path,
    traces_path: Path,
    source_manifest_path: Path,
    manifest_out: Path,
    min_runtime_rows: int,
    min_trace_rows: int,
) -> dict[str, Any]:
    findings: list[str] = []
    if not summary_path.exists():
        findings.append(f"missing nuPlan summary: {summary_path}")
    if not traces_path.exists():
        findings.append(f"missing nuPlan traces: {traces_path}")
    if not source_manifest_path.exists():
        findings.append(f"missing nuPlan source manifest: {source_manifest_path}")

    summary, summary_findings = _read_summary(summary_path) if summary_path.exists() else ({}, [])
    findings.extend(summary_findings)
    trace_stats = (
        _trace_stats(traces_path)
        if traces_path.exists()
        else {
            "trace_rows_scanned": 0,
            "validation_surfaces": [],
            "scenario_count_scanned": 0,
            "controllers_scanned": [],
            "orius_trace_rows_scanned": 0,
            "orius_true_constraint_violations": 0,
        }
    )
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8")) if source_manifest_path.exists() else {}
    )

    surface = str(summary.get("validation_surface", ""))
    status = str(summary.get("status", ""))
    source_dataset = str(summary.get("source_dataset", ""))
    claim_boundary = str(summary.get("claim_boundary", "")).lower()
    runtime_rows = int(_floatish(summary.get("orius_runtime_rows")))
    trace_rows = int(trace_stats["trace_rows_scanned"])
    orius_tsvr = _floatish(summary.get("orius_tsvr"), 1.0)
    certificate_rate = _floatish(summary.get("certificate_valid_rate"))
    postcondition_rate = _floatish(summary.get("domain_postcondition_pass_rate"))

    if surface != NUPLAN_SURFACE:
        findings.append(f"AV full-freeze surface is not {NUPLAN_SURFACE}: {surface}")
    if status != NUPLAN_STATUS:
        findings.append(f"unexpected nuPlan full-freeze status: {status}")
    if "nuplan" not in source_dataset.lower():
        findings.append(f"AV full-freeze source dataset is not nuPlan: {source_dataset}")
    if runtime_rows < min_runtime_rows:
        findings.append(
            f"nuPlan ORIUS runtime rows below full-freeze floor: {runtime_rows} < {min_runtime_rows}"
        )
    if trace_rows < min_trace_rows:
        findings.append(f"nuPlan trace rows below full-freeze floor: {trace_rows} < {min_trace_rows}")
    if orius_tsvr > MAX_NUPLAN_RUNTIME_TSVR:
        findings.append(f"nuPlan ORIUS TSVR exceeds promoted empirical epsilon: {orius_tsvr}")
    if certificate_rate < MIN_NUPLAN_PASS_RATE:
        findings.append(f"nuPlan certificate-valid rate below promoted floor: {certificate_rate}")
    if postcondition_rate < MIN_NUPLAN_PASS_RATE:
        findings.append(f"nuPlan domain postcondition pass rate below promoted floor: {postcondition_rate}")
    if _boolish(summary.get("carla_completed")):
        findings.append("nuPlan full-freeze summary claims completed CARLA simulation")
    if _boolish(summary.get("road_deployed")):
        findings.append("nuPlan full-freeze summary claims road deployment")
    if _boolish(summary.get("full_autonomous_driving_closure_claimed")):
        findings.append("nuPlan full-freeze summary claims full autonomous-driving field closure")
    for phrase in (
        "does not claim completed carla",
        "road deployment",
        "full autonomous-driving field closure",
    ):
        if phrase not in claim_boundary:
            findings.append(f"nuPlan claim boundary missing phrase: {phrase}")
    if trace_stats["validation_surfaces"] and trace_stats["validation_surfaces"] != [NUPLAN_SURFACE]:
        findings.append(f"unexpected nuPlan trace surfaces: {trace_stats['validation_surfaces']}")
    if "orius" not in trace_stats["controllers_scanned"]:
        findings.append("nuPlan traces do not contain ORIUS controller rows")
    if trace_stats["orius_trace_rows_scanned"]:
        trace_tsvr = int(trace_stats["orius_true_constraint_violations"]) / float(
            trace_stats["orius_trace_rows_scanned"]
        )
        if trace_tsvr > MAX_NUPLAN_RUNTIME_TSVR:
            findings.append("nuPlan ORIUS trace TSVR exceeds promoted empirical epsilon")
    if source_manifest.get("status") != NUPLAN_STATUS:
        findings.append(
            f"nuPlan source manifest status is not {NUPLAN_STATUS}: {source_manifest.get('status')}"
        )

    passed = not findings
    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "nuplan_full_freeze_pass" if passed else "nuplan_full_freeze_failed",
        "pass": passed,
        "summary": str(summary_path),
        "traces": str(traces_path),
        "source_manifest": str(source_manifest_path),
        "summary_row": summary,
        "trace_stats": trace_stats,
        "source_dataset": source_dataset,
        "validation_surface": surface,
        "primary_target": NUPLAN_SURFACE,
        "orius_runtime_rows": runtime_rows,
        "trace_rows": trace_rows,
        "orius_tsvr": orius_tsvr,
        "certificate_valid_rate": certificate_rate,
        "domain_postcondition_pass_rate": postcondition_rate,
        "min_runtime_rows": min_runtime_rows,
        "min_trace_rows": min_trace_rows,
        "findings": findings,
        "claim_boundary": (
            "Full AV freeze uses bounded nuPlan replay/surrogate runtime-contract evidence only; "
            "it does not claim completed CARLA simulation, road deployment, or full autonomous-driving field closure."
        ),
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate bounded nuPlan full-freeze artifacts.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--min-runtime-rows", type=int, default=1_531_104)
    parser.add_argument("--min-trace-rows", type=int, default=12_248_832)
    args = parser.parse_args()
    manifest = validate(
        summary_path=args.summary,
        traces_path=args.traces,
        source_manifest_path=args.manifest,
        manifest_out=args.manifest_out,
        min_runtime_rows=args.min_runtime_rows,
        min_trace_rows=args.min_trace_rows,
    )
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
