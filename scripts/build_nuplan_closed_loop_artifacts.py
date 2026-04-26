#!/usr/bin/env python3
"""Promote bounded nuPlan replay output into next-tier AV evidence files.

This intentionally records a bounded offline replay/surrogate surface. It does
not claim CARLA completion, road deployment, or full autonomous-driving field closure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
DEFAULT_SURFACE_DIR = REPO_ROOT / "data" / "orius_av" / "av" / "processed_nuplan_allzip_grouped"
DEFAULT_OUT = REPO_ROOT / "reports" / "predeployment_external_validation"
NUPLAN_SURFACE = "nuplan_allzip_grouped_runtime_replay_surrogate"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _controller_row(summary: pd.DataFrame, controller: str) -> dict[str, Any]:
    if "controller" in summary.columns:
        rows = summary[summary["controller"].astype(str) == controller]
        if not rows.empty:
            return rows.iloc[0].to_dict()
    return {}


def build_nuplan_closed_loop_artifacts(
    *,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
    surface_dir: Path = DEFAULT_SURFACE_DIR,
    out_dir: Path = DEFAULT_OUT,
) -> dict[str, str]:
    runtime_summary_path = runtime_dir / "runtime_summary.csv"
    runtime_traces_path = runtime_dir / "runtime_traces.csv"
    surface_report_path = surface_dir / "nuplan_surface_report.json"
    runtime_report_path = runtime_dir / "runtime_report.json"
    if not runtime_summary_path.exists() or not runtime_traces_path.exists():
        raise FileNotFoundError(
            "nuPlan bounded evidence requires runtime_summary.csv and runtime_traces.csv"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_summary = pd.read_csv(runtime_summary_path)
    surface_report = _load_json(surface_report_path)
    runtime_report = _load_json(runtime_report_path)
    orius = _controller_row(runtime_summary, "orius")
    baseline = _controller_row(runtime_summary, "baseline")
    trace_columns = [
        "trace_id",
        "scenario_id",
        "shard_id",
        "step_index",
        "fault_family",
        "controller",
        "intervened",
        "fallback_used",
        "certificate_valid",
        "true_constraint_violated",
        "domain_postcondition_passed",
        "t11_status",
        "certificate_hash",
    ]
    summary_path = out_dir / "nuplan_closed_loop_summary.csv"
    traces_path = out_dir / "nuplan_closed_loop_traces.csv"
    manifest_path = out_dir / "nuplan_closed_loop_manifest.json"
    if traces_path.exists():
        traces_path.unlink()

    runtime_trace_rows = 0
    orius_runtime_rows = 0
    certificate_valid_sum = 0
    postcondition_sum = 0
    first_trace = True
    for chunk in pd.read_csv(
        runtime_traces_path,
        usecols=lambda column: column in set(trace_columns),
        chunksize=200_000,
        low_memory=False,
    ):
        runtime_trace_rows += int(len(chunk))
        if "controller" in chunk:
            orius_mask = chunk["controller"].astype(str) == "orius"
            orius_chunk = chunk.loc[orius_mask]
            orius_runtime_rows += int(len(orius_chunk))
            if "certificate_valid" in orius_chunk:
                certificate_valid_sum += int(orius_chunk["certificate_valid"].astype(bool).sum())
            if "domain_postcondition_passed" in orius_chunk:
                postcondition_sum += int(orius_chunk["domain_postcondition_passed"].astype(bool).sum())
        chunk = chunk.copy()
        chunk["validation_surface"] = NUPLAN_SURFACE
        chunk.to_csv(traces_path, mode="w" if first_trace else "a", header=first_trace, index=False)
        first_trace = False
    certificate_valid_rate = certificate_valid_sum / orius_runtime_rows if orius_runtime_rows else 0.0
    postcondition_rate = postcondition_sum / orius_runtime_rows if orius_runtime_rows else 0.0

    summary_row = {
        "validation_surface": NUPLAN_SURFACE,
        "status": "completed_bounded_replay_not_carla",
        "source_dataset": surface_report.get("source_dataset", "nuplan_singapore"),
        "evidence_level": "offline_runtime_replay_surrogate",
        "db_count": int(surface_report.get("db_count", 0) or 0),
        "scenario_count": int(surface_report.get("scenario_count", 0) or 0),
        "runtime_trace_rows": runtime_trace_rows,
        "orius_runtime_rows": orius_runtime_rows,
        "orius_tsvr": float(orius.get("tsvr", 0.0) or 0.0),
        "baseline_tsvr": float(baseline.get("tsvr", 0.0) or 0.0),
        "orius_intervention_rate": float(orius.get("intervention_rate", 0.0) or 0.0),
        "orius_fallback_activation_rate": float(orius.get("fallback_activation_rate", 0.0) or 0.0),
        "certificate_valid_rate": certificate_valid_rate,
        "domain_postcondition_pass_rate": postcondition_rate,
        "carla_completed": False,
        "road_deployed": False,
        "full_autonomous_driving_closure_claimed": False,
        "claim_boundary": (
            "Completed all-zip grouped nuPlan runtime replay/surrogate evidence only; "
            "does not claim completed CARLA simulation, road deployment, or full autonomous-driving field closure."
        ),
    }

    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)
    manifest = {
        "status": "completed_bounded_replay_not_carla",
        "summary": str(summary_path.relative_to(REPO_ROOT)),
        "traces": str(traces_path.relative_to(REPO_ROOT)),
        "runtime_summary": str(runtime_summary_path.relative_to(REPO_ROOT)),
        "runtime_traces": str(runtime_traces_path.relative_to(REPO_ROOT)),
        "surface_report": str(surface_report_path.relative_to(REPO_ROOT)),
        "runtime_report": str(runtime_report_path.relative_to(REPO_ROOT)),
        "claim_boundary": summary_row["claim_boundary"],
        "runtime_report_summary": runtime_report,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "summary": str(summary_path),
        "traces": str(traces_path),
        "manifest": str(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    artifacts = build_nuplan_closed_loop_artifacts(
        runtime_dir=args.runtime_dir,
        surface_dir=args.surface_dir,
        out_dir=args.out,
    )
    print(f"[build_nuplan_closed_loop_artifacts] wrote {artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
