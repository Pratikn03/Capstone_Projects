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

from scripts.build_av_closed_loop_planner_artifacts import build_av_closed_loop_planner_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = (
    REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
)
DEFAULT_SURFACE_DIR = REPO_ROOT / "data" / "orius_av" / "av" / "processed_nuplan_allzip_grouped"
DEFAULT_OUT = REPO_ROOT / "reports" / "predeployment_external_validation"
NUPLAN_SURFACE = "nuplan_allzip_grouped_runtime_replay_surrogate"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
    planner_max_rows: int | None = None,
) -> dict[str, str]:
    runtime_summary_path = runtime_dir / "runtime_summary.csv"
    runtime_traces_path = runtime_dir / "runtime_traces.csv"
    surface_report_path = surface_dir / "nuplan_surface_report.json"
    runtime_report_path = runtime_dir / "runtime_report.json"
    if not runtime_summary_path.exists() or not runtime_traces_path.exists():
        raise FileNotFoundError("nuPlan bounded evidence requires runtime_summary.csv and runtime_traces.csv")

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
    governance_columns = [
        "scenario_id",
        "controller",
        "step_index",
        "certificate_hash",
        "prev_hash",
        "audit_fields_present",
        "audit_fields_required",
        "validity_status",
        "validity_score",
    ]
    read_columns = set(trace_columns) | set(governance_columns)
    summary_path = out_dir / "nuplan_closed_loop_summary.csv"
    traces_path = out_dir / "nuplan_closed_loop_traces.csv"
    manifest_path = out_dir / "nuplan_closed_loop_manifest.json"
    governance_path = runtime_dir / "runtime_governance_summary.csv"
    certos_summary_path = runtime_dir / "certos_verification_summary.json"
    if traces_path.exists():
        traces_path.unlink()

    runtime_trace_rows = 0
    orius_runtime_rows = 0
    certificate_valid_sum = 0
    postcondition_sum = 0
    certificate_rows = 0
    payload_pass_rows = 0
    expiry_present_rows = 0
    expiry_consistent_rows = 0
    chain_link_failures = 0
    last_hash: str | None = None
    first_trace = True
    for chunk in pd.read_csv(
        runtime_traces_path,
        usecols=lambda column: column in read_columns,
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
            if not orius_chunk.empty and "certificate_hash" in orius_chunk:
                cert_hashes = orius_chunk["certificate_hash"].fillna("").astype(str)
                certificate_rows += int((cert_hashes != "").sum())
                if {"audit_fields_present", "audit_fields_required"} <= set(orius_chunk.columns):
                    present = pd.to_numeric(orius_chunk["audit_fields_present"], errors="coerce").fillna(0)
                    required = pd.to_numeric(orius_chunk["audit_fields_required"], errors="coerce").fillna(1)
                    payload_pass_rows += int((present >= required).sum())
                else:
                    payload_pass_rows += int((cert_hashes != "").sum())
                if "validity_status" in orius_chunk:
                    expiry_present = orius_chunk["validity_status"].fillna("").astype(str) != ""
                elif "validity_score" in orius_chunk:
                    expiry_present = orius_chunk["validity_score"].notna()
                else:
                    expiry_present = pd.Series(False, index=orius_chunk.index)
                expiry_present_rows += int(expiry_present.sum())
                if "validity_score" in orius_chunk:
                    validity_score = pd.to_numeric(orius_chunk["validity_score"], errors="coerce")
                    expiry_consistent_rows += int((expiry_present & validity_score.fillna(0).ge(0)).sum())
                else:
                    expiry_consistent_rows += int(expiry_present.sum())
                if {"certificate_hash", "prev_hash"} <= set(orius_chunk.columns):
                    chain_rows = orius_chunk[["certificate_hash", "prev_hash"]]
                    for cert_hash, prev_hash in chain_rows.itertuples(index=False, name=None):
                        cert_hash_text = "" if pd.isna(cert_hash) else str(cert_hash)
                        prev_hash_text = "" if pd.isna(prev_hash) else str(prev_hash)
                        if not cert_hash_text:
                            chain_link_failures += 1
                            continue
                        if last_hash is None:
                            if prev_hash_text:
                                chain_link_failures += 1
                        elif prev_hash_text != last_hash:
                            chain_link_failures += 1
                        last_hash = cert_hash_text
        chunk = chunk.copy()
        chunk["validation_surface"] = NUPLAN_SURFACE
        export_columns = [column for column in trace_columns if column in chunk.columns] + [
            "validation_surface"
        ]
        chunk[export_columns].to_csv(
            traces_path, mode="w" if first_trace else "a", header=first_trace, index=False
        )
        first_trace = False
    certificate_valid_rate = certificate_valid_sum / orius_runtime_rows if orius_runtime_rows else 0.0
    postcondition_rate = postcondition_sum / orius_runtime_rows if orius_runtime_rows else 0.0
    payload_pass_rate = payload_pass_rows / max(certificate_rows, 1)
    expiry_presence_rate = expiry_present_rows / max(certificate_rows, 1)
    expiry_consistency_rate = expiry_consistent_rows / max(expiry_present_rows, 1)
    failure_rows = int(
        (certificate_rows - payload_pass_rows)
        + chain_link_failures
        + max(0, expiry_present_rows - expiry_consistent_rows)
    )
    governance_summary = {
        "certificate_rows": int(certificate_rows),
        "chain_valid": bool(chain_link_failures == 0 and certificate_rows > 0),
        "failure_rows": failure_rows,
        "required_payload_pass_rate": float(payload_pass_rate),
        "expiry_metadata_presence_rate": float(expiry_presence_rate),
        "expiry_consistency_rate": float(expiry_consistency_rate),
        "audit_completeness_rate": float(max(0.0, 1.0 - failure_rows / max(certificate_rows, 1))),
        "chain_link_failures": int(chain_link_failures),
        "chain_scope": "global_orius_certificate_hash_chain",
        "claim_boundary": (
            "Governance is computed from bounded offline nuPlan replay runtime traces; "
            "it is not road deployment or unrestricted AV field evidence."
        ),
    }
    governance_rows = [
        {"metric": "certificate_rows", "value": int(certificate_rows)},
        {"metric": "chain_valid", "value": 1.0 if governance_summary["chain_valid"] else 0.0},
        {"metric": "failure_rows", "value": failure_rows},
        {"metric": "required_payload_pass_rate", "value": payload_pass_rate},
        {"metric": "expiry_metadata_presence_rate", "value": expiry_presence_rate},
        {"metric": "expiry_consistency_rate", "value": expiry_consistency_rate},
        {"metric": "audit_completeness_rate", "value": governance_summary["audit_completeness_rate"]},
        {"metric": "chain_link_failures", "value": int(chain_link_failures)},
    ]
    pd.DataFrame(governance_rows).to_csv(governance_path, index=False)
    certos_summary_path.write_text(json.dumps(governance_summary, indent=2) + "\n", encoding="utf-8")

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
    planner_manifest = build_av_closed_loop_planner_artifacts(
        runtime_dir=runtime_dir,
        out_dir=out_dir / "av_closed_loop_planner",
        max_rows=planner_max_rows,
    )
    manifest = {
        "status": "completed_bounded_replay_not_carla",
        "summary": _display_path(summary_path),
        "traces": _display_path(traces_path),
        "planner_closed_loop_manifest": _display_path(
            out_dir / "av_closed_loop_planner" / "av_closed_loop_manifest.json"
        ),
        "planner_closed_loop_summary": planner_manifest["summary_csv"],
        "planner_utility_safety_frontier": planner_manifest["frontier_csv"],
        "planner_stress_family_summary": planner_manifest["stress_family_csv"],
        "planner_ablation_summary": planner_manifest["ablation_csv"],
        "runtime_summary": _display_path(runtime_summary_path),
        "runtime_traces": _display_path(runtime_traces_path),
        "runtime_governance_summary": _display_path(governance_path),
        "certos_verification_summary": _display_path(certos_summary_path),
        "surface_report": _display_path(surface_report_path),
        "runtime_report": _display_path(runtime_report_path),
        "claim_boundary": summary_row["claim_boundary"],
        "runtime_report_summary": runtime_report,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "summary": str(summary_path),
        "traces": str(traces_path),
        "manifest": str(manifest_path),
        "planner_closed_loop_manifest": str(
            out_dir / "av_closed_loop_planner" / "av_closed_loop_manifest.json"
        ),
        "planner_closed_loop_summary": str(
            out_dir / "av_closed_loop_planner" / "av_planner_closed_loop_summary.csv"
        ),
        "planner_utility_safety_frontier": str(
            out_dir / "av_closed_loop_planner" / "av_utility_safety_frontier.csv"
        ),
        "planner_stress_family_summary": str(
            out_dir / "av_closed_loop_planner" / "av_closed_loop_stress_family_summary.csv"
        ),
        "planner_ablation_summary": str(
            out_dir / "av_closed_loop_planner" / "av_closed_loop_ablation_summary.csv"
        ),
        "runtime_governance_summary": str(governance_path),
        "certos_verification_summary": str(certos_summary_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--surface-dir", type=Path, default=DEFAULT_SURFACE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--planner-max-rows", type=int, default=None)
    args = parser.parse_args()
    artifacts = build_nuplan_closed_loop_artifacts(
        runtime_dir=args.runtime_dir,
        surface_dir=args.surface_dir,
        out_dir=args.out,
        planner_max_rows=args.planner_max_rows,
    )
    print(f"[build_nuplan_closed_loop_artifacts] wrote {artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
