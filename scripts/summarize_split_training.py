#!/usr/bin/env python3
"""Summarize split CPU training runs into one release-readiness report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = REPO_ROOT / "reports" / "split_training"

DOMAINS = {
    "DE": {
        "label": "Battery / DE",
        "slug": "de",
        "canonical_metrics": REPO_ROOT / "reports" / "week2_metrics.json",
    },
    "AV": {
        "label": "Autonomous Vehicles",
        "slug": "av",
        "canonical_metrics": REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped" / "training_summary.csv",
    },
    "HEALTHCARE": {
        "label": "Healthcare",
        "slug": "healthcare",
        "canonical_metrics": REPO_ROOT / "reports" / "healthcare" / "week2_metrics.json",
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _log_findings(path: Path, completion_markers: tuple[str, ...] = ("✅ Pipeline completed",)) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "completed": False, "errors": ["missing log"]}
    text = path.read_text(encoding="utf-8", errors="replace")
    error_markers = ("Traceback", "KeyboardInterrupt", "❌", "Error:", "Exception")
    errors = [marker for marker in error_markers if marker in text]
    return {
        "exists": True,
        "completed": any(marker in text for marker in completion_markers),
        "acceptance_warning": "Acceptance gates not met" in text,
        "errors": errors,
        "last_lines": text.splitlines()[-20:],
    }


def _completion_evidence(split_dir: Path, slug: str) -> dict[str, Any]:
    """Find recovery logs that prove a split run completed after the primary log stopped."""
    markers = (
        "✅ Pipeline completed",
        "release GBM all complete",
        "Promoted accepted candidate run",
    )
    log_dir = split_dir / "logs"
    if not log_dir.exists():
        return {"exists": False, "completed": False, "errors": ["missing log directory"], "path": None}
    candidates = [
        path
        for path in sorted(log_dir.glob(f"train_{slug}*.log"))
        if path.is_file() and not path.name.startswith("._")
    ]
    best: dict[str, Any] = {"exists": False, "completed": False, "errors": ["missing completion log"], "path": None}
    for path in candidates:
        findings = _log_findings(path, completion_markers=markers)
        findings["path"] = str(path.relative_to(REPO_ROOT))
        if findings["completed"] and not findings["errors"]:
            return findings
        if findings["exists"]:
            best = findings
    return best


def _target_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    targets = metrics.get("targets", {}) if isinstance(metrics.get("targets"), dict) else {}
    rows: list[dict[str, Any]] = []
    for target, payload in targets.items():
        if not isinstance(payload, dict):
            continue
        gbm = payload.get("gbm", {}) if isinstance(payload.get("gbm"), dict) else {}
        uncertainty = gbm.get("uncertainty", {}) if isinstance(gbm.get("uncertainty"), dict) else {}
        tuning = gbm.get("tuning_meta", {}) if isinstance(gbm.get("tuning_meta"), dict) else {}
        rows.append(
            {
                "target": target,
                "rmse": _safe_float(gbm.get("rmse")),
                "mape": _safe_float(gbm.get("mape")),
                "picp_90": _safe_float(uncertainty.get("picp_90")),
                "tuning_trials": tuning.get("n_complete_trials"),
                "retention_decision": payload.get("retention_decision"),
                "retention_reason": payload.get("retention_reason"),
            }
        )
    return rows


def _selection_rows(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = selection.get("targets", [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def _domain_summary(release_id: str, dataset: str, split_dir: Path) -> dict[str, Any]:
    meta = DOMAINS[dataset]
    slug = str(meta["slug"])
    artifacts_root = REPO_ROOT / "artifacts" / "runs" / slug / release_id
    reports_root = REPO_ROOT / "reports" / "runs" / slug / release_id
    registry_dir = artifacts_root / "registry"
    log_path = split_dir / "logs" / f"train_{slug}.log"
    manifest_path = registry_dir / "run_manifest.json"
    metrics_path = reports_root / "week2_metrics.json"
    selection_path = registry_dir / f"tuning_summary_{dataset.lower()}.json"
    promotion_path = registry_dir / "promotion_record.json"
    model_files = [
        path
        for path in (artifacts_root / "models").rglob("*")
        if path.is_file() and not path.name.startswith("._")
    ] if (artifacts_root / "models").exists() else []
    uncertainty_files = [
        path
        for path in (artifacts_root / "uncertainty").rglob("*")
        if path.is_file() and not path.name.startswith("._")
    ] if (artifacts_root / "uncertainty").exists() else []
    metrics = _load_json(metrics_path)
    selection = _load_json(selection_path)
    manifest = _load_json(manifest_path)
    log = _log_findings(log_path)
    completion_evidence = _completion_evidence(split_dir, slug)
    missing = []
    for label, path in (
        ("run manifest", manifest_path),
        ("candidate metrics", metrics_path),
        ("selection summary", selection_path),
        ("canonical promotion record", promotion_path),
    ):
        if not path.exists():
            missing.append(label)
    if not model_files:
        missing.append("model artifacts")
    if not uncertainty_files:
        missing.append("uncertainty artifacts")
    artifact_recovered = (
        manifest.get("accepted") is True
        and manifest.get("promoted_at")
        and completion_evidence["completed"]
        and not completion_evidence["errors"]
        and not missing
    )
    status = "complete" if (log["completed"] and not log["errors"]) or artifact_recovered else "incomplete"
    if not log["completed"] and not log["errors"]:
        status = "running_or_pending"
    if artifact_recovered:
        status = "complete"
    return {
        "dataset": dataset,
        "domain": meta["label"],
        "status": status,
        "log": str(log_path.relative_to(REPO_ROOT)),
        "log_completed": log["completed"],
        "completion_evidence": completion_evidence,
        "artifact_recovered": bool(artifact_recovered),
        "log_errors": log["errors"],
        "acceptance_warning": log["acceptance_warning"],
        "accepted": manifest.get("accepted"),
        "promoted_at": manifest.get("promoted_at"),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)) if manifest_path.exists() else None,
        "metrics": str(metrics_path.relative_to(REPO_ROOT)) if metrics_path.exists() else None,
        "selection": str(selection_path.relative_to(REPO_ROOT)) if selection_path.exists() else None,
        "promotion_record": str(promotion_path.relative_to(REPO_ROOT)) if promotion_path.exists() else None,
        "model_file_count": len(model_files),
        "uncertainty_file_count": len(uncertainty_files),
        "targets": _target_rows(metrics),
        "selection_targets": _selection_rows(selection),
        "missing": missing,
    }


def _nuplan_summary(split_dir: Path) -> dict[str, Any]:
    path = split_dir / "nuplan_full_av_gate.json"
    payload = _load_json(path)
    row = payload.get("summary_row", {}) if isinstance(payload.get("summary_row"), dict) else {}
    return {
        "path": str(path.relative_to(REPO_ROOT)) if path.exists() else None,
        "pass": bool(payload.get("pass", False)),
        "status": payload.get("status"),
        "validation_surface": payload.get("validation_surface") or row.get("validation_surface"),
        "source_dataset": payload.get("source_dataset") or row.get("source_dataset"),
        "orius_tsvr": _safe_float(payload.get("orius_tsvr") or row.get("orius_tsvr")),
        "baseline_tsvr": _safe_float(row.get("baseline_tsvr")),
        "fallback_activation_rate": _safe_float(row.get("orius_fallback_activation_rate")),
        "intervention_rate": _safe_float(row.get("orius_intervention_rate")),
        "certificate_valid_rate": _safe_float(payload.get("certificate_valid_rate") or row.get("certificate_valid_rate")),
        "claim_boundary": payload.get("claim_boundary") or row.get("claim_boundary"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "domain",
        "status",
        "accepted",
        "promoted_at",
        "model_file_count",
        "uncertainty_file_count",
        "missing",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "; ".join(row[field]) if field == "missing" else row.get(field) for field in fields})


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Split CPU Training Summary",
        "",
        f"- Release ID: `{summary['release_id']}`",
        f"- Generated: `{summary['generated_at_utc']}`",
        f"- Overall status: **{summary['overall_status']}**",
        "",
        "## Domain Runs",
        "",
        "| Domain | Status | Accepted | Promoted | Models | Uncertainty | Missing |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["domains"]:
        missing = ", ".join(row["missing"]) if row["missing"] else "none"
        lines.append(
            f"| {row['domain']} | {row['status']} | {row['accepted']} | "
            f"{bool(row['promoted_at'])} | {row['model_file_count']} | "
            f"{row['uncertainty_file_count']} | {missing} |"
        )
    lines.extend(["", "## Target Metrics", ""])
    for row in summary["domains"]:
        lines.append(f"### {row['domain']}")
        if not row["targets"]:
            lines.append("- No candidate `week2_metrics.json` yet.")
            lines.append("")
            continue
        lines.append("| Target | RMSE | MAPE | PICP90 | Trials | Retention |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for target in row["targets"]:
            lines.append(
                f"| {target['target']} | {target['rmse']} | {target['mape']} | "
                f"{target['picp_90']} | {target['tuning_trials']} | "
                f"{target.get('retention_decision') or ''} |"
            )
        lines.append("")
    av_gate = summary["av_nuplan_gate"]
    lines.extend(
        [
            "## AV nuPlan Gate",
            "",
            f"- Pass: **{av_gate['pass']}**",
            f"- Surface: `{av_gate['validation_surface']}`",
            f"- ORIUS TSVR: `{av_gate['orius_tsvr']}`",
            f"- Baseline TSVR: `{av_gate['baseline_tsvr']}`",
            f"- Fallback activation: `{av_gate['fallback_activation_rate']}`",
            f"- Certificate valid rate: `{av_gate['certificate_valid_rate']}`",
            f"- Claim boundary: {av_gate['claim_boundary']}",
            "",
            "## Missing / Next Gates",
            "",
        ]
    )
    for item in summary["missing_items"]:
        lines.append(f"- {item}")
    if not summary["missing_items"]:
        lines.append("- None recorded by this split-run summarizer.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(release_id: str) -> dict[str, Any]:
    split_dir = SPLIT_ROOT / release_id
    domains = [_domain_summary(release_id, dataset, split_dir) for dataset in DOMAINS]
    av_gate = _nuplan_summary(split_dir)
    missing_items: list[str] = []
    for row in domains:
        for item in row["missing"]:
            missing_items.append(f"{row['dataset']}: {item}")
        if row["status"] != "complete":
            missing_items.append(f"{row['dataset']}: training log status is {row['status']}")
        if row["accepted"] is not True:
            missing_items.append(f"{row['dataset']}: acceptance/promotion not complete")
    if not av_gate["pass"]:
        missing_items.append("AV: nuPlan full gate did not pass")
    overall_status = "pass" if not missing_items else "blocked"
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "release_id": release_id,
        "overall_status": overall_status,
        "domains": domains,
        "av_nuplan_gate": av_gate,
        "missing_items": sorted(set(missing_items)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", default=None)
    args = parser.parse_args()
    release_id = args.release_id
    if release_id is None:
        latest = SPLIT_ROOT / "latest_release_id.txt"
        if not latest.exists():
            raise SystemExit("missing --release-id and latest_release_id.txt")
        release_id = latest.read_text(encoding="utf-8").strip()
    summary = summarize(release_id)
    out_dir = SPLIT_ROOT / release_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "combined_training_summary.json"
    md_path = out_dir / "combined_training_summary.md"
    csv_path = out_dir / "combined_training_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    _write_csv(csv_path, summary["domains"])
    print(json.dumps({"summary": str(json_path), "markdown": str(md_path), "csv": str(csv_path), "status": summary["overall_status"]}, indent=2))
    return 0 if summary["overall_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
