#!/usr/bin/env python3
"""Build a Waymo AV dataset readiness report for ORIUS.

The report is intentionally bounded: it inventories local Waymo Motion shards,
processed ORIUS AV surfaces, and runtime replay evidence. It does not claim
closed-loop simulator validation or road deployment.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "orius_av" / "raw" / "waymo_motion" / "validation"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "orius_av" / "av" / "processed_full_corpus"
DEFAULT_CANONICAL_REPORTS = REPO_ROOT / "reports" / "orius_av" / "full_corpus"
DEFAULT_AUDIT_REPORTS = REPO_ROOT / "reports" / "orius_av" / "waymo_dataset_audit"
DEFAULT_OUT = REPO_ROOT / "reports" / "data_expansion" / "waymo_readiness"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _file_count(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.glob(pattern) if p.is_file() and not p.name.startswith("._"))


def _bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.name.startswith("._"))


def _human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _parquet_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    df = pd.read_parquet(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }
    if "scenario_id" in df.columns:
        summary["scenario_count"] = int(df["scenario_id"].nunique())
    if "split" in df.columns:
        summary["split_counts"] = {
            str(k): int(v) for k, v in df["split"].value_counts().sort_index().to_dict().items()
        }
    return summary


def _csv_first_matching_row(path: Path, key: str, value: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get(key) == value:
                return dict(row)
    return {}


def build_report(
    raw_dir: Path, processed_dir: Path, canonical_reports: Path, audit_reports: Path
) -> dict[str, Any]:
    replay_summary = _parquet_summary(processed_dir / "replay_windows.parquet")
    step_summary = _parquet_summary(processed_dir / "step_features.parquet")
    split_summaries = {
        split: _parquet_summary(processed_dir / "splits" / f"{split}.parquet")
        for split in ("train", "val", "calibration", "test")
    }
    canonical_orius = _csv_first_matching_row(
        canonical_reports / "runtime_summary.csv", "controller", "orius"
    )
    audit_orius = _csv_first_matching_row(audit_reports / "runtime_summary.csv", "controller", "orius")
    audit_always_brake = _csv_first_matching_row(
        audit_reports / "runtime_summary.csv", "controller", "always_brake"
    )
    return {
        "generated_at_utc": _utc_now(),
        "dataset": "Waymo Open Motion validation shards",
        "status": "offline_runtime_replay_ready",
        "claim_boundary": [
            "Waymo evidence is offline runtime replay evidence.",
            "This is not closed-loop CARLA/nuPlan simulation.",
            "This is not road deployment or full autonomous-driving field closure.",
        ],
        "raw": {
            "path": str(raw_dir),
            "exists": raw_dir.exists(),
            "validation_shards_present": _file_count(raw_dir, "validation_tfexample.tfrecord-*"),
            "size_bytes": _bytes(raw_dir),
            "size_human": _human_size(_bytes(raw_dir)),
        },
        "processed": {
            "path": str(processed_dir),
            "size_bytes": _bytes(processed_dir),
            "size_human": _human_size(_bytes(processed_dir)),
            "replay_windows": replay_summary,
            "step_features": step_summary,
            "splits": split_summaries,
        },
        "runtime_evidence": {
            "canonical_full_corpus_dir": str(canonical_reports),
            "bounded_audit_dir": str(audit_reports),
            "canonical_orius_row": canonical_orius,
            "bounded_audit_orius_row": audit_orius,
            "bounded_audit_always_brake_row": audit_always_brake,
            "bounded_audit_utility_above_always_brake": (
                float(audit_orius.get("useful_work_total", 0.0))
                > float(audit_always_brake.get("useful_work_total", 0.0))
                if audit_orius and audit_always_brake
                else None
            ),
        },
        "next_steps": [
            "Use more official Waymo shards if available, then rebuild the processed surface.",
            "Run nuPlan/CARLA closed-loop validation before claiming external closed-loop AV evidence.",
            "Keep Waymo rows framed as runtime replay, not deployment validation.",
        ],
    }


def write_outputs(out_dir: Path, payload: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "waymo_dataset_readiness.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        {"metric": "validation_shards_present", "value": payload["raw"]["validation_shards_present"]},
        {"metric": "raw_size", "value": payload["raw"]["size_human"]},
        {"metric": "processed_size", "value": payload["processed"]["size_human"]},
        {"metric": "replay_rows", "value": payload["processed"]["replay_windows"].get("rows")},
        {"metric": "replay_scenarios", "value": payload["processed"]["replay_windows"].get("scenario_count")},
        {"metric": "step_feature_rows", "value": payload["processed"]["step_features"].get("rows")},
        {
            "metric": "step_feature_scenarios",
            "value": payload["processed"]["step_features"].get("scenario_count"),
        },
        {
            "metric": "bounded_audit_orius_tsvr",
            "value": payload["runtime_evidence"]["bounded_audit_orius_row"].get("tsvr"),
        },
        {
            "metric": "bounded_audit_orius_fallback",
            "value": payload["runtime_evidence"]["bounded_audit_orius_row"].get("fallback_activation_rate"),
        },
        {
            "metric": "bounded_audit_utility_above_always_brake",
            "value": payload["runtime_evidence"]["bounded_audit_utility_above_always_brake"],
        },
    ]
    with (out_dir / "waymo_dataset_readiness.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Waymo Dataset Readiness",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Status",
        "",
        f"- Status: `{payload['status']}`",
        f"- Raw shards present: `{payload['raw']['validation_shards_present']}`",
        f"- Raw size: `{payload['raw']['size_human']}`",
        f"- Processed replay rows: `{payload['processed']['replay_windows'].get('rows')}`",
        f"- Processed replay scenarios: `{payload['processed']['replay_windows'].get('scenario_count')}`",
        f"- Step-feature rows: `{payload['processed']['step_features'].get('rows')}`",
        "",
        "## Bounded Runtime Audit",
        "",
        f"- ORIUS TSVR: `{payload['runtime_evidence']['bounded_audit_orius_row'].get('tsvr')}`",
        f"- ORIUS fallback rate: `{payload['runtime_evidence']['bounded_audit_orius_row'].get('fallback_activation_rate')}`",
        f"- Utility above always-brake: `{payload['runtime_evidence']['bounded_audit_utility_above_always_brake']}`",
        "",
        "## Claim Boundary",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["claim_boundary"])
    lines.extend(["", "## Next Steps", ""])
    lines.extend(f"- {item}" for item in payload["next_steps"])
    (out_dir / "waymo_dataset_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for root in (out_dir, Path(payload["runtime_evidence"]["bounded_audit_dir"])):
        if root.exists():
            for sidecar in root.rglob("._*"):
                if sidecar.is_file():
                    sidecar.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Waymo dataset readiness report.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--canonical-reports", type=Path, default=DEFAULT_CANONICAL_REPORTS)
    parser.add_argument("--audit-reports", type=Path, default=DEFAULT_AUDIT_REPORTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = build_report(args.raw_dir, args.processed_dir, args.canonical_reports, args.audit_reports)
    write_outputs(args.out, payload)
    print(f"[waymo_dataset_readiness] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
