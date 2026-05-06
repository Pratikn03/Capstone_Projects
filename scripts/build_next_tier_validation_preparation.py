#!/usr/bin/env python3
"""Build prepared-but-not-completed next-tier validation manifests."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "reports" / "predeployment_external_validation"
AV_CFG = REPO_ROOT / "configs" / "validation" / "nuplan_carla_predeployment.yml"
HC_CFG = REPO_ROOT / "configs" / "validation" / "healthcare_heldout_runtime.yml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return payload if isinstance(payload, dict) else {}


def _path_status(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        rows.append(
            {
                "path": raw,
                "exists": path.exists(),
                "file_count": sum(1 for child in path.rglob("*") if child.is_file()) if path.is_dir() else 0,
                "size_bytes": path.stat().st_size if path.is_file() else None,
            }
        )
    return rows


def _all_artifacts_exist(paths: list[str]) -> bool:
    return all((REPO_ROOT / path).exists() for path in paths)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_next_tier_validation_preparation(*, out_dir: Path = DEFAULT_OUT) -> dict[str, Any]:
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    av_cfg = _load_yaml(AV_CFG)
    hc_cfg = _load_yaml(HC_CFG)

    av_required = list(av_cfg.get("completion_required_artifacts", []))
    nuplan_required = list(av_cfg.get("nuplan_completion_artifacts", []))
    carla_required = list(av_cfg.get("carla_completion_artifacts", []))
    hc_required = list(hc_cfg.get("required_runtime_artifacts_for_claim", []))
    av_completed = _all_artifacts_exist(av_required)
    nuplan_completed = _all_artifacts_exist(nuplan_required) if nuplan_required else False
    carla_completed = _all_artifacts_exist(carla_required) if carla_required else False
    hc_completed = _all_artifacts_exist(hc_required)

    av_manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "domain": "Autonomous Vehicles",
        "status": "completed" if av_completed else "prepared_not_completed",
        "completed_evidence": av_completed,
        "nuplan_completed_evidence": nuplan_completed,
        "carla_completed_evidence": carla_completed,
        "claim_allowed": av_completed,
        "claim_boundary": av_cfg.get("claim_boundary", ""),
        "nuplan_dataset_discovery": _path_status(list(av_cfg.get("nuplan", {}).get("expected_roots", []))),
        "carla_dataset_discovery": _path_status(list(av_cfg.get("carla", {}).get("expected_roots", []))),
        "required_splits": list(av_cfg.get("nuplan", {}).get("required_splits", [])),
        "adapter_status": {
            "nuplan": av_cfg.get("nuplan", {}).get("adapter_status", "todo"),
            "carla": av_cfg.get("carla", {}).get("adapter_status", "todo"),
        },
        "completion_required_artifacts": av_required,
        "nuplan_completion_artifacts": nuplan_required,
        "carla_completion_artifacts": carla_required,
    }

    split_manifest_path = Path(str(hc_cfg.get("split_manifest", "")))
    if not split_manifest_path.is_absolute():
        split_manifest_path = REPO_ROOT / split_manifest_path
    split_manifest = {}
    if split_manifest_path.exists():
        try:
            split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            split_manifest = {}

    hc_manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "domain": "Medical and Healthcare Monitoring",
        "status": "completed" if hc_completed else "prepared_not_completed",
        "completed_evidence": hc_completed,
        "heldout_claim_allowed": hc_completed,
        "claim_boundary": hc_cfg.get("claim_boundary", ""),
        "prepared_split_dir": hc_cfg.get("prepared_split_dir"),
        "split_manifest_exists": split_manifest_path.exists(),
        "split_manifest": split_manifest,
        "required_runtime_artifacts_for_claim": hc_required,
        "completion_rule": hc_cfg.get("completion_rule", {}),
    }

    av_path = out_dir / "nuplan_carla_preparation_manifest.json"
    hc_path = out_dir / "healthcare_heldout_runtime_preparation_manifest.json"
    av_path.write_text(json.dumps(av_manifest, indent=2) + "\n", encoding="utf-8")
    hc_path.write_text(json.dumps(hc_manifest, indent=2) + "\n", encoding="utf-8")

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "prepared_not_completed",
        "claim_boundary": (
            "These manifests prepare next-tier validation only. They do not promote "
            "nuPlan/CARLA closed-loop evidence or healthcare held-out runtime evidence "
            "until the required runtime artifacts exist."
        ),
        "manifests": {
            "av": _display_path(av_path),
            "healthcare": _display_path(hc_path),
        },
        "completed": {
            "av_nuplan_carla": av_completed,
            "av_nuplan_allzip_grouped": nuplan_completed,
            "av_carla": carla_completed,
            "healthcare_heldout_runtime": hc_completed,
        },
    }
    (out_dir / "next_tier_validation_preparation_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "next_tier_validation_preparation.md").write_text(
        "\n".join(
            [
                "# Next-Tier Validation Preparation",
                "",
                summary["claim_boundary"],
                "",
                f"- AV nuPlan/CARLA completed: `{av_completed}`",
                f"- Healthcare held-out runtime completed: `{hc_completed}`",
                "- Current status: prepared, not promoted as completed evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summary = build_next_tier_validation_preparation(out_dir=args.out)
    print(f"[next_tier_validation_preparation] status={summary['status']} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
