#!/usr/bin/env python3
"""Stage AV and healthcare dataset expansion for the next freeze run.

This script is intentionally conservative:
- It inventories existing AV and healthcare data.
- It writes a next-run acquisition manifest and command sheet.
- It can execute the open PhysioNet healthcare expansion only when explicitly
  requested.
- It does not attempt to bypass provider/license-gated AV downloads.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "reports" / "data_expansion" / "next_run"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def _active_freeze() -> dict[str, Any]:
    result = _run(
        [
            "ps",
            "-axo",
            "pid,ppid,etime,%cpu,%mem,command",
        ]
    )
    lines: list[str] = []
    for line in result.stdout.splitlines():
        lowered = line.lower()
        # Codex/Desktop git staging may include script names as file paths in
        # `git add` arguments. That is not an active training/freeze process.
        if " git add" in lowered or "/git add" in lowered:
            continue
        is_python = "python" in lowered or ".venv/bin/python" in lowered
        is_freeze = "scripts/run_three_domain_offline_freeze.py" in line
        is_training = "scripts/train_dataset.py" in line and "--dataset" in line
        is_legacy_training = "orius.forecasting.train" in line
        if is_python and (is_freeze or is_training or is_legacy_training):
            lines.append(line)
    return {"active": bool(lines), "processes": lines}


def _du(path: Path) -> str | None:
    if not path.exists():
        return None
    result = _run(["du", "-sh", str(path)])
    if result.returncode != 0:
        return None
    return result.stdout.strip().split()[0]


def _count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob(pattern) if p.is_file() and not p.name.startswith("._"))


def _count_parquet_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    code = (
        "import pandas as pd, sys; "
        "p=sys.argv[1]; "
        "print(len(pd.read_parquet(p)))"
    )
    result = _run([str(REPO_ROOT / ".venv" / "bin" / "python"), "-c", code, str(path)])
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    commands = payload["commands"]
    av = payload["inventory"]["av"]
    hc = payload["inventory"]["healthcare"]
    lines = [
        "# Next-Run Dataset Expansion",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Current Inventory",
        "",
        f"- AV total local data: `{av['total_size']}`",
        f"- AV raw Waymo data: `{av['waymo_raw_size']}` with `{av['waymo_validation_shards_present']}` validation shards present",
        f"- AV processed training rows: `{av['processed_training_rows']}`",
        f"- Healthcare processed rows: `{hc['processed_rows']}`",
        f"- Healthcare MIMIC manifest rows: `{hc['mimic_manifest_rows']}`",
        "",
        "## Important Boundary",
        "",
        "Do not treat downloaded raw data as validation by itself. After acquisition, rebuild features, train, rebuild runtime artifacts, and rerun the strict gates.",
        "",
        "AV provider downloads are license-gated. This script only inventories and normalizes local raw exports; it does not bypass Waymo or Argoverse access controls.",
        "",
        "## Recommended Next Commands",
        "",
        "Run these after the current max freeze completes, unless you intentionally accept slower training.",
        "",
        "### Healthcare: Expand Open PhysioNet MIMIC Numerics",
        "",
        "```zsh",
        commands["healthcare_mimic_expand"],
        "```",
        "",
        "### Healthcare: Refresh BIDMC Full CSV Corpus",
        "",
        "```zsh",
        commands["healthcare_bidmc_full"],
        "```",
        "",
        "### AV: After Manually Adding More Waymo Shards",
        "",
        "Put additional official Waymo Motion validation shards under:",
        "",
        "`data/orius_av/raw/waymo_motion/validation/`",
        "",
        "Then run:",
        "",
        "```zsh",
        commands["av_waymo_rebuild"],
        "```",
        "",
        "### AV: After Manually Adding Argoverse 2 Motion Exports",
        "",
        "Put prepared Argoverse 2 Motion CSV/Parquet exports under:",
        "",
        "`data/av/raw/argoverse2_motion/`",
        "",
        "Then run:",
        "",
        "```zsh",
        commands["av_argoverse_normalize"],
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    av_raw = REPO_ROOT / "data" / "orius_av" / "raw" / "waymo_motion"
    av_validation = av_raw / "validation"
    av_processed = REPO_ROOT / "data" / "orius_av" / "av" / "processed"
    av_features = av_processed / "features.parquet"
    hc_processed = REPO_ROOT / "data" / "healthcare" / "processed"
    hc_features = hc_processed / "features.parquet"
    hc_mimic_manifest = REPO_ROOT / "data" / "healthcare" / "mimic3" / "processed" / "mimic3_manifest.json"

    mimic_rows = None
    if hc_mimic_manifest.exists():
        try:
            mimic_rows = json.loads(hc_mimic_manifest.read_text(encoding="utf-8")).get("total_rows")
        except json.JSONDecodeError:
            mimic_rows = None

    commands = {
        "healthcare_mimic_expand": (
            f".venv/bin/python scripts/build_physionet_healthcare_bridge.py "
            f"--n-patients {args.healthcare_patients} "
            f"--out-dir data/healthcare/mimic3_next_run/processed"
        ),
        "healthcare_bidmc_full": (
            ".venv/bin/python scripts/download_healthcare_datasets.py "
            "--source bidmc --download-signals --download-breaths "
            "--out data/healthcare/processed/healthcare_bidmc_next_run_orius.csv"
        ),
        "av_waymo_rebuild": (
            ".venv/bin/python scripts/build_waymo_av_validation_surface.py "
            "--raw-dir data/orius_av/raw/waymo_motion/validation "
            "--out-dir data/orius_av/av/processed_next_run "
            f"--max-shards {args.waymo_max_shards} "
            f"--max-scenarios {args.waymo_max_scenarios}\n"
            ".venv/bin/python scripts/run_waymo_av_dry_run.py "
            "--raw-dir data/orius_av/raw/waymo_motion/validation "
            "--processed-dir data/orius_av/av/processed_next_run "
            f"--subset-size {args.waymo_subset_size} "
            "--reports-dir reports/orius_av/next_run "
            "--models-dir artifacts/models_orius_av_next_run "
            "--uncertainty-dir artifacts/uncertainty/orius_av_next_run"
        ),
        "av_argoverse_normalize": (
            ".venv/bin/python scripts/download_av_datasets.py "
            "--source argoverse2_motion "
            "--out data/av/processed/argoverse2_motion_orius_next_run.csv"
        ),
    }

    return {
        "generated_at_utc": _utc_now(),
        "intent": "next_run_dataset_expansion",
        "downloads_started_by_default": False,
        "active_freeze": _active_freeze(),
        "inventory": {
            "av": {
                "total_size": _du(REPO_ROOT / "data" / "orius_av"),
                "waymo_raw_size": _du(av_raw),
                "processed_size": _du(av_processed),
                "waymo_validation_shards_present": _count_files(av_validation, "validation_tfexample.tfrecord-*"),
                "processed_training_rows": _count_parquet_rows(av_features),
            },
            "healthcare": {
                "processed_size": _du(hc_processed),
                "processed_rows": _count_parquet_rows(hc_features),
                "mimic_manifest_rows": mimic_rows,
            },
        },
        "targets": {
            "healthcare_patients_next_run": args.healthcare_patients,
            "waymo_max_shards_next_run": args.waymo_max_shards,
            "waymo_max_scenarios_next_run": args.waymo_max_scenarios,
            "waymo_subset_size_next_run": args.waymo_subset_size,
        },
        "commands": commands,
        "claim_boundary": [
            "downloaded data alone is not evidence of deployment readiness",
            "AV provider-license datasets must be acquired through official terms",
            "healthcare PhysioNet data remains offline retrospective validation until prospective/site validation is run",
        ],
    }


def maybe_execute_healthcare(args: argparse.Namespace, manifest: dict[str, Any]) -> int:
    if not args.execute_healthcare:
        return 0
    if manifest["active_freeze"]["active"] and not args.allow_during_freeze:
        print("Active freeze detected. Refusing healthcare download without --allow-during-freeze.")
        return 2
    cmd = manifest["commands"]["healthcare_mimic_expand"].split()
    print("Executing:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage next-run AV/healthcare dataset expansion")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--healthcare-patients", type=int, default=200)
    parser.add_argument("--waymo-max-shards", type=int, default=30)
    parser.add_argument("--waymo-max-scenarios", type=int, default=10000)
    parser.add_argument("--waymo-subset-size", type=int, default=8000)
    parser.add_argument("--execute-healthcare", action="store_true")
    parser.add_argument("--allow-during-freeze", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "next_run_dataset_expansion_manifest.json"
    md_path = args.out_dir / "NEXT_RUN_DATASET_EXPANSION.md"
    _safe_write_json(json_path, manifest)
    _write_markdown(md_path, manifest)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if manifest["active_freeze"]["active"]:
        print("Active freeze detected; acquisition commands were staged but not executed.")
    return maybe_execute_healthcare(args, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
