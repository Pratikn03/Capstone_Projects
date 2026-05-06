#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub==0.36.0",
#   "mypy==1.15.0",
#   "pathspec==0.12.1",
#   "pytest==8.4.2",
#   "pyyaml==6.0.3",
#   "ruff==0.9.10",
# ]
# ///
"""Run ORIUS sanitized-code quality gates inside a Hugging Face Job."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

COMMANDS = [
    ["python", "-m", "compileall", "-q", "src", "scripts", "tests"],
    ["python", "-m", "ruff", "check", "src", "scripts", "tests", "services"],
    ["python", "-m", "ruff", "format", "--check", "src", "scripts", "tests", "services"],
    ["python", "-m", "mypy", "src/orius/dc3s", "src/orius/cpsbench_iot", "src/orius/universal_theory"],
    ["python", "scripts/audit_code_health.py", "--config", "configs/publish_audit.yaml"],
    ["python", "scripts/validate_generated_artifact_policy.py"],
    [
        "python",
        "-m",
        "pytest",
        "-q",
        "tests/test_audit_code_health.py",
        "tests/test_clean_artifact_release.py",
        "tests/test_nuplan_av_surface.py",
        "tests/test_battery_av_pipeline.py",
        "tests/test_av_waymo_dry_run.py",
        "tests/test_validate_paper_claims.py",
        "tests/test_camera_ready_figure_lineage.py",
    ],
]


def _run(command: list[str], cwd: Path) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-8000:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HF dataset repo containing the sanitized bundle.")
    parser.add_argument(
        "--bundle-prefix", required=True, help="Path inside dataset, e.g. bundles/orius-code-quality-..."
    )
    parser.add_argument(
        "--upload-report", action="store_true", help="Upload quality_report.json back to the dataset."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = Path(snapshot_download(repo_id=args.dataset, repo_type="dataset"))
    bundle_dir = snapshot / args.bundle_prefix
    if not bundle_dir.is_dir():
        print(f"missing bundle directory: {bundle_dir}", file=sys.stderr)
        return 1

    results = [_run(command, bundle_dir) for command in COMMANDS]
    passed = all(result["returncode"] == 0 for result in results)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": args.dataset,
        "bundle_prefix": args.bundle_prefix,
        "pass": passed,
        "results": results,
    }
    report_path = bundle_dir / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.upload_report:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(report_path),
            repo_id=args.dataset,
            repo_type="dataset",
            path_in_repo=f"{args.bundle_prefix}/quality_report.json",
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
