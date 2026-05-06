#!/usr/bin/env python3
"""Upload a sanitized code-quality bundle and launch a Hugging Face QA job.

By default this script is dry-run oriented. Upload and job launch both require
explicit flags because they transmit files to a third-party service.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

try:
    from validate_code_quality_bundle import validate_bundle
except ImportError:
    from scripts.validate_code_quality_bundle import validate_bundle


DEFAULT_DATASET = "pratikn03/orius-code-quality-private"
DEFAULT_FLAVOR = "cpu-upgrade"
DEFAULT_TIMEOUT = "2h"


def _load_manifest(bundle_dir: Path) -> dict[str, object]:
    manifest_path = bundle_dir / "manifest.json"
    return cast(dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8")))


def _job_command(dataset: str, bundle_name: str, flavor: str, timeout: str) -> list[str]:
    return [
        "hf",
        "jobs",
        "uv",
        "run",
        "--flavor",
        flavor,
        "--timeout",
        timeout,
        "--secrets",
        "HF_TOKEN",
        "--",
        "python",
        "scripts/hf_jobs/code_quality_job.py",
        "--dataset",
        dataset,
        "--bundle-prefix",
        f"bundles/{bundle_name}",
        "--upload-report",
    ]


def _upload_bundle(bundle_dir: Path, dataset: str, bundle_name: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for upload") from exc

    api = HfApi()
    api.create_repo(repo_id=dataset, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=dataset,
        repo_type="dataset",
        path_in_repo=f"bundles/{bundle_name}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-dir", type=Path, required=True, help="Validated local code-quality bundle directory."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Private HF dataset repo.")
    parser.add_argument("--flavor", default=DEFAULT_FLAVOR, help="HF Jobs flavor.")
    parser.add_argument("--timeout", default=DEFAULT_TIMEOUT, help="HF Jobs timeout.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned upload/job steps only.")
    parser.add_argument(
        "--confirm-upload", action="store_true", help="Actually upload bundle files to HF dataset."
    )
    parser.add_argument("--launch-job", action="store_true", help="Launch the HF code-quality job.")
    parser.add_argument("--confirm-job", action="store_true", help="Confirm remote HF job launch.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()
    validation = validate_bundle(bundle_dir)
    if not validation["pass"]:
        print("Refusing HF upload/job because bundle validation failed:", file=sys.stderr)
        for error in validation["errors"]:
            print(f"- {error}", file=sys.stderr)
        return 1

    manifest = _load_manifest(bundle_dir)
    bundle_name = str(manifest["bundle_name"])
    command = _job_command(args.dataset, bundle_name, args.flavor, args.timeout)
    planned = {
        "dataset": args.dataset,
        "bundle_dir": str(bundle_dir),
        "bundle_name": bundle_name,
        "file_count": manifest.get("file_count"),
        "upload_path": f"bundles/{bundle_name}",
        "job_command": command,
        "dry_run": args.dry_run,
    }
    print(json.dumps(planned, indent=2, sort_keys=True))

    if args.dry_run:
        return 0
    if args.confirm_upload:
        _upload_bundle(bundle_dir, args.dataset, bundle_name)
        print(f"uploaded sanitized bundle to hf://datasets/{args.dataset}/bundles/{bundle_name}")
    else:
        print("upload skipped; pass --confirm-upload to transmit the sanitized bundle")

    if args.launch_job:
        if not args.confirm_job:
            print("job launch skipped; pass --confirm-job with --launch-job to start HF compute")
            return 0
        completed = subprocess.run(command, check=False)
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
