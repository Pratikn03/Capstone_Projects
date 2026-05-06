#!/usr/bin/env python3
"""Fail if generated/local-only artifacts are tracked or visible as new files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.classify_repo_artifacts import classify_path

BLOCKED_TRACKED_CATEGORIES = {
    "temporary_ai_codex_artifact",
    "cache_build_output",
    "local_dataset",
    "model_artifact",
    "generated_runtime_artifact",
}
TRACKED_ALLOWLIST = {
    # Compact benchmark bundle intentionally tracked as a reproducibility witness.
    "reports/orius_bench/benchmark_bundle.tar.gz",
}


def _git(args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def validate(include_deleted: bool = False) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    warnings: list[str] = []

    deleted = set(_git(["ls-files", "-d"]))
    for rel in _git(["ls-files"]):
        if rel in TRACKED_ALLOWLIST:
            continue
        category = classify_path(rel)
        if category in BLOCKED_TRACKED_CATEGORIES:
            if rel in deleted and not include_deleted:
                warnings.append(f"tracked generated artifact is deleted and should remain removed: {rel}")
                continue
            findings.append(f"tracked {category}: {rel}")

    for rel in _git(["ls-files", "-o", "--exclude-standard"]):
        category = classify_path(rel)
        if category in {"temporary_ai_codex_artifact", "cache_build_output"}:
            findings.append(f"visible untracked {category}: {rel}")

    return findings, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-deleted", action="store_true", help="Treat deleted tracked generated artifacts as failures"
    )
    args = parser.parse_args()

    findings, warnings = validate(include_deleted=args.include_deleted)
    for warning in warnings:
        print(f"[validate_generated_artifact_policy] WARN {warning}")
    if findings:
        print("[validate_generated_artifact_policy] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_generated_artifact_policy] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
