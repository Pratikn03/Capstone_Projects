#!/usr/bin/env python3
"""Validate the clean-clone reproducibility spine for ORIUS."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_generated_artifact_policy import validate as validate_artifact_policy  # noqa: E402
from scripts.cleanup_appledouble import default_exclude_parts, find_sidecars  # noqa: E402

REQUIRED_PATHS = [
    "requirements.lock.txt",
    "frontend/package-lock.json",
    "reports/publication/three_domain_ml_benchmark.csv",
    "reports/publication/active_theorem_audit.json",
    "reports/publication/certificate_schema_witnesses.csv",
    "scripts/validate_generated_artifact_policy.py",
    "scripts/validate_api_auth_coverage.py",
]
REQUIRED_PYTEST_MARKERS = {"slow", "integration", "local_data", "artifact_mutation", "load"}


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def validate(*, allow_dirty: bool = False) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    warnings: list[str] = []

    if not allow_dirty:
        status = _git(["status", "--short"])
        if status:
            findings.append("git working tree is not clean; commit or discard generated outputs before release")

    for rel in REQUIRED_PATHS:
        if not (REPO_ROOT / rel).exists():
            findings.append(f"required reproducibility path missing: {rel}")

    pytest_ini = (REPO_ROOT / "pytest.ini").read_text(encoding="utf-8") if (REPO_ROOT / "pytest.ini").exists() else ""
    for marker in REQUIRED_PYTEST_MARKERS:
        if f"{marker}:" not in pytest_ini:
            findings.append(f"pytest marker missing: {marker}")
    if "not local_data" not in pytest_ini or "not artifact_mutation" not in pytest_ini:
        findings.append("pytest default addopts must exclude local_data and artifact_mutation tests")

    artifact_findings, artifact_warnings = validate_artifact_policy()
    findings.extend(artifact_findings)
    warnings.extend(artifact_warnings)

    sidecars = find_sidecars(REPO_ROOT, default_exclude_parts(REPO_ROOT))
    if sidecars:
        findings.append(f"AppleDouble sidecar exists: {sidecars[0].relative_to(REPO_ROOT).as_posix()}")

    return findings, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-dirty", action="store_true", help="Do not fail on current uncommitted source edits")
    args = parser.parse_args()

    findings, warnings = validate(allow_dirty=args.allow_dirty)
    for warning in warnings:
        print(f"[validate_reproducibility_95] WARN {warning}")
    if findings:
        print("[validate_reproducibility_95] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_reproducibility_95] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
