#!/usr/bin/env python3
"""Validate the clean-clone reproducibility spine for ORIUS."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cleanup_appledouble import default_exclude_parts, find_sidecars
from scripts.validate_generated_artifact_policy import validate as validate_artifact_policy

REQUIRED_PATHS = [
    "requirements.lock.txt",
    "frontend/package-lock.json",
    "reports/publication/three_domain_ml_benchmark.csv",
    "reports/publication/active_theorem_audit.json",
    "reports/publication/certificate_schema_witnesses.csv",
    "reports/publication/runtime_release_contract_witnesses.csv",
    "reports/publication/runtime_release_contract_witnesses.json",
    "scripts/validate_generated_artifact_policy.py",
    "scripts/validate_api_auth_coverage.py",
    "scripts/validate_runtime_release_contract.py",
    "configs/script_registry.yml",
]
REQUIRED_PYTEST_MARKERS = {"slow", "integration", "local_data", "artifact_mutation", "load"}
SCRIPT_REGISTRY = REPO_ROOT / "configs" / "script_registry.yml"
ALLOWED_SCRIPT_REGISTRY_STATUSES = {"manual", "local_only", "archived", "entrypoint"}


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _git_path_list(args: list[str]) -> list[str]:
    output = _git(args)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _load_script_registry(path: Path = SCRIPT_REGISTRY) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = payload.get("scripts", payload) if isinstance(payload, dict) else {}
    normalized: dict[str, dict[str, str]] = {}
    if isinstance(entries, list):
        normalized.update(
            {
            str(entry.get("path", "")).strip(): {
                "status": str(entry.get("status", "")).strip(),
                "reason": str(entry.get("reason", "")).strip(),
            }
            for entry in entries
            if isinstance(entry, dict) and str(entry.get("path", "")).strip()
            }
        )
    if isinstance(entries, dict):
        for path_key, value in entries.items():
            if isinstance(value, dict):
                normalized[str(path_key)] = {
                    "status": str(value.get("status", "")).strip(),
                    "reason": str(value.get("reason", "")).strip(),
                }
            else:
                normalized[str(path_key)] = {"status": str(value).strip(), "reason": ""}
    legacy = payload.get("legacy_manual_scripts", {}) if isinstance(payload, dict) else {}
    if isinstance(legacy, dict):
        reason = str(
            legacy.get(
                "reason",
                "Historical script retained outside the default release path; requires explicit invocation.",
            )
        ).strip()
        for script in legacy.get("paths", []) or []:
            script_path = str(script).strip()
            if script_path and script_path not in normalized:
                normalized[script_path] = {"status": "manual", "reason": reason}
    return normalized


def _reference_texts() -> dict[str, str]:
    candidates = set(_git_path_list(["ls-files"]))
    candidates.update(_git_path_list(["ls-files", "-o", "--exclude-standard"]))
    reference_texts: dict[str, str] = {}
    allowed_prefixes = ("Makefile", "docs/", "tests/", "scripts/", "configs/", "reports/publication/README.md")
    for rel in sorted(candidates):
        if rel.endswith(".pyc") or "__pycache__" in rel:
            continue
        if rel == "Makefile" or rel.startswith(allowed_prefixes[1:]) or rel.startswith("reports/publication/README.md"):
            path = REPO_ROOT / rel
            if path.is_file() and path.stat().st_size <= 2_000_000:
                try:
                    reference_texts[rel] = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
    return reference_texts


def validate_script_reference_policy(
    *,
    script_paths: list[str],
    reference_texts: dict[str, str],
    registry: dict[str, dict[str, str]],
) -> list[str]:
    findings: list[str] = []
    for script in sorted(set(script_paths)):
        script_path = Path(script)
        module_name = script_path.with_suffix("").as_posix().replace("/", ".")
        basename = script_path.name
        import_name = script_path.stem
        needles = {
            script,
            basename,
            module_name,
            f"scripts.{import_name}",
            f"from scripts import {import_name}",
            f"import scripts.{import_name}",
        }
        entry = registry.get(script)
        if entry is not None:
            status = str(entry.get("status", "")).strip()
            reason = str(entry.get("reason", "")).strip()
            if status not in ALLOWED_SCRIPT_REGISTRY_STATUSES:
                findings.append(f"script registry status is invalid for {script}: {status}")
            if not reason:
                findings.append(f"script registry reason is missing for {script}")
            continue
        referenced = any(
            any(needle in text for needle in needles)
            for source, text in reference_texts.items()
            if source != script
        )
        if not referenced:
            findings.append(f"script has no repo reference or registry entry: {script}")
    return findings


def validate(*, allow_dirty: bool = False) -> tuple[list[str], list[str]]:
    findings: list[str] = []
    warnings: list[str] = []

    if not allow_dirty:
        status = _git(["status", "--short"])
        if status:
            findings.append(
                "git working tree is not clean; commit or discard generated outputs before release"
            )

    for rel in REQUIRED_PATHS:
        if not (REPO_ROOT / rel).exists():
            findings.append(f"required reproducibility path missing: {rel}")

    pytest_ini = (
        (REPO_ROOT / "pytest.ini").read_text(encoding="utf-8") if (REPO_ROOT / "pytest.ini").exists() else ""
    )
    for marker in REQUIRED_PYTEST_MARKERS:
        if f"{marker}:" not in pytest_ini:
            findings.append(f"pytest marker missing: {marker}")
    if "not local_data" not in pytest_ini or "not artifact_mutation" not in pytest_ini:
        findings.append("pytest default addopts must exclude local_data and artifact_mutation tests")

    artifact_findings, artifact_warnings = validate_artifact_policy()
    findings.extend(artifact_findings)
    warnings.extend(artifact_warnings)

    script_paths = _git_path_list(["ls-files", "scripts/*.py"])
    script_paths.extend(_git_path_list(["ls-files", "-o", "--exclude-standard", "scripts/*.py"]))
    findings.extend(
        validate_script_reference_policy(
            script_paths=script_paths,
            reference_texts=_reference_texts(),
            registry=_load_script_registry(),
        )
    )

    sidecars = find_sidecars(REPO_ROOT, default_exclude_parts(REPO_ROOT))
    if sidecars:
        findings.append(f"AppleDouble sidecar exists: {sidecars[0].relative_to(REPO_ROOT).as_posix()}")

    return findings, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-dirty", action="store_true", help="Do not fail on current uncommitted source edits"
    )
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
