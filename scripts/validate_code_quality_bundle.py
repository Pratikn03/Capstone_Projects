#!/usr/bin/env python3
"""Validate a sanitized ORIUS code-quality bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

try:
    from build_code_quality_bundle import (
        DISALLOWED_PARTS,
        DISALLOWED_SUFFIXES,
        PRIVATE_HEALTHCARE_MARKERS,
        SECRET_MARKERS,
    )
except ImportError:
    from scripts.build_code_quality_bundle import (
        DISALLOWED_PARTS,
        DISALLOWED_SUFFIXES,
        PRIVATE_HEALTHCARE_MARKERS,
        SECRET_MARKERS,
    )


REQUIRED_TOP_LEVEL = {"src", "scripts", "tests", "configs"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_text(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    findings = [f"secret marker {marker}" for marker in SECRET_MARKERS if marker in text]
    if path.suffix.lower() in {".csv", ".tsv"} and all(
        marker in text for marker in PRIVATE_HEALTHCARE_MARKERS
    ):
        findings.append("private healthcare row identifiers")
    return findings


def validate_bundle(bundle_dir: Path) -> dict[str, object]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {"bundle_dir": str(bundle_dir), "pass": False, "errors": ["missing manifest.json"]}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("upload_performed") is not False:
        errors.append("manifest must record upload_performed=false")
    if manifest.get("upload_requires_confirmation") is not True:
        errors.append("manifest must require upload confirmation")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        errors.append("manifest files must be a non-empty list")
        files = []

    top_level: set[str] = set()
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            errors.append("manifest contains a non-object file entry")
            continue
        rel = str(entry.get("relative_path", ""))
        if not rel:
            errors.append("file entry missing relative_path")
            continue
        if rel in seen:
            errors.append(f"duplicate manifest entry: {rel}")
        seen.add(rel)
        path = bundle_dir / rel
        parts = set(Path(rel).parts)
        top_level.add(Path(rel).parts[0])
        if path.name.startswith("._"):
            errors.append(f"AppleDouble sidecar included: {rel}")
        if parts & DISALLOWED_PARTS:
            errors.append(f"disallowed path part included: {rel}")
        if path.suffix.lower() in DISALLOWED_SUFFIXES:
            errors.append(f"disallowed heavy/private suffix included: {rel}")
        if not path.is_file():
            errors.append(f"manifest file missing from bundle: {rel}")
            continue
        expected_hash = entry.get("sha256")
        if expected_hash != _sha256(path):
            errors.append(f"sha256 mismatch: {rel}")
        for field in ("source_path", "category", "claim_scope", "size_bytes"):
            if field not in entry:
                errors.append(f"manifest entry missing {field}: {rel}")
        for finding in _scan_text(path):
            errors.append(f"{rel}: {finding}")

    missing_top = REQUIRED_TOP_LEVEL - top_level
    if missing_top:
        errors.append(f"missing required top-level bundle paths: {sorted(missing_top)}")

    for path in bundle_dir.rglob("*"):
        if not path.is_file() or path == manifest_path:
            continue
        rel = path.relative_to(bundle_dir).as_posix()
        if path.name.startswith("._"):
            errors.append(f"AppleDouble sidecar present outside manifest: {rel}")
        if set(Path(rel).parts) & DISALLOWED_PARTS:
            errors.append(f"disallowed path part present outside manifest: {rel}")
        if path.suffix.lower() in DISALLOWED_SUFFIXES:
            errors.append(f"disallowed heavy/private suffix present outside manifest: {rel}")

    return {
        "bundle_dir": str(bundle_dir),
        "pass": not errors,
        "file_count": len(files),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path, help="Bundle directory to validate.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_bundle(args.bundle_dir)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["pass"]:
        print(f"code-quality bundle valid: {result['bundle_dir']} ({result['file_count']} files)")
    else:
        print(f"code-quality bundle failed: {result['bundle_dir']}")
        for error in cast(list[str], result["errors"]):
            print(f"- {error}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
