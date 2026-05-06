#!/usr/bin/env python3
"""Validate a workspace cleanup manifest before any destructive action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

SAFE_CATEGORIES = {"appledouble", "cache", "duplicate_release"}
NEVER_DELETE_PARTS = {".git", ".venv", ".venv2"}
RELEASE_PREFIX = "artifacts/releases/orius-three-domain-artifact-"
PERSONAL_FILE_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".doc", ".docx"}


def _parts(path: str) -> set[str]:
    return set(Path(path).parts)


def _validate_candidate(candidate: dict[str, object]) -> list[str]:
    errors: list[str] = []
    path = str(candidate.get("path", ""))
    category = str(candidate.get("category", ""))
    safe_to_delete = candidate.get("safe_to_delete") is True

    if not path:
        return ["candidate is missing path"]
    if path.startswith("/") or ".." in Path(path).parts:
        errors.append(f"unsafe relative path: {path}")
    if _parts(path) & NEVER_DELETE_PARTS:
        errors.append(f"manifest must not include protected path: {path}")

    name = Path(path).name
    if category == "appledouble":
        if not name.startswith("._"):
            errors.append(f"appledouble candidate is not a sidecar: {path}")
    elif category == "cache":
        if not candidate.get("is_dir"):
            errors.append(f"cache candidate must be a directory: {path}")
    elif category == "duplicate_release":
        if not path.startswith(RELEASE_PREFIX):
            errors.append(f"duplicate release must stay under clean release root: {path}")
        if not candidate.get("is_dir"):
            errors.append(f"duplicate release candidate must be a directory: {path}")
    elif safe_to_delete:
        errors.append(f"unknown safe-to-delete category: {category} at {path}")

    suffix = Path(path).suffix.lower()
    if safe_to_delete and suffix in PERSONAL_FILE_SUFFIXES and not name.startswith("._"):
        errors.append(f"safe deletion candidate targets a non-sidecar personal file type: {path}")
    return errors


def validate_manifest(manifest_path: Path) -> dict[str, object]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("mode") != "dry_run":
        errors.append("manifest mode must be dry_run")
    if payload.get("deletion_performed") is not False:
        errors.append("manifest must not record deletion_performed=true")
    if payload.get("delete_requires_confirmation") is not True:
        errors.append("manifest must require confirmation before deletion")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        errors.append("manifest candidates must be a list")
        candidates = []

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            errors.append(f"candidate {index} is not an object")
            continue
        errors.extend(_validate_candidate(candidate))

    safe_categories = {
        str(candidate.get("category"))
        for candidate in candidates
        if isinstance(candidate, dict) and candidate.get("safe_to_delete") is True
    }
    unknown_safe = safe_categories - SAFE_CATEGORIES
    if unknown_safe:
        errors.append(f"unknown safe categories: {sorted(unknown_safe)}")

    return {
        "manifest": str(manifest_path),
        "pass": not errors,
        "candidate_count": len(candidates),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Manifest JSON to validate.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_manifest(args.manifest)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["pass"]:
        print(f"workspace cleanup manifest valid: {result['manifest']}")
    else:
        print(f"workspace cleanup manifest failed: {result['manifest']}")
        for error in cast(list[str], result["errors"]):
            print(f"- {error}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
