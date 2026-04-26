#!/usr/bin/env python3
"""Validate the clean ORIUS three-domain artifact release folder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:  # pragma: no cover - direct script execution uses the fallback branch.
    from scripts.clean_artifact_release_common import (
        DISALLOWED_PARTS,
        DISALLOWED_SUFFIXES,
        PUBLIC_STALE_TEXT_MARKERS,
        REQUIRED_RELEASE_PATHS,
        SENSITIVE_NAME_MARKERS,
        STALE_REPORT_MARKERS,
        posix,
    )
except ModuleNotFoundError:  # pragma: no cover
    from clean_artifact_release_common import (  # type: ignore
        DISALLOWED_PARTS,
        DISALLOWED_SUFFIXES,
        PUBLIC_STALE_TEXT_MARKERS,
        REQUIRED_RELEASE_PATHS,
        SENSITIVE_NAME_MARKERS,
        STALE_REPORT_MARKERS,
        posix,
    )


HASH_CHUNK_BYTES = 16 * 1024 * 1024
TEXT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt", ".yaml", ".yml"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        rows[rel] = digest
    return rows


def _is_bad_path(rel: str) -> str | None:
    path = Path(rel)
    lower = rel.lower()
    if path.name.startswith("._"):
        return "AppleDouble file"
    if any(part in DISALLOWED_PARTS for part in path.parts):
        return "cache, dependency, virtualenv, or git path"
    if path.suffix.lower() in DISALLOWED_SUFFIXES:
        return f"disallowed suffix {path.suffix}"
    if any(marker in path.name.lower() for marker in SENSITIVE_NAME_MARKERS):
        return "sensitive-looking filename"
    if lower.startswith("evidence/healthcare/") and path.suffix.lower() == ".parquet":
        return "patient-level healthcare parquet evidence is excluded"
    if any(marker in lower for marker in STALE_REPORT_MARKERS):
        return "stale legacy report path"
    return None


def _scan_text_for_stale_claims(path: Path) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 2_000_000:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return []
    findings: list[str] = []
    for marker in PUBLIC_STALE_TEXT_MARKERS:
        if marker not in text:
            continue
        if marker == "full autonomous-driving field closure claimed" and (
            "no road deployment or full autonomous-driving field closure claimed" in text
            or "not completed carla simulation, road deployment, or full autonomous-driving field closure" in text
        ):
            continue
        findings.append(marker)
    return findings


def validate_release(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir)
    if not root.exists():
        raise FileNotFoundError(f"Release directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Release path is not a directory: {root}")

    manifest_path = root / "manifest.json"
    sha_path = root / "MANIFEST.sha256"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not sha_path.exists():
        raise FileNotFoundError(f"Missing checksum manifest: {sha_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    claim_boundary = str(manifest.get("claim_boundary", ""))
    claim_lower = claim_boundary.lower()
    for phrase in ("battery", "nuplan", "healthcare"):
        if phrase not in claim_lower:
            raise ValueError(f"Claim boundary is missing required promoted domain phrase: {phrase}")
    if "six defended domains" in claim_lower or "six-domain" in claim_lower:
        raise ValueError("Claim boundary still advertises a six-domain public claim")

    for rel in REQUIRED_RELEASE_PATHS:
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"Missing required release file: {rel}")

    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    rel_files = {posix(path.relative_to(root)) for path in files}

    for rel in sorted(rel_files):
        problem = _is_bad_path(rel)
        if problem:
            raise ValueError(f"Disallowed file in release ({problem}): {rel}")
        stale_markers = _scan_text_for_stale_claims(root / rel)
        if stale_markers:
            raise ValueError(f"Stale public claim marker(s) {stale_markers} found in {rel}")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("manifest.json must contain a list field named 'files'")

    entry_by_rel: dict[str, dict[str, Any]] = {}
    required_entry_fields = {"relative_path", "source_path", "category", "claim_scope", "size_bytes", "sha256"}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest files entries must be objects")
        missing = sorted(required_entry_fields - set(entry))
        if missing:
            raise ValueError(f"manifest file entry is missing fields {missing}: {entry}")
        rel = str(entry["relative_path"])
        if rel in {"manifest.json", "MANIFEST.sha256"}:
            raise ValueError("manifest.json and MANIFEST.sha256 are tracked by MANIFEST.sha256, not manifest files")
        if rel in entry_by_rel:
            raise ValueError(f"Duplicate manifest file entry: {rel}")
        source_path = str(entry["source_path"])
        problem = _is_bad_path(rel) or _is_bad_path(source_path)
        if problem:
            raise ValueError(f"Disallowed manifest entry ({problem}): {rel}")
        if any(marker in source_path.lower() for marker in STALE_REPORT_MARKERS):
            raise ValueError(f"Stale source path in manifest: {source_path}")
        entry_by_rel[rel] = entry

    expected_manifest_entries = rel_files - {"manifest.json", "MANIFEST.sha256"}
    if set(entry_by_rel) != expected_manifest_entries:
        missing = sorted(expected_manifest_entries - set(entry_by_rel))
        extra = sorted(set(entry_by_rel) - expected_manifest_entries)
        raise ValueError(f"manifest file list mismatch; missing={missing[:10]} extra={extra[:10]}")

    for rel, entry in sorted(entry_by_rel.items()):
        path = root / rel
        size = path.stat().st_size
        if int(entry["size_bytes"]) != size:
            raise ValueError(f"Size mismatch for {rel}: manifest={entry['size_bytes']} actual={size}")
        digest = _sha256(path)
        if str(entry["sha256"]) != digest:
            raise ValueError(f"SHA256 mismatch for {rel}")

    sha_rows = _sha_manifest(sha_path)
    expected_sha_rows = rel_files - {"MANIFEST.sha256"}
    if set(sha_rows) != expected_sha_rows:
        missing = sorted(expected_sha_rows - set(sha_rows))
        extra = sorted(set(sha_rows) - expected_sha_rows)
        raise ValueError(f"MANIFEST.sha256 mismatch; missing={missing[:10]} extra={extra[:10]}")
    for rel, expected in sorted(sha_rows.items()):
        actual = _sha256(root / rel)
        if expected != actual:
            raise ValueError(f"MANIFEST.sha256 digest mismatch for {rel}")

    return {
        "passed": True,
        "bundle_id": manifest.get("bundle_id"),
        "file_count": len(rel_files),
        "manifest_file_count": len(entry_by_rel),
        "total_size_bytes": sum((root / rel).stat().st_size for rel in rel_files),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a clean ORIUS artifact release")
    parser.add_argument("bundle_dir")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = validate_release(args.bundle_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
