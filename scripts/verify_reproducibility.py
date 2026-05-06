#!/usr/bin/env python3
"""Verify tracked publication artifacts against the release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "reports" / "publication" / "release_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_root(path: Path) -> Path:
    return path.parent


def _verify_manifest(manifest_path: Path) -> dict[str, object]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_hashes = payload.get("artifact_hashes_sha256", {})
    if not isinstance(artifact_hashes, dict):
        raise ValueError("release_manifest.json missing artifact_hashes_sha256 mapping")

    artifact_root = _artifact_root(manifest_path)
    checked: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []

    for name, expected_hash in sorted(artifact_hashes.items()):
        artifact_path = artifact_root / str(name)
        if not artifact_path.exists():
            mismatches.append(
                {
                    "artifact": str(name),
                    "status": "missing",
                    "expected_sha256": str(expected_hash),
                }
            )
            continue
        actual_hash = _sha256(artifact_path)
        row = {
            "artifact": str(name),
            "path": str(artifact_path.relative_to(REPO_ROOT)),
            "expected_sha256": str(expected_hash),
            "actual_sha256": actual_hash,
            "matches": actual_hash == str(expected_hash),
        }
        checked.append(row)
        if actual_hash != str(expected_hash):
            mismatches.append(row)

    return {
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "artifact_count": len(artifact_hashes),
        "checked_count": len(checked),
        "mismatch_count": len(mismatches),
        "matches": len(mismatches) == 0,
        "mismatches": mismatches,
        "checked": checked,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify publication artifact hashes against the release manifest"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Release manifest path")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report output path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = _verify_manifest(args.manifest.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
