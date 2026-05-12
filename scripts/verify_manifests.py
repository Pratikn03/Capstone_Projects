#!/usr/bin/env python3
"""Verify SHA-256 hashes recorded in tracked reports/**/artifact_manifest.json.

Backs the §XI.E reproducibility claim: every artifact recorded in an
artifact_manifest.json is re-hashed and compared against the recorded digest.
Exits non-zero on any mismatch, missing file, or unreadable manifest.
By default only Git-tracked manifests are checked so local-only report caches
and ignored heavyweight run outputs do not make the clean-clone gate unstable.
Use --include-untracked to audit every local manifest under reports/.

Paths in manifests may be absolute (legacy: /Volumes/T9/gridpulse/...) or
relative to the manifest's parent directory. The verifier resolves both, and
rewrites legacy absolute prefixes onto the current repo root so a fresh clone
can still verify cached artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

LEGACY_ABS_PREFIX = "/Volumes/T9/gridpulse/"


@dataclass
class ManifestResult:
    manifest: Path
    ok: list[str] = field(default_factory=list)
    mismatched: list[tuple[str, str, str]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.mismatched and not self.missing


def _resolve(path_str: str, manifest_dir: Path, repo_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        if p.exists():
            return p
        if path_str.startswith(LEGACY_ABS_PREFIX):
            rel = path_str[len(LEGACY_ABS_PREFIX):]
            return repo_root / rel
        return p
    return (manifest_dir / p).resolve()


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def discover_manifests(reports_root: Path, repo_root: Path, include_untracked: bool) -> list[Path]:
    reports_root = reports_root.resolve()
    repo_root = repo_root.resolve()
    if include_untracked:
        return sorted(reports_root.rglob("artifact_manifest.json"))

    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return sorted(reports_root.rglob("artifact_manifest.json"))

    manifests: list[Path] = []
    for line in proc.stdout.splitlines():
        candidate = (repo_root / line).resolve()
        if candidate.name == "artifact_manifest.json" and _is_relative_to(candidate, reports_root):
            manifests.append(candidate)
    return sorted(manifests)


def verify_manifest(manifest_path: Path, repo_root: Path) -> ManifestResult:
    result = ManifestResult(manifest=manifest_path)
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        result.error = f"unreadable manifest: {exc}"
        return result

    artifacts = data.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        result.error = "manifest has no 'artifacts' dict"
        return result

    manifest_dir = manifest_path.parent
    for recorded_path, expected_digest in artifacts.items():
        resolved = _resolve(recorded_path, manifest_dir, repo_root)
        if not resolved.exists():
            result.missing.append(recorded_path)
            continue
        actual = _sha256(resolved)
        if actual.lower() != str(expected_digest).lower():
            result.mismatched.append((recorded_path, expected_digest, actual))
        else:
            result.ok.append(recorded_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports"),
        help="Root directory to scan for artifact_manifest.json files.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repo root used to rewrite legacy absolute paths.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Treat missing artifacts as a warning rather than a failure "
        "(useful when heavy artifacts live behind LFS).",
    )
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="Audit every local artifact_manifest.json under --reports-root, including ignored "
        "or untracked heavyweight run outputs.",
    )
    args = parser.parse_args(argv)

    manifests = discover_manifests(args.reports_root, args.repo_root, args.include_untracked)
    if not manifests:
        print(f"verify_manifests: no manifests found under {args.reports_root}", file=sys.stderr)
        return 1

    failed = warned = 0
    total_ok = total_mismatch = total_missing = 0
    for manifest in manifests:
        res = verify_manifest(manifest, args.repo_root.resolve())
        rel = manifest.relative_to(args.repo_root) if manifest.is_relative_to(args.repo_root) else manifest
        if res.error:
            print(f"FAIL {rel}: {res.error}")
            failed += 1
            continue
        total_ok += len(res.ok)
        total_mismatch += len(res.mismatched)
        total_missing += len(res.missing)
        is_fail = bool(res.mismatched) or (bool(res.missing) and not args.allow_missing)
        is_warn = bool(res.missing) and args.allow_missing and not res.mismatched
        if not res.mismatched and not res.missing:
            status = "OK  "
        elif is_fail:
            status = "FAIL"
        else:
            status = "WARN"
        if status == "OK  ":
            print(f"{status} {rel}: {len(res.ok)} artifacts verified")
            continue
        print(
            f"{status} {rel}: ok={len(res.ok)} "
            f"mismatch={len(res.mismatched)} missing={len(res.missing)}"
        )
        for p, exp, got in res.mismatched:
            print(f"    MISMATCH {p}\n      expected {exp}\n      got      {got}")
        for p in res.missing:
            print(f"    MISSING  {p}")
        if is_fail:
            failed += 1
        elif is_warn:
            warned += 1

    print(
        f"\nSummary: manifests={len(manifests)} failed={failed} warned={warned} "
        f"artifacts_ok={total_ok} mismatched={total_mismatch} missing={total_missing}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
