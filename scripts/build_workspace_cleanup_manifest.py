#!/usr/bin/env python3
"""Build a non-destructive workspace cleanup manifest.

The manifest is intentionally conservative: it lists local-only artifacts that
are deletion candidates, but it does not delete them. Actual deletion remains a
separate action that should be confirmed after reviewing the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

try:  # Script execution: python scripts/build_workspace_cleanup_manifest.py
    from cleanup_appledouble import default_exclude_parts, find_sidecars
except ImportError:  # Test/import execution: import scripts.build_workspace_cleanup_manifest
    from scripts.cleanup_appledouble import default_exclude_parts, find_sidecars


DEFAULT_OUT = Path("reports/audit/workspace_cleanup_manifest.json")
CACHE_DIR_NAMES = {
    ".cache",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
CACHE_DIR_PARTS = {
    "frontend/.next",
    "frontend/node_modules",
    "node_modules",
    ".next",
    "build",
    "dist",
}
PRUNE_DIR_NAMES = {".git", ".venv", ".venv2"}
RELEASE_PREFIX = "orius-three-domain-artifact-"
DOWNLOAD_HELPER_NAME_PARTS = ("download", "kaggle", "drive", "hf_dataset", "sync_nuplan")
CODE_REFERENCE_ROOTS = ("Makefile", "README.md", "docs", "configs", "tests", "scripts", "src")


@dataclass(frozen=True)
class Candidate:
    path: Path
    category: str
    reason: str
    safe_to_delete: bool
    is_dir: bool = False
    review_required: bool = False


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _repo_root_from_cwd() -> Path:
    return Path.cwd().resolve()


def _rel(path: Path, root: Path) -> str:
    return path.resolve(strict=False).relative_to(root).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_stats(path: Path) -> tuple[int, int]:
    file_count = 0
    size_bytes = 0
    for current_root, dirs, files in os.walk(path):
        current = Path(current_root)
        dirs[:] = [name for name in dirs if name not in PRUNE_DIR_NAMES]
        for name in files:
            file_path = current / name
            try:
                stat = file_path.stat()
            except FileNotFoundError:
                continue
            file_count += 1
            size_bytes += stat.st_size
    return file_count, size_bytes


def _candidate_payload(candidate: Candidate, root: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": _rel(candidate.path, root),
        "category": candidate.category,
        "reason": candidate.reason,
        "safe_to_delete": candidate.safe_to_delete,
        "review_required": candidate.review_required,
        "is_dir": candidate.is_dir,
    }
    try:
        stat = candidate.path.stat()
    except FileNotFoundError:
        payload["missing_at_manifest_time"] = True
        return payload

    payload["size_bytes"] = stat.st_size
    payload["mtime_ns"] = stat.st_mtime_ns
    if candidate.is_dir:
        file_count, tree_size = _tree_stats(candidate.path)
        payload["tree_file_count"] = file_count
        payload["tree_size_bytes"] = tree_size
        marker = candidate.path / "manifest.json"
        if marker.is_file():
            payload["manifest_json_sha256"] = _sha256(marker)
    elif candidate.path.is_file():
        payload["sha256"] = _sha256(candidate.path)
    return payload


def _is_pruned(path: Path, root: Path) -> bool:
    rel = _rel(path, root)
    parts = set(path.relative_to(root).parts)
    return bool(parts & PRUNE_DIR_NAMES) or rel.startswith("artifacts/code_quality/")


def _cache_candidates(root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for current_root, dirs, _files in os.walk(root):
        current = Path(current_root)
        dirs[:] = [name for name in dirs if name not in PRUNE_DIR_NAMES]
        for dirname in list(dirs):
            path = current / dirname
            rel = _rel(path, root)
            if dirname in CACHE_DIR_NAMES or rel in CACHE_DIR_PARTS:
                candidates.append(
                    Candidate(
                        path=path,
                        category="cache",
                        reason="local build/test cache",
                        safe_to_delete=True,
                        is_dir=True,
                    )
                )
                dirs.remove(dirname)
    return sorted(candidates, key=lambda item: _rel(item.path, root))


def _appledouble_candidates(root: Path) -> list[Candidate]:
    excludes = default_exclude_parts(root)
    return [
        Candidate(
            path=path,
            category="appledouble",
            reason="macOS AppleDouble metadata sidecar",
            safe_to_delete=True,
        )
        for path in find_sidecars(root, excludes)
    ]


def _release_candidates(root: Path) -> list[Candidate]:
    release_root = root / "artifacts" / "releases"
    if not release_root.is_dir():
        return []
    releases = sorted(
        [path for path in release_root.iterdir() if path.is_dir() and path.name.startswith(RELEASE_PREFIX)],
        key=lambda path: (path.name, path.stat().st_mtime_ns),
    )
    if len(releases) <= 1:
        return []
    newest = releases[-1]
    candidates: list[Candidate] = []
    for path in releases[:-1]:
        has_manifest = (path / "manifest.json").is_file()
        candidates.append(
            Candidate(
                path=path,
                category="duplicate_release",
                reason=f"older clean artifact release; newest retained is {newest.name}",
                safe_to_delete=has_manifest,
                is_dir=True,
                review_required=not has_manifest,
            )
        )
    return candidates


def _reference_texts(root: Path) -> list[Path]:
    paths: list[Path] = []
    for entry in CODE_REFERENCE_ROOTS:
        base = root / entry
        if base.is_file():
            paths.append(base)
        elif base.is_dir():
            for current_root, dirs, files in os.walk(base):
                current = Path(current_root)
                dirs[:] = [name for name in dirs if name not in PRUNE_DIR_NAMES and name != "__pycache__"]
                for name in files:
                    suffix = Path(name).suffix.lower()
                    if suffix in {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ""}:
                        paths.append(current / name)
    return paths


def _referenced_names(root: Path) -> set[str]:
    referenced: set[str] = set()
    texts = _reference_texts(root)
    for text_path in texts:
        try:
            text = text_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for script in (root / "scripts").glob("*.py"):
            if script.name in text and text_path != script:
                referenced.add(script.name)
    return referenced


def _download_helper_candidates(root: Path) -> list[Candidate]:
    scripts_dir = root / "scripts"
    if not scripts_dir.is_dir():
        return []
    referenced = _referenced_names(root)
    candidates: list[Candidate] = []
    for path in sorted(scripts_dir.glob("*.py")):
        lowered = path.name.lower()
        if not any(part in lowered for part in DOWNLOAD_HELPER_NAME_PARTS):
            continue
        if path.name in referenced:
            continue
        candidates.append(
            Candidate(
                path=path,
                category="stale_download_helper_review",
                reason="download/sync helper with no code/doc/test reference found",
                safe_to_delete=False,
                review_required=True,
            )
        )
    return candidates


def build_manifest(root: Path) -> dict[str, object]:
    root = root.resolve()
    candidates = [
        *_appledouble_candidates(root),
        *_cache_candidates(root),
        *_release_candidates(root),
        *_download_helper_candidates(root),
    ]
    seen: set[str] = set()
    payloads: list[dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: (_rel(item.path, root), item.category)):
        rel = _rel(candidate.path, root)
        if rel in seen:
            continue
        seen.add(rel)
        payloads.append(_candidate_payload(candidate, root))

    counts = Counter(str(item["category"]) for item in payloads)
    return {
        "generated_at_utc": _utc_now(),
        "root": str(root),
        "mode": "dry_run",
        "deletion_performed": False,
        "delete_requires_confirmation": True,
        "summary": {
            "candidate_count": len(payloads),
            "safe_to_delete_count": sum(1 for item in payloads if item.get("safe_to_delete") is True),
            "review_required_count": sum(1 for item in payloads if item.get("review_required") is True),
            "by_category": dict(sorted(counts.items())),
        },
        "candidates": payloads,
    }


def write_manifest(manifest: dict[str, object], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root to scan.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Manifest output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = build_manifest(root)
    out_path = args.out if args.out.is_absolute() else root / args.out
    write_manifest(manifest, out_path)
    summary = cast(Mapping[str, object], manifest["summary"])
    print(
        "workspace cleanup dry-run manifest written: "
        f"{out_path} ({summary['candidate_count']} candidates, "
        f"{summary['safe_to_delete_count']} safe, {summary['review_required_count']} review)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
