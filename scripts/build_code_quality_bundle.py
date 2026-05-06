#!/usr/bin/env python3
"""Build a sanitized ORIUS code-quality bundle for remote QA.

The bundle is a local artifact only. Uploading it to Hugging Face or any other
third party must be an explicit separate action after reviewing the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

DEFAULT_OUT_ROOT = Path("artifacts/code_quality")
DEFAULT_HF_DATASET = "pratikn03/orius-code-quality-private"
ALLOWLIST_PATHS = (
    ".github/workflows",
    "configs",
    "docs/artifact_policy.md",
    "docs/reproducibility.md",
    "docs/claim_ledger.md",
    "iot",
    "scripts",
    "services",
    "src",
    "tests",
    "Makefile",
    "README.md",
    "mypy.ini",
    "pyproject.toml",
    "pytest.ini",
    "requirements.lock.txt",
    "requirements.txt",
    "ruff.toml",
)
DISALLOWED_PARTS = {
    ".cache",
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv2",
    "artifacts",
    "data",
    "node_modules",
    "reports",
}
DISALLOWED_SUFFIXES = {
    ".7z",
    ".db",
    ".doc",
    ".docx",
    ".duckdb",
    ".gz",
    ".jpg",
    ".jpeg",
    ".mov",
    ".mp4",
    ".onnx",
    ".parquet",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".sqlite",
    ".tar",
    ".tgz",
    ".zip",
}
SECRET_MARKERS = (
    "AWS_" + "SECRET_ACCESS_KEY=",
    "BEGIN " + "PRIVATE KEY",
    "HF" + "_TOKEN=",
    "OPENAI" + "_API_KEY=",
    "pass" + "word=",
    "secret" + "_key=",
)
PRIVATE_HEALTHCARE_MARKERS = (
    "subject_id",
    "hadm_id",
    "stay_id",
)


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ignore_missing_rmtree_error(_func, _path, exc_info) -> None:
    error = exc_info[1]
    if isinstance(error, FileNotFoundError):
        return
    raise error


def _remove_generated_appledouble_sidecars(bundle_dir: Path) -> int:
    removed = 0
    for path in sorted(bundle_dir.rglob("._*"), reverse=True):
        try:
            if path.is_dir():
                shutil.rmtree(path, onerror=_ignore_missing_rmtree_error)
            else:
                path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def _rel(path: Path, root: Path) -> str:
    return path.resolve(strict=False).relative_to(root).as_posix()


def _blocked_by_parts(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & DISALLOWED_PARTS)


def _is_allowed_file(path: Path, root: Path) -> bool:
    rel = _rel(path, root)
    name = path.name
    if name.startswith("._"):
        return False
    if _blocked_by_parts(Path(rel)):
        return False
    suffix = path.suffix.lower()
    if suffix in DISALLOWED_SUFFIXES:
        return False
    return not name.endswith((".pyc", ".pyo"))


def _iter_allowlisted_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for raw in ALLOWLIST_PATHS:
        path = root / raw
        if path.is_file():
            if _is_allowed_file(path, root):
                files.append(path)
            continue
        if not path.is_dir():
            continue
        for current_root, dirs, names in os.walk(path):
            current = Path(current_root)
            dirs[:] = [
                name
                for name in dirs
                if not name.startswith("._") and not _blocked_by_parts(Path(_rel(current / name, root)))
            ]
            for name in names:
                file_path = current / name
                if _is_allowed_file(file_path, root):
                    files.append(file_path)
    return sorted(set(files), key=lambda item: _rel(item, root))


def _category_for(path: str) -> str:
    first = path.split("/", 1)[0]
    if path.startswith(".github/"):
        return "ci_workflow"
    if first in {"src", "services", "iot"}:
        return "source_code"
    if first == "scripts":
        return "cli_script"
    if first == "tests":
        return "test"
    if first == "configs":
        return "config"
    if first == "docs":
        return "documentation"
    return "project_metadata"


def _scan_text(path: Path) -> tuple[list[str], list[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return [], []
    secrets = [marker for marker in SECRET_MARKERS if marker in text]
    private_rows: list[str] = []
    if path.suffix.lower() in {".csv", ".tsv", ".parquet"} and all(
        marker in text for marker in PRIVATE_HEALTHCARE_MARKERS
    ):
        private_rows.append("healthcare_row_identifier_columns")
    return secrets, private_rows


def build_bundle(
    root: Path, out_root: Path, *, stamp: str | None = None, create_tar: bool = True
) -> dict[str, object]:
    root = root.resolve()
    stamp = stamp or _utc_stamp()
    bundle_name = f"orius-code-quality-{stamp}"
    bundle_dir = (out_root if out_root.is_absolute() else root / out_root).resolve() / bundle_name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir, onerror=_ignore_missing_rmtree_error)
    bundle_dir.mkdir(parents=True)

    files = _iter_allowlisted_files(root)
    manifest_entries: list[dict[str, object]] = []
    errors: list[str] = []
    for source in files:
        rel = _rel(source, root)
        secret_markers, private_markers = _scan_text(source)
        if secret_markers:
            errors.append(f"{rel}: contains secret marker(s) {secret_markers}")
            continue
        if private_markers:
            errors.append(f"{rel}: contains private healthcare row marker(s) {private_markers}")
            continue

        dest = bundle_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        manifest_entries.append(
            {
                "relative_path": rel,
                "source_path": str(source),
                "category": _category_for(rel),
                "claim_scope": "three_domain_code_quality_only",
                "size_bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "bundle_name": bundle_name,
        "bundle_dir": str(bundle_dir),
        "target_hf_dataset": DEFAULT_HF_DATASET,
        "upload_performed": False,
        "upload_requires_confirmation": True,
        "claim_boundary": "Sanitized code-quality bundle only; excludes raw datasets, reports, private rows, secrets, caches, and model/runtime artifacts.",
        "file_count": len(manifest_entries),
        "files": manifest_entries,
        "errors": errors,
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    removed_sidecars = _remove_generated_appledouble_sidecars(bundle_dir)
    if removed_sidecars:
        manifest["removed_generated_appledouble_sidecars"] = removed_sidecars
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _remove_generated_appledouble_sidecars(bundle_dir)

    if create_tar:
        tar_path = bundle_dir.with_suffix(".tar.gz")
        if tar_path.exists():
            tar_path.unlink()
        with tarfile.open(tar_path, "w:gz") as archive:
            archive.add(bundle_dir, arcname=bundle_dir.name)
        manifest["tarball"] = str(tar_path)
        manifest["tarball_sha256"] = _sha256(tar_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument(
        "--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Output root for local bundle."
    )
    parser.add_argument("--stamp", help="Deterministic stamp for tests or reproducible builds.")
    parser.add_argument("--no-tar", action="store_true", help="Do not create a tar.gz archive.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_bundle(args.root, args.out_root, stamp=args.stamp, create_tar=not args.no_tar)
    manifest_path = Path(str(manifest["bundle_dir"])) / "manifest.json"
    print(f"code-quality bundle written: {manifest_path} ({manifest['file_count']} files)")
    errors = cast(list[str], manifest["errors"])
    if errors:
        for error in errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
