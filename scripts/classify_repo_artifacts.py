#!/usr/bin/env python3
"""Classify repo paths into source, evidence, generated, and local-only classes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".css",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".md",
    ".tex",
}
LOCAL_DATA_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".tfrecord", ".tfrecords"}
MODEL_SUFFIXES = {".pkl", ".pt", ".onnx", ".joblib", ".ckpt", ".safetensors"}
RUNTIME_SUFFIXES = {".duckdb", ".parquet", ".wal"}
CACHE_DIRS = {
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".hypothesis",
    ".playwright-mcp",
    "__pycache__",
    "node_modules",
    "htmlcov",
}
TEMP_NAMES = {".DS_Store"}


def _parts(path: str | Path) -> tuple[str, ...]:
    return Path(path).as_posix().split("/")


def classify_path(path: str | Path) -> str:
    """Return the repository artifact class for a relative path."""
    rel = Path(path).as_posix()
    name = Path(rel).name
    suffixes = [s.lower() for s in Path(rel).suffixes]
    suffix = Path(rel).suffix.lower()
    parts = _parts(rel)

    if name == "external_sources_manifest.json":
        return "small_canonical_evidence"
    if (
        name.startswith("._")
        or name in TEMP_NAMES
        or ("smoke" in name.lower() and suffix in {".png", ".jpg", ".jpeg"})
    ):
        return "temporary_ai_codex_artifact"
    if any(part in CACHE_DIRS for part in parts):
        return "cache_build_output"
    if parts[0] == "data":
        return "local_dataset" if rel != "data/DATASET_DOWNLOAD_GUIDE.md" else "source"
    if parts[0] == "artifacts":
        return "model_artifact" if suffix in MODEL_SUFFIXES else "generated_artifact"
    if suffix in MODEL_SUFFIXES:
        return "model_artifact"
    if suffix in RUNTIME_SUFFIXES:
        return "generated_runtime_artifact"
    if suffix in LOCAL_DATA_SUFFIXES or suffixes[-2:] == [".tar", ".gz"]:
        return "local_dataset"
    if parts[0] == "reports":
        if "runtime_traces.csv" in rel or "certificate_expiry_trace.csv" in rel:
            return "generated_runtime_artifact"
        if parts[:2] == ["reports", "publication"] and suffix in {".csv", ".json", ".md", ".tex"}:
            return "small_canonical_evidence"
        return "generated_report"
    if parts[0] == "paper" and "assets" in parts:
        return "small_canonical_evidence"
    if suffix in SOURCE_SUFFIXES or name in {
        "Makefile",
        "Dockerfile",
        "package-lock.json",
        "requirements.lock.txt",
    }:
        return "source"
    return "other"


def _git_paths(args: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line for line in proc.stdout.splitlines() if line]


def iter_inventory(include_untracked: bool = False) -> Iterable[dict[str, object]]:
    tracked_paths = set(_git_paths(["ls-files"]))
    paths = set(tracked_paths)
    if include_untracked:
        paths.update(_git_paths(["ls-files", "-o", "--exclude-standard"]))
    for rel in sorted(paths):
        path = REPO_ROOT / rel
        yield {
            "path": rel,
            "category": classify_path(rel),
            "tracked": rel in tracked_paths,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-untracked", action="store_true")
    parser.add_argument("--format", choices={"json", "csv"}, default="csv")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = list(iter_inventory(include_untracked=args.include_untracked))
    if args.format == "json":
        payload = json.dumps(rows, indent=2, sort_keys=True)
    else:
        from io import StringIO

        handle = StringIO()
        writer = csv.DictWriter(handle, fieldnames=["path", "category", "tracked", "exists", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)
        payload = handle.getvalue()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="" if payload.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
