"""Shared helpers for repo-local real-data contracts and provenance."""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from orius.data_pipeline.external_raw import EXTERNAL_DATA_ROOT_ENV, get_external_dataset_dir


DEFAULT_MIN_FREE_GIB = 250.0


@dataclass(frozen=True)
class ResolvedRawSource:
    """Resolved raw-data location for a dataset."""

    path: Path
    source_kind: str
    checked_locations: tuple[str, ...]


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def summarize_files(root: Path, *, limit: int = 200) -> dict[str, Any]:
    """Return a compact inventory of files under *root*."""
    if not root.exists():
        return {
            "root": str(root),
            "total_files": 0,
            "total_bytes": 0,
            "files": [],
            "truncated": False,
        }

    files = sorted(path for path in root.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in files)
    listed = [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": int(path.stat().st_size),
        }
        for path in files[:limit]
    ]
    return {
        "root": str(root),
        "total_files": len(files),
        "total_bytes": int(total_bytes),
        "files": listed,
        "truncated": len(files) > limit,
    }


def summarize_csv_output(csv_path: Path) -> dict[str, Any]:
    """Return row/column summary for a CSV output file."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return {
        "processed_output": str(csv_path),
        "rows": int(rows),
        "columns": header,
        "generated_at_utc": utc_now_iso(),
    }


def summarize_disk_usage(target: Path, *, min_free_gib: float = DEFAULT_MIN_FREE_GIB) -> dict[str, Any]:
    """Return disk usage information for *target*."""
    usage = shutil.disk_usage(target)
    free_gib = usage.free / (1024**3)
    total_gib = usage.total / (1024**3)
    used_gib = usage.used / (1024**3)
    return {
        "path": str(target),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "total_gib": round(total_gib, 2),
        "used_gib": round(used_gib, 2),
        "free_gib": round(free_gib, 2),
        "min_free_gib": float(min_free_gib),
        "passes_threshold": free_gib >= min_free_gib,
    }


def require_min_free_space(target: Path, *, min_free_gib: float = DEFAULT_MIN_FREE_GIB) -> dict[str, Any]:
    """Raise if the target volume is below the requested free-space threshold."""
    summary = summarize_disk_usage(target, min_free_gib=min_free_gib)
    if not summary["passes_threshold"]:
        raise RuntimeError(
            f"Insufficient free space under {target}: "
            f"{summary['free_gib']:.2f} GiB available, need at least {min_free_gib:.2f} GiB."
        )
    return summary


def _tool_search_path() -> str:
    """Build a stable executable search path that prefers the active project venv."""
    repo_root = Path(__file__).resolve().parents[3]
    candidate_dirs = [
        str(Path(sys.executable).resolve().parent),
        str((repo_root / ".venv" / "bin").resolve()),
    ]
    existing_path = os.environ.get("PATH")
    if existing_path:
        candidate_dirs.append(existing_path)
    unique_dirs = list(dict.fromkeys(candidate_dirs))
    return os.pathsep.join(unique_dirs)


def tool_status(tool_names: Sequence[str]) -> dict[str, str | None]:
    """Return executable paths for requested CLI tools."""
    search_path = _tool_search_path()
    return {name: shutil.which(name, path=search_path) for name in tool_names}


def module_status(module_names: Sequence[str]) -> dict[str, bool]:
    """Return whether Python modules are importable in the current interpreter."""
    return {name: importlib.util.find_spec(name) is not None for name in module_names}


def resolve_repo_or_external_raw_dir(
    repo_dir: Path,
    *,
    external_dataset_key: str | None = None,
    explicit_root: Path | None = None,
    required: bool = False,
) -> ResolvedRawSource | None:
    """Resolve a dataset root using repo-local storage first, then external storage."""
    checked_locations = [str(repo_dir)]
    if repo_dir.exists():
        return ResolvedRawSource(path=repo_dir, source_kind="repo_local", checked_locations=tuple(checked_locations))

    if external_dataset_key is not None:
        external_dir = get_external_dataset_dir(external_dataset_key, explicit_root, required=False)
        checked_locations.append(
            f"${EXTERNAL_DATA_ROOT_ENV}/{external_dataset_key}"
            if external_dir is None
            else str(external_dir)
        )
        if external_dir is not None and external_dir.exists():
            return ResolvedRawSource(
                path=external_dir,
                source_kind="external",
                checked_locations=tuple(checked_locations),
            )

    if required:
        raise FileNotFoundError(
            "Missing required raw dataset. Checked: " + " ; ".join(checked_locations)
        )
    return None


def build_provenance_manifest(
    *,
    domain: str,
    dataset_key: str,
    provider: str,
    version: str,
    raw_source: ResolvedRawSource,
    processed_output: Path,
    output_summary: dict[str, Any],
    raw_inventory: dict[str, Any],
    source_urls: Sequence[str],
    license_notes: str,
    access_notes: str,
    canonical_source: bool,
    used_fallback: bool,
    notes: Sequence[str] | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard provenance manifest payload."""
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "domain": domain,
        "dataset_key": dataset_key,
        "provider": provider,
        "version": version,
        "source_kind": raw_source.source_kind,
        "checked_locations": list(raw_source.checked_locations),
        "raw_root": str(raw_source.path),
        "processed_output": str(processed_output),
        "canonical_source": bool(canonical_source),
        "used_fallback": bool(used_fallback),
        "license_notes": license_notes,
        "access_notes": access_notes,
        "source_urls": list(source_urls),
        "raw_inventory": raw_inventory,
        "output_summary": output_summary,
        "notes": list(notes or ()),
    }
    if extras:
        payload.update(extras)
    return payload
