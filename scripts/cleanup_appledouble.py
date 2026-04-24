#!/usr/bin/env python3
"""Remove macOS AppleDouble sidecar files without touching active outputs.

The script writes a manifest before deletion so cleanup is auditable. By
default it skips known active freeze outputs and any currently running nuPlan
pipeline output directories detected from the process table.
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
import subprocess


DEFAULT_ACTIVE_RELEASE = "PREDEPLOY_MAX_BG_20260422T122751Z"
DEFAULT_EXCLUDE_PARTS = {
    f"reports/predeployment_freeze/{DEFAULT_ACTIVE_RELEASE}",
    f"artifacts/runs/de/{DEFAULT_ACTIVE_RELEASE}",
    f"reports/runs/de/{DEFAULT_ACTIVE_RELEASE}",
    ".tmp/nuplan_max",
}

ACTIVE_WRITE_PATTERNS = ("*.crdownload", "*.tmp", "*.parquet.tmp")
NUPLAN_PIPELINE_PATH_FLAGS = {
    "--av-processed-dir",
    "--av-reports-dir",
    "--av-models-dir",
    "--av-uncertainty-dir",
    "--overall-dir",
    "--out-dir",
    "--temp-dir",
}


def _repo_relative_prefix(root: Path, raw_path: str) -> str | None:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        return None


def _active_nuplan_pipeline_excludes(root: Path) -> set[str]:
    try:
        completed = subprocess.run(
            ["ps", "ax", "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return set()
    if completed.returncode != 0:
        return set()

    excludes: set[str] = set()
    active_commands = ("run_battery_av_pipeline.py", "build_nuplan_av_surface.py")
    for line in completed.stdout.splitlines():
        if not any(command in line for command in active_commands):
            continue
        if "run_battery_av_pipeline.py" in line and "--av-source nuplan" not in line:
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        for index, token in enumerate(tokens):
            flag = token.split("=", 1)[0]
            if flag not in NUPLAN_PIPELINE_PATH_FLAGS:
                continue
            value = token.split("=", 1)[1] if "=" in token else None
            if value is None and index + 1 < len(tokens):
                value = tokens[index + 1]
            if value:
                rel = _repo_relative_prefix(root, value)
                if rel:
                    excludes.add(rel)
    return excludes


def default_exclude_parts(root: Path) -> set[str]:
    exclude_parts = set(DEFAULT_EXCLUDE_PARTS)
    exclude_parts.update(_active_nuplan_pipeline_excludes(root))
    return exclude_parts


def _is_excluded(path: Path, root: Path, exclude_parts: set[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(rel == item or rel.startswith(f"{item}/") for item in exclude_parts)


def find_sidecars(root: Path, exclude_parts: set[str]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("._*")
        if path.is_file() and not _is_excluded(path, root, exclude_parts)
    )


def _active_write_candidates(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for pattern in ACTIVE_WRITE_PATTERNS:
        candidates.update(path for path in root.rglob(pattern) if path.is_file())
    return sorted(candidates)


def _open_paths(paths: list[Path]) -> list[Path]:
    if not paths:
        return []
    try:
        completed = subprocess.run(
            ["lsof", *[str(path) for path in paths]],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if completed.returncode not in (0, 1):
        return []
    opened: list[Path] = []
    stdout = completed.stdout.splitlines()
    for path in paths:
        if any(str(path) in line for line in stdout[1:]):
            opened.append(path)
    return opened


def _delete_git_object_sidecars(root: Path) -> int:
    """Remove Git-object AppleDouble files that macOS can recreate late."""
    git_root = root / ".git" / "objects"
    if not git_root.exists():
        return 0
    deleted = 0
    for path in git_root.rglob("._*"):
        if path.is_file():
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument("--manifest", default="reports/audit/appledouble_cleanup_manifest.json")
    parser.add_argument("--delete", action="store_true", help="Delete files after writing the manifest.")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Write the audit manifest without deleting. Omitted dry-runs are read-only.",
    )
    parser.add_argument(
        "--allow-active",
        action="store_true",
        help="Allow deletion while active download/temp outputs are open. Intended only for manual emergencies.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Repo-relative path prefix to skip. Can be repeated.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    exclude_parts = default_exclude_parts(root)
    exclude_parts.update(str(item).strip("/").replace("\\", "/") for item in args.exclude if item)

    sidecars = find_sidecars(root, exclude_parts)
    open_writes = _open_paths(_active_write_candidates(root))
    manifest_path = root / args.manifest
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "delete_requested": bool(args.delete),
        "manifest_requested": bool(args.write_manifest or args.delete),
        "excluded_prefixes": sorted(exclude_parts),
        "n_sidecars": len(sidecars),
        "active_open_write_paths": [path.relative_to(root).as_posix() for path in open_writes],
        "sidecars": [path.relative_to(root).as_posix() for path in sidecars],
    }

    if args.delete and open_writes and not args.allow_active:
        print("[cleanup_appledouble] refused: active download/temp outputs are open")
        for path in open_writes:
            print(path.relative_to(root).as_posix())
        return 2

    if args.write_manifest or args.delete:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    deleted = 0
    if args.delete:
        for path in sidecars:
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
        payload["deleted"] = deleted
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        # Writing the manifest on macOS/external drives can itself create a new
        # AppleDouble sidecar. Remove it so the cleanup operation is idempotent.
        manifest_sidecar = manifest_path.with_name(f"._{manifest_path.name}")
        if manifest_sidecar.exists():
            manifest_sidecar.unlink()
        deleted += _delete_git_object_sidecars(root)
        payload["deleted"] = deleted
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        manifest_sidecar = manifest_path.with_name(f"._{manifest_path.name}")
        if manifest_sidecar.exists():
            manifest_sidecar.unlink()

    manifest_label = manifest_path.relative_to(root) if args.write_manifest or args.delete else "not-written"
    print(
        f"[cleanup_appledouble] sidecars={len(sidecars)} deleted={deleted} "
        f"active_open_writes={len(open_writes)} manifest={manifest_label}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
