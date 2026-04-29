#!/usr/bin/env python3
"""Remove macOS AppleDouble sidecar files without touching active outputs.

The script writes a manifest before deletion so cleanup is auditable. By
default it skips known active freeze outputs and any currently running nuPlan
pipeline output directories detected from the process table.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
import subprocess


DEFAULT_ACTIVE_RELEASE = "PREDEPLOY_MAX_BG_20260422T122751Z"
DEFAULT_PRUNE_PARTS = {
    ".git",
    ".mypy_cache",
    ".playwright-mcp",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv2",
    "frontend/.next",
    "frontend/node_modules",
}
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
TRAIN_DIR_PATH_FLAGS = {
    "--artifacts-dir",
    "--reports-dir",
    "--uncertainty-artifacts-dir",
    "--backtests-dir",
}
TRAIN_FILE_PATH_FLAGS = {
    "--walk-forward-report",
    "--validation-report",
    "--data-manifest-output",
}
ACTIVE_RELEASE_COMMANDS = ("run_three_domain_offline_freeze.py", "train_dataset.py")


def _repo_relative_prefix(root: Path, raw_path: str) -> str | None:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve(strict=False).relative_to(root).as_posix()
    except ValueError:
        return None


def _flag_value(tokens: list[str], flag: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _release_excludes(release_id: str) -> set[str]:
    release_id = release_id.strip()
    if not release_id:
        return set()
    domains = ("de", "av", "healthcare")
    excludes = {f"reports/predeployment_freeze/{release_id}"}
    excludes.update(f"artifacts/runs/{domain}/{release_id}" for domain in domains)
    excludes.update(f"reports/runs/{domain}/{release_id}" for domain in domains)
    return excludes


def _add_path_exclude(root: Path, excludes: set[str], raw_path: str, *, file_path: bool = False) -> None:
    path = Path(raw_path).expanduser()
    if file_path:
        path = path.parent
    rel = _repo_relative_prefix(root, str(path))
    if rel:
        excludes.add(rel)


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
    active_commands = (
        "run_battery_av_pipeline.py",
        "build_nuplan_av_surface.py",
        "orius.forecasting.train",
        *ACTIVE_RELEASE_COMMANDS,
    )
    for line in completed.stdout.splitlines():
        if not any(command in line for command in active_commands):
            continue
        if "run_battery_av_pipeline.py" in line and "--av-source nuplan" not in line:
            continue
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if any(command in line for command in ACTIVE_RELEASE_COMMANDS):
            release_id = _flag_value(tokens, "--release-id") or _flag_value(tokens, "--run-id")
            if release_id:
                excludes.update(_release_excludes(release_id))
        for index, token in enumerate(tokens):
            flag = token.split("=", 1)[0]
            if flag not in NUPLAN_PIPELINE_PATH_FLAGS | TRAIN_DIR_PATH_FLAGS | TRAIN_FILE_PATH_FLAGS:
                continue
            value = token.split("=", 1)[1] if "=" in token else None
            if value is None and index + 1 < len(tokens):
                value = tokens[index + 1]
            if value:
                _add_path_exclude(root, excludes, value, file_path=flag in TRAIN_FILE_PATH_FLAGS)
    return excludes


def default_exclude_parts(root: Path) -> set[str]:
    exclude_parts = set(DEFAULT_EXCLUDE_PARTS)
    exclude_parts.update(_active_nuplan_pipeline_excludes(root))
    return exclude_parts


def _is_excluded(path: Path, root: Path, exclude_parts: set[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(rel == item or rel.startswith(f"{item}/") for item in exclude_parts)


def _is_pruned(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(rel == item or rel.startswith(f"{item}/") for item in DEFAULT_PRUNE_PARTS)


def _walk_matching_files(root: Path, patterns: tuple[str, ...], exclude_parts: set[str] | None = None) -> list[Path]:
    exclude_parts = exclude_parts or set()
    matches: list[Path] = []
    for current_root, dirs, files in os.walk(root):
        current = Path(current_root)
        dirs[:] = [
            name
            for name in dirs
            if not _is_pruned(current / name, root) and not _is_excluded(current / name, root, exclude_parts)
        ]
        for name in files:
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                path = current / name
                if not _is_excluded(path, root, exclude_parts):
                    matches.append(path)
    return sorted(matches)


def find_sidecars(root: Path, exclude_parts: set[str]) -> list[Path]:
    return _walk_matching_files(root, ("._*",), exclude_parts)


def _active_write_candidates(root: Path) -> list[Path]:
    return _walk_matching_files(root, ACTIVE_WRITE_PATTERNS)


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


def _git_process_active() -> bool:
    try:
        completed = subprocess.run(
            ["ps", "ax", "-o", "pid=,comm=,command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    if completed.returncode != 0:
        return False

    git_names = {
        "git",
        "git-lfs",
        "git-index-pack",
        "git-pack-objects",
        "index-pack",
        "pack-objects",
    }
    for line in completed.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        command_name = Path(parts[1]).name
        argv_name = Path(parts[2].split()[0]).name if len(parts) > 2 and parts[2].split() else ""
        if command_name in git_names or argv_name in git_names:
            return True
    return False


def _has_git_sidecars(root: Path) -> bool:
    git_root = root / ".git"
    if not git_root.exists():
        return False
    for current_root, _dirs, files in os.walk(git_root):
        if any(name.startswith("._") for name in files):
            return True
    return False


def _delete_git_object_sidecars(root: Path) -> int:
    """Remove Git-object AppleDouble files that macOS can recreate late."""
    git_root = root / ".git" / "objects"
    if not git_root.exists():
        return 0
    try:
        subprocess.run(["dot_clean", "-m", str(git_root)], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        pass
    deleted = 0
    for _ in range(3):
        sidecars = [path for path in git_root.rglob("._*") if path.is_file()]
        if not sidecars:
            break
        for path in sidecars:
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

    if args.delete and _has_git_sidecars(root) and _git_process_active():
        print("[cleanup_appledouble] refused: Git AppleDouble files exist while a git process is active")
        return 3

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
    git_sidecars = [path for path in sidecars if path.relative_to(root).as_posix().startswith(".git/")]
    if args.delete and git_sidecars and _git_process_active():
        print("[cleanup_appledouble] refused: Git AppleDouble files exist while a git process is active")
        for path in git_sidecars[:25]:
            print(path.relative_to(root).as_posix())
        if len(git_sidecars) > 25:
            print(f"... {len(git_sidecars) - 25} more")
        return 3

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
