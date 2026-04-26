#!/usr/bin/env python3
"""Validate local GridPulse workspace hygiene without touching source files.

This check is intentionally stricter than git status: it guards against local
filesystem state that breaks lint/test runs or consumes hundreds of GB outside
the tracked repo surface.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from cleanup_appledouble import default_exclude_parts, find_sidecars


PRUNE_PARTS = {".venv", ".venv2", "frontend/node_modules", ".git"}
ROOT_INSTALLER_PATTERNS = ("*.dmg",)
ROOT_DOWNLOAD_PATTERNS = ("*.crdownload", "*.partial", "*.download")
ROOT_DATASET_PATTERNS = ("nuplan-*.zip",)
ACTIVE_WRITE_PATTERNS = ("*.crdownload", "*.tmp", "*.parquet.tmp")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_pruned(path: Path, root: Path) -> bool:
    rel = _rel(path, root)
    return any(rel == part or rel.startswith(f"{part}/") for part in PRUNE_PARTS)


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)


def _pid_is_active(pid: int) -> bool:
    return _run(["ps", "-p", str(pid)], cwd=Path("/")).returncode == 0


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip().splitlines()[0]
        return int(text)
    except (IndexError, OSError, UnicodeDecodeError, ValueError):
        return None


def _pid_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.pid") if path.is_file() and not _is_pruned(path, root))


def _stale_pid_files(root: Path) -> list[Path]:
    stale: list[Path] = []
    for path in _pid_files(root):
        pid = _read_pid(path)
        if pid is not None and not _pid_is_active(pid):
            stale.append(path)
    return stale


def _active_write_candidates(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for pattern in ACTIVE_WRITE_PATTERNS:
        candidates.update(path for path in root.rglob(pattern) if path.is_file() and not _is_pruned(path, root))
    return sorted(candidates)


def _open_paths(paths: list[Path]) -> list[Path]:
    if not paths:
        return []
    try:
        completed = subprocess.run(["lsof", *[str(path) for path in paths]], check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return []
    if completed.returncode not in (0, 1):
        return []
    lines = completed.stdout.splitlines()[1:]
    return [path for path in paths if any(str(path) in line for line in lines)]


def _git_ignored(root: Path, path: Path) -> bool:
    return _run(["git", "check-ignore", "-q", "--", _rel(path, root)], cwd=root).returncode == 0


def _git_process_active() -> bool:
    completed = _run(["ps", "ax", "-o", "pid=,comm=,command="], cwd=Path("/"))
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


def _git_temp_packs(root: Path) -> list[Path]:
    pack_dir = root / ".git" / "objects" / "pack"
    if not pack_dir.exists():
        return []
    return sorted(path for path in pack_dir.glob("tmp_pack_*") if path.is_file())


def _root_matches(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(matches)


def _dangling_last_markers(root: Path) -> list[str]:
    dangling: list[str] = []
    run_id_path = root / ".last_nuplan_max_run_id"
    if run_id_path.exists():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
        if run_id:
            expected = [
                root / "data" / "orius_av" / "av" / f"processed_{run_id}",
                root / "reports" / "orius_av" / run_id,
            ]
            if not any(path.exists() for path in expected):
                dangling.append(f".last_nuplan_max_run_id -> {run_id}")
    return dangling


def _print_group(title: str, root: Path, paths: list[Path], *, limit: int = 25) -> None:
    print(f"[validate_workspace_hygiene] {title}: {len(paths)}")
    for path in paths[:limit]:
        print(f"  {_rel(path, root)}")
    if len(paths) > limit:
        print(f"  ... {len(paths) - limit} more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--exclude-active", action="store_true", help="Apply active-run AppleDouble allowlists.")
    parser.add_argument("--allow-active", action="store_true", help="Do not fail on active open downloads/temp files.")
    parser.add_argument("--delete-stale-pids", action="store_true", help="Delete stale PID marker files.")
    parser.add_argument("--allow-git-temp-packs", action="store_true", help="Warn, but do not fail, on tmp git packs.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    blockers: list[str] = []
    warnings: list[str] = []

    sidecars = find_sidecars(root, default_exclude_parts(root) if args.exclude_active else set())
    _print_group("AppleDouble sidecars", root, sidecars)
    if sidecars:
        blockers.append("AppleDouble sidecars remain")

    stale_pids = _stale_pid_files(root)
    if args.delete_stale_pids:
        deleted = 0
        for path in stale_pids:
            try:
                path.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
        print(f"[validate_workspace_hygiene] deleted stale PID files: {deleted}")
        stale_pids = _stale_pid_files(root)
    _print_group("stale PID files", root, stale_pids)
    if stale_pids:
        blockers.append("stale PID files remain")

    active_open = _open_paths(_active_write_candidates(root))
    _print_group("active open download/temp files", root, active_open)
    if active_open and not args.allow_active:
        blockers.append("active download/temp files are still open")

    downloads = _root_matches(root, ROOT_DOWNLOAD_PATTERNS)
    _print_group("root download fragments", root, downloads)
    if downloads:
        blockers.append("root download fragments remain")

    installers = _root_matches(root, ROOT_INSTALLER_PATTERNS)
    _print_group("root installers", root, installers)
    if installers:
        blockers.append("root installers remain")

    dataset_archives = _root_matches(root, ROOT_DATASET_PATTERNS)
    not_ignored = [path for path in dataset_archives if not _git_ignored(root, path)]
    _print_group("root nuPlan archives", root, dataset_archives)
    if not_ignored:
        blockers.append("root nuPlan archives are not ignored")
        _print_group("unignored nuPlan archives", root, not_ignored)

    dangling_markers = _dangling_last_markers(root)
    if dangling_markers:
        warnings.extend(f"dangling marker: {marker}" for marker in dangling_markers)

    temp_packs = _git_temp_packs(root)
    _print_group("git tmp_pack files", root, temp_packs)
    if temp_packs:
        if _git_process_active():
            blockers.append("git temp packs remain and a git process appears active")
        elif args.allow_git_temp_packs:
            warnings.append("git temp packs remain")
        else:
            blockers.append("stale git temp packs remain")

    for warning in warnings:
        print(f"[validate_workspace_hygiene] WARN: {warning}")
    if blockers:
        for blocker in blockers:
            print(f"[validate_workspace_hygiene] FAIL: {blocker}")
        return 1
    print("[validate_workspace_hygiene] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
