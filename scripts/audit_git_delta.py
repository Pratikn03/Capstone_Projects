#!/usr/bin/env python3
"""Audit Git delta versus a baseline ref for publish traceability."""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DeltaFile:
    path: str
    status: str
    added_lines: int | None
    deleted_lines: int | None
    in_scope: bool
    scope_match: str


def _run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT), text=True).strip()


def _try_run_git(args: list[str]) -> str:
    try:
        return _run_git(args)
    except Exception:
        return ""


def _load_scope_globs(config_path: Path) -> list[str]:
    if not config_path.exists():
        return ["src/orius/**", "services/api/**", "scripts/**", "configs/**", "iot/**"]
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return ["src/orius/**", "services/api/**", "scripts/**", "configs/**", "iot/**"]
    publish = payload.get("publish_audit") if isinstance(payload.get("publish_audit"), dict) else {}
    scope = publish.get("scope") if isinstance(publish.get("scope"), dict) else {}
    includes = scope.get("include") if isinstance(scope.get("include"), list) else None
    return (
        [str(x) for x in includes]
        if includes
        else ["src/orius/**", "services/api/**", "scripts/**", "configs/**", "iot/**"]
    )


def _matches_scope(path: str, patterns: list[str]) -> tuple[bool, str]:
    norm_path = path.replace("\\", "/")
    for patt in patterns:
        patt_norm = str(patt).replace("\\", "/")
        if patt_norm.endswith("/**"):
            prefix = patt_norm[: -len("/**")].rstrip("/")
            if norm_path == prefix or norm_path.startswith(prefix + "/"):
                return True, patt_norm
        if fnmatch.fnmatch(norm_path, patt_norm):
            return True, patt_norm
    return False, "out_of_scope"


def _parse_numstat(lines: str) -> dict[str, tuple[int | None, int | None]]:
    out: dict[str, tuple[int | None, int | None]] = {}
    for raw in lines.splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) < 3:
            continue
        added_raw = parts[0]
        deleted_raw = parts[1]
        path = parts[-1]
        added = None if added_raw == "-" else int(added_raw)
        deleted = None if deleted_raw == "-" else int(deleted_raw)
        out[path] = (added, deleted)
    return out


def _parse_name_status(lines: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in lines.splitlines():
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        path = parts[-1]
        out[path] = status
    return out


def _parse_untracked(status_porcelain: str) -> list[str]:
    paths: list[str] = []
    for raw in status_porcelain.splitlines():
        if raw.startswith("?? "):
            paths.append(raw[3:].strip())
    return paths


def build_delta_report(*, baseline_ref: str, config_path: Path) -> dict[str, Any]:
    baseline_commit = _try_run_git(["rev-parse", baseline_ref])
    if not baseline_commit:
        raise SystemExit(f"Unable to resolve baseline ref: {baseline_ref}")

    scope_globs = _load_scope_globs(config_path)
    head_commit = _try_run_git(["rev-parse", "HEAD"])
    branch = _try_run_git(["rev-parse", "--abbrev-ref", "HEAD"])

    numstat = _parse_numstat(_try_run_git(["diff", "--numstat", "--find-renames", baseline_ref]))
    statuses = _parse_name_status(_try_run_git(["diff", "--name-status", "--find-renames", baseline_ref]))
    untracked = _parse_untracked(_try_run_git(["status", "--porcelain"]))

    all_paths = set(numstat.keys()) | set(statuses.keys())
    files: list[DeltaFile] = []
    for path in sorted(all_paths):
        added, deleted = numstat.get(path, (None, None))
        in_scope, scope_match = _matches_scope(path, scope_globs)
        files.append(
            DeltaFile(
                path=path,
                status=statuses.get(path, "M"),
                added_lines=added,
                deleted_lines=deleted,
                in_scope=in_scope,
                scope_match=scope_match,
            )
        )

    for path in sorted(set(untracked)):
        in_scope, scope_match = _matches_scope(path, scope_globs)
        files.append(
            DeltaFile(
                path=path,
                status="??",
                added_lines=None,
                deleted_lines=None,
                in_scope=in_scope,
                scope_match=scope_match,
            )
        )

    added_total = sum(x.added_lines or 0 for x in files)
    deleted_total = sum(x.deleted_lines or 0 for x in files)
    tracked_files = [x for x in files if x.status != "??"]
    in_scope_files = [x for x in files if x.in_scope]
    out_scope_files = [x for x in files if not x.in_scope]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline_ref": baseline_ref,
        "baseline_commit": baseline_commit,
        "head_commit": head_commit,
        "branch": branch,
        "scope_globs": scope_globs,
        "summary": {
            "total_changed_files": len(files),
            "tracked_changed_files": len(tracked_files),
            "untracked_files": len([x for x in files if x.status == "??"]),
            "in_scope_files": len(in_scope_files),
            "out_of_scope_files": len(out_scope_files),
            "added_lines": int(added_total),
            "deleted_lines": int(deleted_total),
        },
        "files": [
            {
                "path": x.path,
                "status": x.status,
                "added_lines": x.added_lines,
                "deleted_lines": x.deleted_lines,
                "in_scope": x.in_scope,
                "scope_match": x.scope_match,
            }
            for x in files
        ],
    }
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    lines: list[str] = []
    lines.append("# GitHub Delta Audit")
    lines.append("")
    lines.append(f"- Generated at: `{payload.get('generated_at')}`")
    lines.append(f"- Branch: `{payload.get('branch')}`")
    lines.append(f"- Baseline: `{payload.get('baseline_ref')}` (`{payload.get('baseline_commit')}`)")
    lines.append(f"- Head: `{payload.get('head_commit')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Total changed files: `{summary.get('total_changed_files', 0)}`")
    lines.append(f"- In scope files: `{summary.get('in_scope_files', 0)}`")
    lines.append(f"- Out of scope files: `{summary.get('out_of_scope_files', 0)}`")
    lines.append(f"- Added lines: `{summary.get('added_lines', 0)}`")
    lines.append(f"- Deleted lines: `{summary.get('deleted_lines', 0)}`")
    lines.append("")
    lines.append("## File Delta")
    lines.append("| Path | Status | +Lines | -Lines | In Scope | Scope Match |")
    lines.append("|---|---|---:|---:|:---:|---|")
    for row in payload.get("files", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row.get('path')} | {row.get('status')} | {row.get('added_lines')} | {row.get('deleted_lines')} | "
            f"{'yes' if row.get('in_scope') else 'no'} | {row.get('scope_match')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit git delta against baseline ref")
    parser.add_argument("--baseline-ref", default="origin/main")
    parser.add_argument("--out-dir", default="reports/publish")
    parser.add_argument("--scope-config", default="configs/publish_audit.yaml")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.scope_config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    payload = build_delta_report(baseline_ref=str(args.baseline_ref), config_path=config_path)

    json_path = out_dir / "github_delta_report.json"
    md_path = out_dir / "github_delta_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")

    print(json.dumps(payload.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
