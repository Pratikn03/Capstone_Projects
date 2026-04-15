#!/usr/bin/env python3
"""Verify camera-ready LaTeX logs with explicit layout-warning waivers."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAIVERS = REPO_ROOT / "paper" / "camera_ready_warning_waivers.yaml"
DEFAULT_LOGS = [
    REPO_ROOT / "paper" / "paper.log",
    REPO_ROOT / "paper" / "review" / "orius_review_dossier.log",
    REPO_ROOT / "paper" / "ieee" / "orius_ieee_main.log",
    REPO_ROOT / "paper" / "ieee" / "orius_ieee_appendix.log",
    REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_main.log",
    REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_appendix_a.log",
    REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_appendix_b.log",
]

BLOCKING_PATTERNS = {
    "undefined_reference": re.compile(r"LaTeX Warning: Reference `[^`]+` .* undefined"),
    "undefined_citation": re.compile(r"Package natbib Warning: Citation `[^`]+` .* undefined"),
    "undefined_refs_summary": re.compile(r"There were undefined references"),
    "undefined_citations_summary": re.compile(r"There were undefined citations"),
    "citation_rerun": re.compile(r"Citation\(s\) may have changed"),
    "label_rerun": re.compile(r"Label\(s\) may have changed"),
    "duplicate_destination": re.compile(r"duplicate ignored"),
}

LAYOUT_PATTERNS = [
    re.compile(r"^Overfull \\hbox .*"),
    re.compile(r"^Underfull \\hbox .*"),
    re.compile(r"^Package microtype Warning: .*"),
    re.compile(r"^LaTeX Warning: `h' float specifier changed to `ht'\.$"),
]


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _load_waivers(path: Path) -> dict[str, list[re.Pattern[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"camera-ready waiver file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    logs = payload.get("logs", {})
    compiled: dict[str, list[re.Pattern[str]]] = {}
    for log_name, entries in logs.items():
        regexes: list[re.Pattern[str]] = []
        for entry in entries or []:
            pattern = entry if isinstance(entry, str) else entry.get("pattern")
            if not pattern:
                continue
            regexes.append(re.compile(pattern))
        compiled[log_name] = regexes
    return compiled


def _matching_patterns(line: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(line) for pattern in patterns)


def _iter_layout_lines(text: str) -> list[str]:
    matches: list[str] = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in LAYOUT_PATTERNS):
            matches.append(line)
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify camera-ready LaTeX logs with warning waivers.")
    parser.add_argument(
        "--waivers",
        type=Path,
        default=DEFAULT_WAIVERS,
        help="YAML file containing allowed layout-warning regexes per log.",
    )
    parser.add_argument(
        "--log",
        dest="logs",
        action="append",
        type=Path,
        help="Specific log path(s) to verify. Defaults to dissertation, review, and IEEE logs.",
    )
    args = parser.parse_args()

    waivers = _load_waivers(args.waivers)
    logs = args.logs or DEFAULT_LOGS

    failures: list[str] = []
    summary_lines: list[str] = []
    for log_path in logs:
        if log_path.is_absolute():
            resolved = log_path
            rel = _relative(resolved)
        else:
            cwd_candidate = Path.cwd() / log_path
            if cwd_candidate.exists():
                resolved = cwd_candidate
                rel = log_path.as_posix()
            else:
                resolved = REPO_ROOT / log_path
                rel = _relative(resolved)
        if not resolved.exists():
            failures.append(f"{rel}: missing log file")
            continue

        text = resolved.read_text(encoding="utf-8", errors="ignore")
        log_failures: list[str] = []
        for name, pattern in BLOCKING_PATTERNS.items():
            if pattern.search(text):
                log_failures.append(f"{name}: {pattern.pattern}")

        waived_patterns = waivers.get(rel, waivers.get(resolved.name, []))
        unwaived_layout = [
            line for line in _iter_layout_lines(text) if not _matching_patterns(line, waived_patterns)
        ]

        if log_failures:
            failures.extend(f"{rel}: blocking warning `{failure}`" for failure in log_failures)
        if unwaived_layout:
            failures.extend(f"{rel}: unwaived layout warning `{line}`" for line in unwaived_layout)

        summary_lines.append(
            f"{rel}: blocking={len(log_failures)} waived_layout={len(_iter_layout_lines(text)) - len(unwaived_layout)} unwaived_layout={len(unwaived_layout)}"
        )

    print("Camera-ready log summary:")
    for line in summary_lines:
        print(f"  - {line}")

    if failures:
        print("Camera-ready log verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("Camera-ready log verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
