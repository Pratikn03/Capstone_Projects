#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REQUIRED_PARTS = [
    "Problem, Principle, and Gap",
    "ORIUS Framework and Theorem Ladder",
    "Witness Domain Evidence",
    "Universalization and Extensions",
    "Submission Boundary",
]


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _collect_tokens(tex: str, pattern: str) -> set[str]:
    return set(re.findall(pattern, tex))


def _check_sections(tex: str, errors: list[str]) -> None:
    parts: list[str] = []
    for line in tex.splitlines():
        stripped = line.strip()
        match = re.match(r"\\part\{([^}]+)\}", stripped)
        if match:
            parts.append(match.group(1))

    if len(parts) < len(REQUIRED_PARTS):
        errors.append("Top-level part count is lower than the required ORIUS monograph structure.")
        return

    for got, exp_prefix in zip(parts[: len(REQUIRED_PARTS)], REQUIRED_PARTS, strict=False):
        if got != exp_prefix:
            errors.append(f"Top-level part mismatch: expected '{exp_prefix}' but found '{got}'.")
            break
    if r"\documentclass[12pt,oneside]{book}" not in tex:
        errors.append("paper.tex must declare the book-class monograph surface.")
    if r"\bibliography{paper/bibliography/orius_monograph}" not in tex:
        errors.append("paper.tex must use the generated ORIUS monograph bibliography.")
    if "\\appendix" not in tex:
        errors.append("Missing \\appendix marker in paper.tex")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify paper manifest paths and LaTeX token resolution")
    parser.add_argument("--manifest", default="paper/manifest.yaml")
    parser.add_argument("--paper", default="paper/paper.tex")
    parser.add_argument("--stats", default="paper/assets/data/metrics_snapshot.json")
    parser.add_argument(
        "--camera-ready",
        action="store_true",
        help="Promote manifest warnings to errors for the camera-ready freeze lane.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    manifest_path = repo_root / args.manifest
    paper_path = repo_root / args.paper
    stats_path = repo_root / args.stats

    errors: list[str] = []
    warnings: list[str] = []

    if not manifest_path.exists():
        print(f"ERROR: manifest missing: {manifest_path}", file=sys.stderr)
        return 1
    if not paper_path.exists():
        print(f"ERROR: paper missing: {paper_path}", file=sys.stderr)
        return 1

    manifest = _load_manifest(manifest_path)
    paper_text = paper_path.read_text(encoding="utf-8")

    # Structural checks
    _check_sections(paper_text, errors)

    # Placeholder hygiene
    if "[[" in paper_text or "]]" in paper_text:
        errors.append("Found unresolved placeholder token markers [[...]] in paper.tex")

    # Token checks
    figure_tokens_used = _collect_tokens(paper_text, r"\\PaperFigure(?:Wide|Hero)?\{([^}]+)\}\{")
    table_tokens_used = _collect_tokens(paper_text, r"\\PaperTableCSV\{([^}]+)\}\{")

    figure_manifest = set((manifest.get("figures") or {}).keys())
    table_manifest = set((manifest.get("tables") or {}).keys())

    missing_fig_tokens = sorted(figure_tokens_used - figure_manifest)
    missing_tbl_tokens = sorted(table_tokens_used - table_manifest)
    if missing_fig_tokens:
        warnings.append(f"Figure tokens used but not defined in manifest: {missing_fig_tokens}")
    if missing_tbl_tokens:
        warnings.append(f"Table tokens used but not defined in manifest: {missing_tbl_tokens}")

    # Path checks for all manifest-mapped files
    for section_name in ("figures", "tables", "data", "configs"):
        for token, entry in (manifest.get(section_name) or {}).items():
            path = entry.get("path")
            if not path:
                errors.append(f"{section_name}.{token} missing path")
                continue
            fpath = repo_root / path
            if not fpath.exists():
                errors.append(f"Missing manifest file: {path}")
                continue
            if fpath.is_file() and fpath.stat().st_size == 0:
                errors.append(f"Empty manifest file: {path}")

    # Check path hygiene (/tmp not allowed in committed stats snapshot)
    if stats_path.exists():
        raw = stats_path.read_text(encoding="utf-8")
        if "/tmp/" in raw:
            errors.append(f"Path hygiene failure: {stats_path} contains /tmp/")
        try:
            json.loads(raw)
        except Exception as exc:
            errors.append(f"Invalid JSON in metrics snapshot: {exc}")
    else:
        warnings.append(f"Metrics snapshot not found: {stats_path}")

    macro_surface_active = bool(figure_tokens_used or table_tokens_used)

    # Ensure every manifest token is either used or explicitly marked optional when
    # the canonical manuscript still uses the legacy manifest macro surface.
    if macro_surface_active:
        optional_tokens = {
            token
            for section in (manifest.get("figures") or {}, manifest.get("tables") or {})
            for token, entry in section.items()
            if (entry or {}).get("optional", False)
        }
        unused = sorted(
            ((figure_manifest | table_manifest) - (figure_tokens_used | table_tokens_used)) - optional_tokens
        )
        if unused:
            warnings.append(f"Unused manifest tokens: {unused}")
    else:
        print("Manifest macro surface inactive in canonical monograph; unused-token audit skipped.")

    if args.camera_ready and warnings:
        errors.extend(f"camera-ready warning promoted to error: {warning}" for warning in warnings)
        warnings = []

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("Verification failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("Manifest and paper checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
