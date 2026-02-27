#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

EXPECTED_SECTIONS_PREFIX = [
    "Introduction",
    "Background and Related Work",
    "Problem Formulation and Safety Definitions",
    "System Overview: GridPulse Execution Pipeline",
    "Forecasting and Regime-Aware CQR Uncertainty",
    "Telemetry Reliability Scoring and Drift Detection",
    "Dispatch Optimization: Robust and Risk-Aware Formulations",
    "DC\\textsuperscript{3}S Safety Shield",
    "Benchmark and Experimental Protocol",
    "Results",
    "Discussion, Limitations, and Conclusion",
]

EXPECTED_APPENDIX_PREFIX = [
    "Appendix A:",
    "Appendix B:",
    "Appendix C:",
    "Appendix D:",
    "Appendix E:",
]


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _collect_tokens(tex: str, pattern: str) -> set[str]:
    return set(re.findall(pattern, tex))


def _check_sections(tex: str, errors: list[str]) -> None:
    sections: list[str] = []
    for line in tex.splitlines():
        stripped = line.strip()
        if stripped.startswith("\\section{") and stripped.endswith("}"):
            sections.append(stripped[len("\\section{") : -1])

    if len(sections) < len(EXPECTED_SECTIONS_PREFIX):
        errors.append("Top-level section count is lower than required 1-11 structure.")
        return

    for got, exp_prefix in zip(sections[: len(EXPECTED_SECTIONS_PREFIX)], EXPECTED_SECTIONS_PREFIX):
        if not got.startswith(exp_prefix):
            errors.append(
                f"Top-level section mismatch: expected prefix '{exp_prefix}' but found '{got}'."
            )
            break

    appendices = [s for s in sections if s.startswith("Appendix ")]
    if len(appendices) < len(EXPECTED_APPENDIX_PREFIX):
        errors.append("Missing one or more required appendices (A-E).")
        return
    for got, exp_prefix in zip(appendices[: len(EXPECTED_APPENDIX_PREFIX)], EXPECTED_APPENDIX_PREFIX):
        if not got.startswith(exp_prefix):
            errors.append(
                f"Appendix ordering/title mismatch: expected prefix '{exp_prefix}' but found '{got}'."
            )
            break


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify paper manifest paths and LaTeX token resolution")
    parser.add_argument("--manifest", default="paper/manifest.yaml")
    parser.add_argument("--paper", default="paper/paper.tex")
    parser.add_argument("--stats", default="paper/assets/data/metrics_snapshot.json")
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
    figure_tokens_used = _collect_tokens(paper_text, r"\\PaperFigure\{([^}]+)\}\{")
    table_tokens_used = _collect_tokens(paper_text, r"\\PaperTableCSV\{([^}]+)\}\{")

    figure_manifest = set((manifest.get("figures") or {}).keys())
    table_manifest = set((manifest.get("tables") or {}).keys())

    missing_fig_tokens = sorted(figure_tokens_used - figure_manifest)
    missing_tbl_tokens = sorted(table_tokens_used - table_manifest)
    if missing_fig_tokens:
        errors.append(f"Figure tokens used but not defined in manifest: {missing_fig_tokens}")
    if missing_tbl_tokens:
        errors.append(f"Table tokens used but not defined in manifest: {missing_tbl_tokens}")

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

    # Ensure every manifest token is either used or explicitly marked optional
    optional_tokens = set(
        token
        for section in (manifest.get("figures") or {}, manifest.get("tables") or {})
        for token, entry in section.items()
        if (entry or {}).get("optional", False)
    )
    unused = sorted(((figure_manifest | table_manifest) - (figure_tokens_used | table_tokens_used)) - optional_tokens)
    if unused:
        warnings.append(f"Unused manifest tokens: {unused}")

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
