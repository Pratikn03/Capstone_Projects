#!/usr/bin/env python3
"""Inventory and classify repository files for cleanup decisions.

This script is intentionally conservative. It classifies files into:

- keep_active_source
- keep_active_generated
- keep_provenance
- legacy_reference
- local_runtime_state
- safe_delete
- ambiguous_review

The goal is not to decide every file from filename alone, but to produce a
repo-grounded keep/delete surface that can be inspected and rerun.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "audit"

LATEX_BUILD_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".lof",
    ".log",
    ".lot",
    ".nav",
    ".out",
    ".snm",
    ".synctex.gz",
    ".toc",
}

ROOT_ACTIVE_SOURCE_FILENAMES = {
    "README.md",
    "LICENSE",
    "Makefile",
    "pyproject.toml",
    "pytest.ini",
    "preamble.tex",
    "DATA.md",
    "PRODUCTION_GUIDE.md",
    "CODEX_IMPLEMENTATION_UPDATE.md",
    "audit.py",
    "restructure_thesis.py",
    "requirements.txt",
    "requirements.lock.txt",
}


@dataclass
class FileRecord:
    path: str
    tracked: bool
    size_bytes: int
    top_level: str
    category: str
    rationale: str


def _git_list(*args: str) -> set[str]:
    try:
        output = subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return set()
    return {line.strip() for line in output.splitlines() if line.strip()}


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file():
            files.append(path)
    return files


def _is_safe_delete_dir(rel_path: str) -> tuple[bool, str]:
    prefixes = {
        ".hypothesis/": "local property-test cache",
        ".pytest_cache/": "local pytest cache",
        "tmp/": "temporary scratch outputs",
        "reports/coverage/": "generated coverage HTML",
    }
    for prefix, rationale in prefixes.items():
        if rel_path.startswith(prefix):
            return True, rationale
    return False, ""


def _classify(rel_path: str, tracked: bool) -> tuple[str, str]:
    path = Path(rel_path)
    top = path.parts[0] if path.parts else "<root>"
    suffix = path.suffix.lower()
    name = path.name

    safe_delete_dir, safe_delete_reason = _is_safe_delete_dir(rel_path)
    if safe_delete_dir:
        return "safe_delete", safe_delete_reason

    if name in {".DS_Store", ".coverage", "coverage.xml", "tsconfig.tsbuildinfo"}:
        return "safe_delete", "local machine or generated coverage/build state"

    if "__pycache__" in path.parts:
        return "safe_delete", "python bytecode cache"

    if suffix in LATEX_BUILD_SUFFIXES:
        return "safe_delete", "rebuildable LaTeX build byproduct"

    if top in {".venv"} or ("node_modules" in path.parts):
        return "local_runtime_state", "environment/vendor dependency tree"

    if len(path.parts) == 1:
        if name in ROOT_ACTIVE_SOURCE_FILENAMES:
            return "keep_active_source", "repo-root control file or authoring surface"
        if suffix in {".py", ".sh", ".md", ".tex", ".toml", ".yml", ".yaml", ".json", ".csv"} and tracked:
            return "keep_active_source", "tracked repo-root script, config, or documentation surface"
        if suffix == ".pdf":
            return "legacy_reference", "compiled PDF retained outside the active package surface"

    if top in {"src", "scripts", "services", "configs", "tests", "deploy", "docker", "iot"}:
        return "keep_active_source", "runtime or verification source code"

    if top == "frontend":
        if "src" in path.parts:
            return "keep_active_source", "frontend application source"
        return "local_runtime_state", "frontend package metadata or local build state"

    if top in {"chapters", "appendices", "frontmatter", "backmatter"}:
        return "keep_active_source", "thesis source-of-truth surface"

    if top == "paper":
        if rel_path in {
            "paper/paper.tex",
            "paper/metrics_manifest.json",
            "paper/claim_matrix.csv",
            "paper/README.md",
            "paper/sync_rules.md",
            "paper/accuracy_audit.md",
            "paper/manifest.yaml",
            "paper/orius_program_manifest.json",
            "paper/paper.docx",
            "paper/paper.pdf",
        }:
            return "keep_active_generated" if rel_path.endswith(
                (".pdf", ".docx")
            ) else "keep_active_source", "paper control surface explicitly referenced by manuscript policy"
        if rel_path.startswith("paper/slides/"):
            return (
                ("keep_active_generated", "active slide deck compiled artifact")
                if suffix == ".pdf"
                else ("keep_active_source", "active slide deck source")
            )
        if rel_path.startswith(
            (
                "paper/monograph/",
                "paper/review/",
                "paper/ieee/",
                "paper/assets/",
                "paper/bibliography/",
                "paper/longform/",
            )
        ):
            return "keep_active_generated", "generator-owned or active derivative manuscript surface"
        if rel_path in {"paper/paper_r1.tex", "paper/paper_r1.pdf"}:
            return (
                "legacy_reference",
                "legacy derivative retained for provenance and referenced by release manifests/tests",
            )
        if rel_path in {
            "paper/paper_single_column_backup.tex",
            "paper/file.md",
            "paper/# Conference-Main R1 Upgrade for GridPul.prompt.md",
        }:
            return (
                "legacy_reference",
                "legacy or working-note manuscript aid not part of canonical control surface",
            )
        return "ambiguous_review", "paper-area file outside the explicit authority list"

    if top == "chapters_merged":
        return (
            "legacy_reference",
            "legacy merged chapter surface still referenced by audit/manuscript-count tooling",
        )

    if top == "reports":
        if rel_path.startswith("reports/publication/"):
            if suffix == ".tar.gz" or name.endswith(".tar.gz"):
                return (
                    "legacy_reference",
                    "frozen package archive duplicated by extracted legacy archive contents",
                )
            if suffix in {".csv", ".json", ".md", ".tex", ".png", ".pdf"}:
                return "keep_active_generated", "tracked publication artifact or matrix"
            return "ambiguous_review", "publication artifact with uncommon type"
        if rel_path.startswith("reports/audit/"):
            return "keep_provenance", "repo audit output used for cleanup and integrity review"
        if rel_path.startswith("reports/legacy_archive/"):
            return "keep_provenance", "explicitly archived provenance bundle"
        if rel_path.startswith(
            (
                "reports/publish/",
                "reports/figures/",
                "reports/eia930/figures/",
                "reports/fault_benchmark/",
                "reports/industrial/figures/",
            )
        ):
            return (
                "keep_provenance",
                "tracked reporting or figure artifact used by publication and audit surfaces",
            )
        if len(path.parts) >= 3 and path.parts[2] == "figures":
            return "keep_provenance", "per-domain report figure artifact retained for auditability"
        if rel_path.startswith("reports/paper") or name.startswith("research_metrics_"):
            return "keep_provenance", "tracked reporting surface retained for publication traceability"
        if rel_path.startswith(
            (
                "reports/runs/",
                "reports/closure_refresh/",
                "reports/camera_ready/",
                "reports/universal_",
                "reports/orius_framework_proof/",
            )
        ):
            return (
                "keep_provenance",
                "run or audit evidence referenced by manifests and reproducibility surfaces",
            )
        if suffix in {".log"}:
            return "safe_delete", "rebuildable log outside the active publication surface"
        return "ambiguous_review", "report surface not clearly active or archived"

    if top == "artifacts":
        if rel_path.startswith(
            (
                "artifacts/runs/",
                "artifacts/models",
                "artifacts/backtests/",
                "artifacts/uncertainty/",
                "artifacts/canonical_runs/",
            )
        ):
            return "keep_provenance", "model/run provenance referenced by release and training surfaces"
        if rel_path.startswith("artifacts/paper"):
            return "keep_provenance", "paper-era locked artifact surface kept for provenance"
        if name == ".DS_Store":
            return "safe_delete", "macOS finder metadata"
        return "ambiguous_review", "artifact surface outside the main provenance paths"

    if top == "data":
        if (
            rel_path.endswith("PLACE_REAL_AV_DATA_HERE.md")
            or rel_path.endswith("PLACE_REAL_NAVIGATION_DATA_HERE.md")
            or rel_path.endswith("PLACE_REAL_AEROSPACE_DATA_HERE.md")
        ):
            return "keep_active_source", "tracked raw-data contract placeholder"
        return "keep_provenance", "data or data manifest surface"

    if top in {"docs", "orius-plan"}:
        return "keep_active_source", "governance, workflow, or planning documentation"

    if top in {".idea", ".claude"}:
        return "local_runtime_state", "local editor/assistant configuration"

    if rel_path == "orius_battery_409page_figures_upgraded_main.pdf":
        return "legacy_reference", "compiled thesis PDF regenerated from tracked LaTeX sources"

    return "ambiguous_review", "no explicit rule matched"


def _write_csv(path: Path, rows: list[FileRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "tracked", "size_bytes", "top_level", "category", "rationale"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_markdown(path: Path, rows: list[FileRecord], summary: dict[str, object]) -> None:
    by_category: defaultdict[str, list[FileRecord]] = defaultdict(list)
    for row in rows:
        by_category[row.category].append(row)

    lines = [
        "# Repo File Inventory",
        "",
        f"- Total files inventoried: {summary['files_total']}",
        f"- Total bytes inventoried: {summary['bytes_total']}",
        "",
        "## Category Summary",
        "",
        "| Category | Files | Bytes |",
        "|---|---:|---:|",
    ]
    for category, info in summary["category_summary"].items():
        lines.append(f"| {category} | {info['files']} | {info['bytes']} |")

    for category in [
        "safe_delete",
        "legacy_reference",
        "ambiguous_review",
        "local_runtime_state",
    ]:
        lines.extend(
            [
                "",
                f"## {category}",
                "",
                "| Path | Size | Rationale |",
                "|---|---:|---|",
            ]
        )
        for row in sorted(by_category.get(category, []), key=lambda item: (-item.size_bytes, item.path))[
            :120
        ]:
            lines.append(f"| `{row.path}` | {row.size_bytes} | {row.rationale} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_inventory() -> tuple[list[FileRecord], dict[str, object]]:
    tracked_files = _git_list("ls-files")
    files = _iter_files(REPO_ROOT)
    rows: list[FileRecord] = []
    category_counter: Counter[str] = Counter()
    bytes_by_category: Counter[str] = Counter()

    for path in files:
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        tracked = rel_path in tracked_files
        category, rationale = _classify(rel_path, tracked)
        record = FileRecord(
            path=rel_path,
            tracked=tracked,
            size_bytes=path.stat().st_size,
            top_level=path.parts[0] if path.parts else "<root>",
            category=category,
            rationale=rationale,
        )
        rows.append(record)
        category_counter[category] += 1
        bytes_by_category[category] += record.size_bytes

    summary = {
        "files_total": len(rows),
        "bytes_total": sum(row.size_bytes for row in rows),
        "category_summary": {
            category: {
                "files": category_counter[category],
                "bytes": bytes_by_category[category],
            }
            for category in sorted(category_counter)
        },
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory and classify repository files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows, summary = build_inventory()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_dir / "repo_file_inventory.csv"
    json_path = args.out_dir / "repo_file_inventory.json"
    md_path = args.out_dir / "repo_file_inventory.md"

    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "rows": [asdict(row) for row in rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(md_path, rows, summary)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
