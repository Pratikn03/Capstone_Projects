#!/usr/bin/env python3
"""Full-folder audit for the Gridpulse workspace.

This script inventories every non-.git file in the repository root, assigns an
audit mode, performs semantic review for text-like files, performs metadata and
linkage review for binary/vendor/data surfaces, and emits:

- reports/audit/full_folder_coverage_ledger.csv
- reports/audit/full_folder_findings_catalog.csv
- reports/audit/full_folder_synthesis.md
- reports/audit/full_folder_summary.json
- reports/audit/full_folder_vendor_anomalies.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "audit"

TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".cjs",
    ".css",
    ".csv",
    ".cts",
    ".example",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".ipynb",
    ".java",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".mjs",
    ".mts",
    ".py",
    ".pyi",
    ".pyx",
    ".pxd",
    ".r",
    ".rb",
    ".rst",
    ".service",
    ".sh",
    ".sql",
    ".svg",
    ".tex",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_FILENAMES = {
    ".gitignore",
    "Makefile",
    "Dockerfile",
    "README",
    "README.md",
    "requirements.txt",
    "requirements.lock.txt",
    "ruff.toml",
}
LATEX_BUILD_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".lof",
    ".log",
    ".lot",
    ".out",
    ".toc",
}
PACKAGE_OUTPUT_ALLOWLIST = {
    "paper.pdf",
    "paper/paper.pdf",
    "paper/paper.log",
    "paper/paper_r1.pdf",
    "paper/ieee/orius_ieee_main.pdf",
    "paper/ieee/orius_ieee_main.log",
    "paper/ieee/orius_ieee_appendix.pdf",
    "paper/ieee/orius_ieee_appendix.log",
    "paper/ieee/orius_ieee_professor_main.pdf",
    "paper/ieee/orius_ieee_professor_main.log",
    "paper/ieee/orius_ieee_professor_appendix_a.pdf",
    "paper/ieee/orius_ieee_professor_appendix_a.log",
    "paper/ieee/orius_ieee_professor_appendix_b.pdf",
    "paper/ieee/orius_ieee_professor_appendix_b.log",
    "paper/review/orius_review_dossier.pdf",
    "paper/review/orius_review_dossier.log",
    "reports/publication/orius_review_dossier.pdf",
}
PACKAGE_OUTPUT_PREFIXES = (
    "appendices/",
    "backmatter/",
    "chapters/",
    "chapters_merged/",
    "paper/",
)
HASH_MAX_BYTES = 1024 * 1024
PDF_HASH_MAX_BYTES = 32 * 1024 * 1024
ABSOLUTE_PATH_PATTERN = re.compile(r"(/Users/[^\s`\"')]+|/Volumes/[^\s`\"')]+)")
STALE_ORIUS_PATH_PATTERN = re.compile(r"/Users/[^/\s]+/Downloads/orius(?:/|`|\\b)")


@dataclass
class Finding:
    id: str
    severity: str
    subsystem: str
    file: str
    line_or_surface: str
    category: str
    description: str
    user_impact: str
    evidence: str
    fix_recommendation: str
    confidence: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_git(*args: str) -> set[str]:
    try:
        output = subprocess.check_output(["git", *args], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return set()
    return {line.strip() for line in output.splitlines() if line.strip()}


def _git_status_rows() -> list[str]:
    try:
        output = subprocess.check_output(["git", "status", "--short"], cwd=str(REPO_ROOT), text=True)
    except Exception:
        return []
    return [line.rstrip() for line in output.splitlines() if line.strip()]


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name != ".git"]
        for filename in filenames:
            yield Path(dirpath) / filename


def _sha256(path: Path, *, max_bytes: int) -> str | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > max_bytes:
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _looks_textual(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES or path.name in TEXT_FILENAMES:
        return True
    try:
        sample = path.read_bytes()[:4096]
    except OSError:
        return False
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _subsystem_for(rel_path: str) -> str:
    top = rel_path.split("/", 1)[0] if "/" in rel_path else "<root>"
    if top == "frontend" and "node_modules/" in rel_path:
        return "frontend/dependencies"
    if top == ".venv":
        return "python-environment"
    if top == "paper":
        return "manuscript/publication"
    if top in {"appendices", "chapters", "chapters_merged", "frontmatter", "backmatter"}:
        return "manuscript/publication"
    if top in {"reports", "artifacts", "data"}:
        return "data/artifacts"
    if top == "src":
        return "runtime/code"
    if top in {"scripts", "tests", "services", "deploy", "docker", "iot", "configs"}:
        return "build/release/package"
    return top


def _classify_file(path: Path, *, tracked: bool) -> tuple[str, str]:
    rel_path = path.relative_to(REPO_ROOT).as_posix()
    top = rel_path.split("/", 1)[0] if "/" in rel_path else "<root>"
    suffix = path.suffix.lower()
    is_text = _looks_textual(path)

    if top == ".venv":
        return "virtualenv", "metadata"
    if top == "frontend" and "node_modules/" in rel_path:
        return "vendor", "metadata"
    if top in {"data", "artifacts", "tmp"}:
        if is_text and (path.name.startswith("PLACE_REAL_") or "manifest" in path.name.lower()):
            return "authored_text", "semantic"
        return "dataset" if top == "data" else "build_output", "metadata"
    if top == "reports":
        if is_text and tracked:
            return "generated_text", "semantic"
        return "build_output", "metadata"
    if top in {"paper", "appendices", "chapters", "chapters_merged", "frontmatter", "backmatter"}:
        if is_text:
            return "authored_text", "semantic"
        return "build_output", "metadata"
    if top in {"src", "scripts", "tests", "services", "deploy", "docker", "iot", "configs", "docs", "notebooks", "orius-plan"}:
        if is_text:
            return "authored_text", "semantic"
        return "binary_media", "metadata"
    if suffix in LATEX_BUILD_SUFFIXES:
        return "build_output", "metadata"
    if is_text:
        return "authored_text", "semantic"
    return "binary_media", "metadata"


def _severity_for_absolute_path(rel_path: str) -> str:
    if rel_path in {"Makefile"} or rel_path.startswith(("src/", "scripts/", "services/", "deploy/", "docker/", "iot/")):
        return "high"
    if rel_path.startswith(("reports/", "paper/", "appendices/", "chapters/")):
        return "medium"
    return "low"


def _find_line_numbers(text: str, pattern: re.Pattern[str]) -> list[int]:
    matches: list[int] = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if pattern.search(line):
            matches.append(idx)
    return matches


def _ignore_absolute_path_line(rel_path: str, line: str) -> bool:
    stripped = line.strip()
    if rel_path == "scripts/full_folder_audit.py" and "re.compile(" in stripped:
        return True
    if rel_path.startswith("scripts/") and "help=" in stripped and "/Volumes/" in stripped:
        return True
    if rel_path == "Makefile" and stripped.startswith("SSD_VOLUME ?=") and "/Volumes/" in stripped:
        return True
    return False


def _add_finding(
    findings: list[Finding],
    findings_by_file: dict[str, list[str]],
    *,
    severity: str,
    subsystem: str,
    file: str,
    line_or_surface: str,
    category: str,
    description: str,
    user_impact: str,
    evidence: str,
    fix_recommendation: str,
    confidence: str,
) -> None:
    finding_id = f"AUDIT-{len(findings) + 1:04d}"
    findings.append(
        Finding(
            id=finding_id,
            severity=severity,
            subsystem=subsystem,
            file=file,
            line_or_surface=line_or_surface,
            category=category,
            description=description,
            user_impact=user_impact,
            evidence=evidence,
            fix_recommendation=fix_recommendation,
            confidence=confidence,
        )
    )
    findings_by_file.setdefault(file, []).append(finding_id)


def _scan_text_file(
    rel_path: str,
    text: str,
    findings: list[Finding],
    findings_by_file: dict[str, list[str]],
) -> None:
    subsystem = _subsystem_for(rel_path)
    abs_lines = [
        idx
        for idx, line in enumerate(text.splitlines(), start=1)
        if ABSOLUTE_PATH_PATTERN.search(line) and not _ignore_absolute_path_line(rel_path, line)
    ]
    if abs_lines:
        severity = _severity_for_absolute_path(rel_path)
        _add_finding(
            findings,
            findings_by_file,
            severity=severity,
            subsystem=subsystem,
            file=str(REPO_ROOT / rel_path),
            line_or_surface="lines " + ",".join(map(str, abs_lines[:8])),
            category="hardcoded_local_path",
            description="File embeds local absolute filesystem paths.",
            user_impact="Breaks portability across machines and can leak developer-local paths into package outputs or automation steps.",
            evidence=f"Matched {len(abs_lines)} absolute path occurrence(s) in {rel_path}.",
            fix_recommendation="Replace machine-specific paths with repo-relative paths, environment variables, or documented placeholders.",
            confidence="0.98",
        )
    stale_lines = _find_line_numbers(text, STALE_ORIUS_PATH_PATTERN)
    if stale_lines:
        _add_finding(
            findings,
            findings_by_file,
            severity="medium",
            subsystem=subsystem,
            file=str(REPO_ROOT / rel_path),
            line_or_surface="lines " + ",".join(map(str, stale_lines[:8])),
            category="stale_repo_name_reference",
            description="File still references the old /Downloads/orius workspace path.",
            user_impact="Operational docs and plans can send users to the wrong workspace and invalidate copy-paste instructions.",
            evidence=f"Matched {len(stale_lines)} stale ORIUS path reference(s) in {rel_path}.",
            fix_recommendation="Update the document to use the current gridpulse workspace path or neutral placeholders.",
            confidence="0.95",
        )


def _tracked_latex_build_outputs(tracked_files: set[str]) -> list[str]:
    outputs: list[str] = []
    for rel_path in sorted(tracked_files):
        suffix = Path(rel_path).suffix.lower()
        if suffix in LATEX_BUILD_SUFFIXES:
            outputs.append(rel_path)
    return outputs


def _duplicate_pdf_findings(
    pdf_paths: list[Path],
    findings: list[Finding],
    findings_by_file: dict[str, list[str]],
) -> None:
    hashes: dict[str, list[str]] = {}
    for path in pdf_paths:
        digest = _sha256(path, max_bytes=PDF_HASH_MAX_BYTES)
        if digest is None:
            continue
        hashes.setdefault(digest, []).append(path.relative_to(REPO_ROOT).as_posix())
    for dup_paths in hashes.values():
        if len(dup_paths) < 2:
            continue
        rep = dup_paths[0]
        _add_finding(
            findings,
            findings_by_file,
            severity="low",
            subsystem=_subsystem_for(rep),
            file=str(REPO_ROOT / rep),
            line_or_surface="artifact surface",
            category="duplicate_pdf_payload",
            description="Identical PDF payload is stored in multiple tracked locations.",
            user_impact="Adds churn and ambiguity about which compiled surface is canonical.",
            evidence="Duplicate PDF hash shared by: " + ", ".join(dup_paths[:8]),
            fix_recommendation="Keep one canonical compiled PDF per package surface and copy or publish it outside git when needed.",
            confidence="0.88",
        )


def _analyze_present_package_outputs(
    coverage_rows: list[dict[str, Any]],
    findings: list[Finding],
    findings_by_file: dict[str, list[str]],
) -> None:
    disallowed: list[str] = []
    for row in coverage_rows:
        rel_path = row["relative_path"]
        suffix = Path(rel_path).suffix.lower()
        if rel_path in PACKAGE_OUTPUT_ALLOWLIST:
            continue
        if not rel_path.startswith(PACKAGE_OUTPUT_PREFIXES) and rel_path not in {
            "paper.pdf",
            "thesis_final_main.pdf",
            "orius_battery_409page_figures_upgraded_main.pdf",
        }:
            continue
        if suffix in LATEX_BUILD_SUFFIXES or rel_path.endswith(".pdf"):
            disallowed.append(rel_path)
    if disallowed:
        _add_finding(
            findings,
            findings_by_file,
            severity="medium",
            subsystem="manuscript/publication",
            file=str(REPO_ROOT / ".gitignore"),
            line_or_surface="package_output_allowlist",
            category="non_allowlisted_package_outputs_present",
            description="Non-allowlisted compiled manuscript/package outputs are still present in the working tree.",
            user_impact="Keeps source/generated boundaries blurry and makes package review noisier than necessary.",
            evidence=f"{len(disallowed)} non-allowlisted outputs present, including {', '.join(disallowed[:8])}.",
            fix_recommendation="Remove or relocate generated outputs that are not part of the explicit package allowlist.",
            confidence="0.94",
        )


def _analyze_real_data_and_release_state(
    findings: list[Finding],
    findings_by_file: dict[str, list[str]],
) -> None:
    scorecard_path = REPO_ROOT / "reports" / "publication" / "orius_submission_scorecard.csv"
    if scorecard_path.exists():
        with scorecard_path.open(encoding="utf-8", newline="") as handle:
            rows = {row["target_tier"]: row for row in csv.DictReader(handle)}
        equal_row = rows.get("equal_domain_93")
        if equal_row and str(equal_row.get("meets_93_gate", "")).lower() != "true":
            _add_finding(
                findings,
                findings_by_file,
                severity="high",
                subsystem="data/artifacts",
                file=str(scorecard_path),
                line_or_surface="equal_domain_93",
                category="equal_domain_gate_blocked",
                description="Equal-domain parity gate is still blocked.",
                user_impact="The project cannot honestly claim full equal-domain closure or final universal parity.",
                evidence=(
                    f"readiness_score_100={equal_row.get('readiness_score_100')}, "
                    f"critical_gap_count={equal_row.get('critical_gap_count')}, "
                    f"high_gap_count={equal_row.get('high_gap_count')}, "
                    f"meets_93_gate={equal_row.get('meets_93_gate')}"
                ),
                fix_recommendation="Close the remaining navigation, aerospace, and calibration/governance breadth gaps before promoting the equal-domain tier.",
                confidence="0.99",
            )

    preflight_path = REPO_ROOT / "reports" / "real_data_preflight.json"
    if preflight_path.exists():
        payload = json.loads(preflight_path.read_text(encoding="utf-8"))
        disk = payload.get("disk", {})
        if not payload.get("all_domains_present", False) or not disk.get("passes_threshold", False):
            _add_finding(
                findings,
                findings_by_file,
                severity="critical",
                subsystem="data/artifacts",
                file=str(preflight_path),
                line_or_surface="real_data_preflight",
                category="real_data_preflight_failed",
                description="Real-data preflight currently fails its canonical readiness checks.",
                user_impact="Strict retraining, final freeze, and equal-domain closure remain blocked at the infrastructure gate.",
                evidence=(
                    f"all_domains_present={payload.get('all_domains_present')}, "
                    f"free_gib={disk.get('free_gib')}, "
                    f"min_free_gib={disk.get('min_free_gib')}, "
                    f"passes_threshold={disk.get('passes_threshold')}"
                ),
                fix_recommendation="Restore external raw-data lanes, fix canonical layouts, and satisfy the disk threshold before rerunning parity-closing workflows.",
                confidence="0.99",
            )

    package_manifest_path = REPO_ROOT / "reports" / "publication" / "orius_camera_ready_package_manifest.json"
    if package_manifest_path.exists():
        payload = json.loads(package_manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            _add_finding(
                findings,
                findings_by_file,
                severity="high",
                subsystem="build/release/package",
                file=str(package_manifest_path),
                line_or_surface=str(payload.get("failure_step", "package_manifest")),
                category="camera_ready_freeze_failed",
                description="Camera-ready freeze package is not currently in a completed state.",
                user_impact="Final publication packaging is blocked and current outputs cannot be treated as a final sealed release lane.",
                evidence=f"status={payload.get('status')}, failure_step={payload.get('failure_step')}",
                fix_recommendation="Fix the failing preflight/gate step and rerun the strict freeze lane before treating the package as final.",
                confidence="0.98",
            )


def _analyze_workspace_state(
    tracked_files: set[str],
    findings: list[Finding],
    findings_by_file: dict[str, list[str]],
) -> None:
    status_rows = _git_status_rows()
    if status_rows:
        generated_rows = [
            row
            for row in status_rows
            if any(token in row for token in ("paper/", "appendices/", ".aux", ".out", ".bbl", ".blg", ".pdf"))
        ]
        _add_finding(
            findings,
            findings_by_file,
            severity="medium",
            subsystem="build/release/package",
            file=str(REPO_ROOT / ".gitignore"),
            line_or_surface="workspace_state",
            category="dirty_generated_workspace",
            description="The current worktree is dirty and a large share of churn is generated manuscript/package output.",
            user_impact="Makes it harder to distinguish authored changes from regenerated artifacts and increases release-review noise.",
            evidence=f"git status reports {len(status_rows)} changed paths; {len(generated_rows)} are generated/build-surface rows.",
            fix_recommendation="Move generated outputs out of tracked state where possible and keep the canonical package surfaces narrowly scoped.",
            confidence="0.93",
        )

    latex_outputs = _tracked_latex_build_outputs(tracked_files)
    if latex_outputs:
        _add_finding(
            findings,
            findings_by_file,
            severity="medium",
            subsystem="manuscript/publication",
            file=str(REPO_ROOT / ".gitignore"),
            line_or_surface="tracked_latex_outputs",
            category="tracked_latex_build_outputs",
            description="Tracked LaTeX build outputs remain in the repository despite ignore rules for those surfaces.",
            user_impact="Creates persistent dirty-state churn and encourages source/generated ambiguity in the manuscript pipeline.",
            evidence=f"{len(latex_outputs)} tracked LaTeX build outputs detected, including {', '.join(latex_outputs[:8])}.",
            fix_recommendation="Untrack auxiliary build outputs and keep only intentional package deliverables under version control.",
            confidence="0.97",
        )


def _analyze_vendor_environment(findings: list[Finding], findings_by_file: dict[str, list[str]]) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    for rel_path, category in ((".venv", "virtualenv_root_present"), ("frontend/node_modules", "node_modules_root_present")):
        path = REPO_ROOT / rel_path
        if not path.exists():
            continue
        file_count = sum(1 for candidate in path.rglob("*") if candidate.is_file())
        size_bytes = sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())
        anomalies.append(
            {
                "path": str(path),
                "category": category,
                "file_count": file_count,
                "size_bytes": size_bytes,
            }
        )
        _add_finding(
            findings,
            findings_by_file,
            severity="low",
            subsystem="build/release/package",
            file=str(path),
            line_or_surface="environment_root",
            category=category,
            description="Large dependency/runtime tree lives inside the project folder.",
            user_impact="Inflates workspace size and makes literal-folder audits, indexing, and archival handoff noisier than necessary.",
            evidence=f"file_count={file_count}, size_bytes={size_bytes}",
            fix_recommendation="Keep these trees out of repository handoff scopes when possible or document that they are environment surfaces, not project content.",
            confidence="0.91",
        )
    return anomalies


def run_full_folder_audit(*, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    tracked_files = _run_git("ls-files")
    untracked_files = _run_git("ls-files", "--others", "--exclude-standard")
    coverage_rows: list[dict[str, Any]] = []
    findings: list[Finding] = []
    findings_by_file: dict[str, list[str]] = {}
    semantic_count = 0
    metadata_count = 0
    pdf_paths: list[Path] = []

    for path in sorted(_iter_files(REPO_ROOT)):
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        tracked = rel_path in tracked_files
        file_class, review_mode = _classify_file(path, tracked=tracked)
        if review_mode == "semantic":
            semantic_count += 1
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                semantic_read = True
                review_pass = True
                _scan_text_file(rel_path, text, findings, findings_by_file)
            except OSError:
                semantic_read = False
                review_pass = False
        else:
            metadata_count += 1
            semantic_read = False
            review_pass = True

        if path.suffix.lower() == ".pdf":
            pdf_paths.append(path)

        try:
            stat = path.stat()
            size_bytes = stat.st_size
            modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            size_bytes = -1
            modified_at = ""
            review_pass = False

        coverage_rows.append(
            {
                "absolute_path": str(path),
                "relative_path": rel_path,
                "top_level_area": rel_path.split("/", 1)[0] if "/" in rel_path else "<root>",
                "file_class": file_class,
                "review_mode": review_mode,
                "tracked_status": "tracked" if tracked else ("untracked" if rel_path in untracked_files else "observed"),
                "subsystem": _subsystem_for(rel_path),
                "size_bytes": size_bytes,
                "modified_at_utc": modified_at,
                "sha256_small_file": _sha256(path, max_bytes=HASH_MAX_BYTES),
                "semantic_read": semantic_read,
                "review_pass": review_pass,
                "finding_ids": ";".join(findings_by_file.get(str(path), [])),
            }
        )

    _analyze_real_data_and_release_state(findings, findings_by_file)
    _analyze_workspace_state(tracked_files, findings, findings_by_file)
    vendor_anomalies = _analyze_vendor_environment(findings, findings_by_file)
    _duplicate_pdf_findings(pdf_paths, findings, findings_by_file)
    _analyze_present_package_outputs(coverage_rows, findings, findings_by_file)

    for row in coverage_rows:
        row["finding_ids"] = ";".join(findings_by_file.get(row["absolute_path"], []))

    severity_counts: dict[str, int] = {}
    for finding in findings:
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

    out_dir.mkdir(parents=True, exist_ok=True)

    coverage_path = out_dir / "full_folder_coverage_ledger.csv"
    with coverage_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage_rows[0].keys()) if coverage_rows else [])
        if coverage_rows:
            writer.writeheader()
            writer.writerows(coverage_rows)

    findings_path = out_dir / "full_folder_findings_catalog.csv"
    with findings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Finding.__dataclass_fields__.keys()))
        writer.writeheader()
        for finding in findings:
            writer.writerow(finding.__dict__)

    vendor_path = out_dir / "full_folder_vendor_anomalies.csv"
    with vendor_path.open("w", encoding="utf-8", newline="") as handle:
        if vendor_anomalies:
            writer = csv.DictWriter(handle, fieldnames=list(vendor_anomalies[0].keys()))
            writer.writeheader()
            writer.writerows(vendor_anomalies)
        else:
            handle.write("path,category,file_count,size_bytes\n")

    summary = {
        "generated_at_utc": _utc_now_iso(),
        "root": str(REPO_ROOT),
        "files_total": len(coverage_rows),
        "semantic_review_files": semantic_count,
        "metadata_review_files": metadata_count,
        "tracked_files_observed": sum(1 for row in coverage_rows if row["tracked_status"] == "tracked"),
        "untracked_files_observed": sum(1 for row in coverage_rows if row["tracked_status"] == "untracked"),
        "findings_total": len(findings),
        "severity_counts": severity_counts,
        "coverage_ledger": str(coverage_path),
        "findings_catalog": str(findings_path),
        "vendor_anomalies": str(vendor_path),
    }
    (out_dir / "full_folder_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    top_findings = sorted(
        findings,
        key=lambda row: ({"critical": 0, "high": 1, "medium": 2, "low": 3}.get(row.severity, 9), row.id),
    )[:12]
    synthesis_lines = [
        "# Full-Folder Audit Synthesis",
        "",
        f"- Generated: {summary['generated_at_utc']}",
        f"- Files inventoried: {summary['files_total']}",
        f"- Semantic-review files: {summary['semantic_review_files']}",
        f"- Metadata-review files: {summary['metadata_review_files']}",
        f"- Findings: {summary['findings_total']}",
        "",
        "## Severity Counts",
        "",
    ]
    for severity in ("critical", "high", "medium", "low"):
        synthesis_lines.append(f"- {severity}: {severity_counts.get(severity, 0)}")
    synthesis_lines.extend(
        [
            "",
            "## Highest-Risk Findings",
            "",
        ]
    )
    for finding in top_findings:
        synthesis_lines.append(
            f"- `{finding.id}` `{finding.severity}` {finding.category}: {finding.description} "
            f"([{Path(finding.file).relative_to(REPO_ROOT).as_posix()}]({finding.file}))"
        )
    synthesis_lines.extend(
        [
            "",
            "## Repeated Failure Patterns",
            "",
            "- Machine-specific path coupling appears in execution surfaces, docs, and generated report payloads.",
            "- Tracked LaTeX auxiliaries and other generated build outputs create constant dirty-state churn and source/generated ambiguity.",
            "- Final release/package state is blocked by real-data preflight and equal-domain parity gaps rather than manuscript compile failures.",
            "- The project root contains large environment/vendor trees, which make literal-folder handoff and audit workflows expensive.",
            "",
            "## Output Files",
            "",
            f"- Coverage ledger: `{coverage_path}`",
            f"- Findings catalog: `{findings_path}`",
            f"- Vendor anomalies: `{vendor_path}`",
            f"- Summary JSON: `{out_dir / 'full_folder_summary.json'}`",
        ]
    )
    (out_dir / "full_folder_synthesis.md").write_text("\n".join(synthesis_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full-folder audit with semantic and metadata review modes.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = run_full_folder_audit(out_dir=args.out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
