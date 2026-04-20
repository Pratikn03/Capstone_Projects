"""Submission-facing audit for the solo ORIUS thesis package."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISH_DIR = REPO_ROOT / "reports" / "publish"
JSON_REPORT = PUBLISH_DIR / "final_submission_audit_report.json"
MD_REPORT = PUBLISH_DIR / "final_submission_audit_report.md"
CANONICAL_AUDIT = REPO_ROOT / "reports" / "final_thesis_submission_audit.md"


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scan_terms(path: Path, banned: list[str]) -> list[str]:
    text = path.read_text(encoding="utf-8").lower()
    return [term for term in banned if term in text]


def _aggregate_monograph_text() -> str:
    chunks: list[str] = []
    roots = [REPO_ROOT / "orius_book.tex", REPO_ROOT / "paper" / "paper.tex"]
    roots.extend(sorted((REPO_ROOT / "paper" / "monograph").glob("*.tex")))
    for path in roots:
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def _run_claim_validator() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "scripts/validate_paper_claims.py"],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = (proc.stdout or "").strip()
    details = output.splitlines()[-1] if output else f"exit_code={proc.returncode}"
    return proc.returncode == 0, details


def _build_checks() -> list[CheckResult]:
    paper_pdf = REPO_ROOT / "paper" / "paper.pdf"
    root_pdf = REPO_ROOT / "paper.pdf"
    paper_log = REPO_ROOT / "paper" / "paper.log"
    metrics_manifest = _load_json(REPO_ROOT / "paper" / "metrics_manifest.json")
    release_manifest = _load_json(REPO_ROOT / "reports" / "publication" / "release_manifest.json")
    artifact_appendix = (REPO_ROOT / "reports" / "publication" / "orius_artifact_appendix.md").read_text(encoding="utf-8")
    paper_tex = (REPO_ROOT / "orius_book.tex").read_text(encoding="utf-8")
    monograph_text = _aggregate_monograph_text()

    checks: list[CheckResult] = []

    pdf_ok = paper_pdf.exists() and root_pdf.exists()
    pdf_details = "both final PDFs exist"
    if pdf_ok:
        same = _sha256(paper_pdf) == _sha256(root_pdf)
        pdf_ok = pdf_ok and same
        pdf_details = "paper/paper.pdf and repo-root paper.pdf exist and match" if same else "final PDFs differ"
    else:
        pdf_details = "missing final PDF output"
    checks.append(CheckResult("Final PDF outputs", pdf_ok, pdf_details))

    log_text = paper_log.read_text(encoding="utf-8") if paper_log.exists() else ""
    latex_bad = [token for token in ["LaTeX Error:", "undefined references", "undefined on input line", "Emergency stop"] if token in log_text]
    checks.append(
        CheckResult(
            "Compiled manuscript log",
            paper_log.exists() and not latex_bad,
            "paper/paper.log exists with no LaTeX errors or unresolved references"
            if paper_log.exists() and not latex_bad
            else f"log issues: {latex_bad or ['missing paper/paper.log']}",
        )
    )

    checks.append(
        CheckResult(
            "Canonical manuscript authority",
            metrics_manifest["metric_policy"]["master_manuscript"] == "orius_book.tex"
            and release_manifest["paper_metric_policy"]["master_manuscript"] == "orius_book.tex",
            "metrics manifest and release manifest both point to orius_book.tex",
        )
    )

    titlepage_hits = _scan_terms(REPO_ROOT / "frontmatter" / "titlepage.tex", ["draft", "committee", "advisor"])
    checks.append(
        CheckResult(
            "Solo title page mode",
            not titlepage_hits,
            "title page has no draft, committee, or advisor wording"
            if not titlepage_hits
            else f"title page contains banned terms: {titlepage_hits}",
        )
    )

    ack_hits = _scan_terms(REPO_ROOT / "frontmatter" / "acknowledgments.tex", ["draft"])
    checks.append(
        CheckResult(
            "Acknowledgments wording",
            not ack_hits and "this thesis" in (REPO_ROOT / "frontmatter" / "acknowledgments.tex").read_text(encoding="utf-8").lower(),
            "acknowledgments use thesis wording and contain no draft marker"
            if not ack_hits
            else f"acknowledgments contain banned terms: {ack_hits}",
        )
    )

    abstract_text = (REPO_ROOT / "frontmatter" / "abstract.tex").read_text(encoding="utf-8").lower()
    abstract_hits = [term for term in ["draft", "dissertation"] if term in abstract_text]
    checks.append(
        CheckResult(
            "Abstract wording",
            not abstract_hits and "this thesis presents" in abstract_text,
            "abstract uses thesis wording and contains no draft/dissertation drift"
            if not abstract_hits and "this thesis presents" in abstract_text
            else f"abstract wording drift: {abstract_hits or ['missing thesis phrasing']}",
        )
    )

    tier_phrases = [
        "battery remains the witness row",
        "autonomous vehicles",
        "healthcare",
        "future additional domains",
    ]
    missing_tiers = [phrase for phrase in tier_phrases if phrase not in monograph_text]
    appendix_tiers = [
        "- Battery: witness row",
        "- AV: defended bounded row",
        "- Healthcare: defended bounded row",
    ]
    missing_appendix_tiers = [phrase for phrase in appendix_tiers if phrase not in artifact_appendix]
    checks.append(
        CheckResult(
            "Evidence tier consistency",
            not missing_tiers and not missing_appendix_tiers,
            "manuscript and artifact appendix use the same evidence-tier labels"
            if not missing_tiers and not missing_appendix_tiers
            else f"missing manuscript tiers={missing_tiers}, missing appendix tiers={missing_appendix_tiers}",
        )
    )

    bibliography_path = REPO_ROOT / "paper" / "bibliography" / "orius_monograph.bib"
    bib_entries = 0
    if bibliography_path.exists():
        bib_entries = sum(1 for line in bibliography_path.read_text(encoding="utf-8").splitlines() if line.startswith("@"))
    checks.append(
        CheckResult(
            "Bibliography depth",
            bib_entries >= 150,
            f"monograph bibliography contains {bib_entries} entries",
        )
    )

    legacy_root = "Pa" + "per"
    legacy_program_pattern = re.compile(
        legacy_root + r"(?:~|\s)?(?:1|2|3|4|5|6)\b"
    )
    lingering_program_refs = sorted(set(legacy_program_pattern.findall(monograph_text)))
    checks.append(
        CheckResult(
            "No active legacy paper-lineage framing",
            not lingering_program_refs,
            "active monograph surface contains no legacy paper-lineage framing"
            if not lingering_program_refs
            else f"active monograph still contains legacy program terms: {lingering_program_refs}",
        )
    )

    claim_sources = (
        (REPO_ROOT / "paper" / "claim_matrix.csv").read_text(encoding="utf-8")
        + "\n"
        + (REPO_ROOT / "paper" / "metrics_manifest.json").read_text(encoding="utf-8")
        + "\n"
        + (REPO_ROOT / "reports" / "publication" / "release_manifest.json").read_text(encoding="utf-8")
    )
    checks.append(
        CheckResult(
            "Claim-critical data sources",
            "data/dashboard" not in claim_sources,
            "claim-critical governance files do not reference ignored dashboard caches"
            if "data/dashboard" not in claim_sources
            else "ignored dashboard cache path found in governance files",
        )
    )

    checks.append(
        CheckResult(
            "IEEE bibliography style",
            "\\bibliographystyle{IEEEtran}" in paper_tex,
            "canonical manuscript declares IEEE bibliography style",
        )
    )

    support_files = [
        REPO_ROOT / "reports" / "final_submission_reproducibility_note.md",
        REPO_ROOT / "reports" / "final_code_data_availability_statement.md",
        REPO_ROOT / "reports" / "final_thesis_submission_checklist.md",
    ]
    missing_support = [str(path.relative_to(REPO_ROOT)) for path in support_files if not path.exists()]
    checks.append(
        CheckResult(
            "Submission support files",
            not missing_support
            and "final_submission_reproducibility_note.md" in artifact_appendix
            and "final_code_data_availability_statement.md" in artifact_appendix,
            "artifact appendix includes reproducibility and availability notes"
            if not missing_support
            else f"missing support files: {missing_support}",
        )
    )

    claims_ok, claim_details = _run_claim_validator()
    checks.append(CheckResult("Claim validator", claims_ok, claim_details))

    return checks


def _write_reports(checks: list[CheckResult]) -> None:
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "generated_at": generated_at,
        "mode": "solo_final_submission",
        "go": all(check.ok for check in checks),
        "checks": [
            {"name": check.name, "ok": check.ok, "details": check.details}
            for check in checks
        ],
    }
    JSON_REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Final Thesis Submission Audit",
        "",
        "**Scope:** solo final submission cleanup, submission-facing text hygiene, promoted-number traceability, and package readiness.",
        "",
        "**Mode:** solo submission package, not committee-formatted defense frontmatter.",
        "",
        "**Evidence-first rule:** all promoted numbers must map to tracked artifacts; no submission-facing claims may depend on ignored local caches.",
        "",
        f"**Generated:** `{generated_at}`",
        "",
        "| Check | Status | Details |",
        "|---|:---:|---|",
    ]
    for check in checks:
        lines.append(f"| {check.name} | {'PASS' if check.ok else 'FAIL'} | {check.details} |")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Overall result: **{'PASS' if all(check.ok for check in checks) else 'FAIL'}**",
            f"- JSON report: `{JSON_REPORT.relative_to(REPO_ROOT)}`",
        ]
    )
    report = "\n".join(lines) + "\n"
    MD_REPORT.write_text(report, encoding="utf-8")
    CANONICAL_AUDIT.write_text(report, encoding="utf-8")


def main() -> None:
    checks = _build_checks()
    _write_reports(checks)
    if not all(check.ok for check in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
