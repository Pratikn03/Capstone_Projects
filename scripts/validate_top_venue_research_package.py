#!/usr/bin/env python3
"""Validate the top-venue ORIUS research package."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"

PACKAGE_MD = PUBLICATION_DIR / "top_venue_research_package.md"
PACKAGE_JSON = PUBLICATION_DIR / "top_venue_research_package.json"
MATRIX_CSV = PUBLICATION_DIR / "reviewer_claim_evidence_matrix.csv"
LIMITATIONS_MD = PUBLICATION_DIR / "research_limitations_boundary.md"
RESPONSES_MD = PUBLICATION_DIR / "reviewer_response_bank.md"
SOURCE_POSITIONING_CSV = PUBLICATION_DIR / "source_backed_research_positioning.csv"
UPLIFT_SCORECARD_CSV = PUBLICATION_DIR / "orius_95plus_uplift_scorecard.csv"
UPLIFT_SCORECARD_JSON = PUBLICATION_DIR / "orius_95plus_uplift_scorecard.json"
UPLIFT_SCORECARD_MD = PUBLICATION_DIR / "orius_95plus_uplift_scorecard.md"

CANONICAL_ORIUS_FRAMING = (
    "ORIUS provides a reliability-aware runtime safety layer for physical AI under "
    "degraded observation, enforcing certificate-backed action release through "
    "uncertainty coverage, repair, and fallback."
)

FORBIDDEN_PATTERNS = {
    "av_full_deployment": re.compile(
        r"\b(AV|autonomous vehicles?)\b.{0,100}\b(full|complete|unrestricted).{0,60}\b(autonomous-driving|road|deployment|closure)\b",
        re.I | re.S,
    ),
    "healthcare_live_clinical": re.compile(
        r"\bhealthcare\b.{0,100}\b(live clinical|clinical deployment|prospective trial|clinical decision support approval)\b",
        re.I | re.S,
    ),
    "validation_harness_headline": re.compile(
        r"\bheadline\b.{0,140}\b(validation[-_ ]harness|diagnostic validation harness)\b", re.I | re.S
    ),
    "completed_freeze_without_manifest": re.compile(r"\b(completed|final|frozen) release\b", re.I),
}

REQUIRED_PHRASES = (
    CANONICAL_ORIUS_FRAMING,
    "observation-only mandatory-release controllers face a lower bound",
    "ORIUS achieves an alpha-bounded upper guarantee",
    "safety-optimal under covered observation ambiguity",
    "runtime assurance",
    "conformal",
    "control barrier",
    "TRIPOD+AI",
    "PROBAST+AI",
    "CONSORT-AI",
    "DECIDE-AI",
    "STARD-AI",
    "MIMIC-IV",
    "eICU",
    "nuPlan closed-loop",
    "CARLA",
    "not live clinical deployment",
    "not clinical decision support approval",
    "not prospective trial evidence",
    "not full autonomous-driving field closure",
    "not yet physical HIL",
    "not equal real-world maturity",
    "runtime-denominator",
)


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _artifact_exists(source: str) -> bool:
    if not source:
        return False
    if source.startswith("http"):
        return True
    return (REPO_ROOT / source).exists()


def _freeze_manifest_exists() -> bool:
    return any(
        (REPO_ROOT / "reports" / "predeployment_freeze").glob(
            "PREDEPLOY*/predeployment_release_manifest.json"
        )
    )


def _is_negative_boundary_context(text: str, start: int, end: int) -> bool:
    """Allow explicit non-claims while still rejecting positive overclaims."""
    sentence_start = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    sentence_end_candidates = [idx for idx in (text.find(".", end), text.find("\n", end)) if idx != -1]
    sentence_end = min(sentence_end_candidates) if sentence_end_candidates else len(text)
    sentence = text[sentence_start:sentence_end].lower()
    negative_markers = (
        "not ",
        "not yet",
        "do not",
        "does not",
        "no final",
        "not allowed",
        "future validation tier",
    )
    return any(marker in sentence for marker in negative_markers)


def main() -> int:
    findings: list[str] = []
    for path in (
        PACKAGE_MD,
        PACKAGE_JSON,
        MATRIX_CSV,
        LIMITATIONS_MD,
        RESPONSES_MD,
        SOURCE_POSITIONING_CSV,
        UPLIFT_SCORECARD_CSV,
        UPLIFT_SCORECARD_JSON,
        UPLIFT_SCORECARD_MD,
    ):
        if not path.exists():
            findings.append(f"missing output: {path}")

    if findings:
        print("[validate_top_venue_research_package] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1

    package_text = _read_text(PACKAGE_MD)
    limitations_text = _read_text(LIMITATIONS_MD)
    responses_text = _read_text(RESPONSES_MD)
    uplift_text = _read_text(UPLIFT_SCORECARD_MD)
    all_text = "\n".join((package_text, limitations_text, responses_text, uplift_text))
    package_json = json.loads(_read_text(PACKAGE_JSON))
    uplift_json = json.loads(_read_text(UPLIFT_SCORECARD_JSON))
    matrix = _read_csv(MATRIX_CSV)
    source_rows = _read_csv(SOURCE_POSITIONING_CSV)
    scorecard_rows = _read_csv(UPLIFT_SCORECARD_CSV)

    for phrase in REQUIRED_PHRASES:
        if phrase.lower() not in all_text.lower():
            findings.append(f"missing required phrase: {phrase}")

    for key, pattern in FORBIDDEN_PATTERNS.items():
        if key == "completed_freeze_without_manifest" and _freeze_manifest_exists():
            continue
        for match in pattern.finditer(all_text):
            if _is_negative_boundary_context(all_text, match.start(), match.end()):
                continue
            findings.append(f"forbidden overclaim pattern matched: {key}")
            break

    if not matrix:
        findings.append("reviewer claim evidence matrix is empty")
    for row in matrix:
        claim_id = row.get("claim_id", "unknown")
        source = row.get("artifact_source", "")
        if row.get("claim_type") == "headline" and not source:
            findings.append(f"{claim_id}: headline claim has no artifact source")
        if source and not _artifact_exists(source):
            findings.append(f"{claim_id}: artifact source does not exist: {source}")
        if row.get("claim_type") == "headline" and "validation_harness" in source.lower():
            findings.append(f"{claim_id}: headline claim uses validation-harness source")

    if not any(
        row.get("claim_id") == "H2" and "lower bound" in row.get("allowed_language", "").lower()
        for row in matrix
    ):
        findings.append("matrix must include theorem lower-bound / upper-bound distinction")
    if not any(row.get("claim_id") == "F1" and row.get("evidence_status") == "incomplete" for row in matrix):
        if not _freeze_manifest_exists():
            findings.append(
                "freeze status must be marked incomplete until predeployment release manifest exists"
            )

    required_source_lanes = {
        "runtime_assurance",
        "uncertainty_coverage",
        "safety_critical_control",
        "clinical_dataset",
        "clinical_reporting",
        "clinical_bias",
        "clinical_trial_reporting",
        "clinical_live_evaluation",
        "clinical_diagnostic_reporting",
        "av_closed_loop",
        "av_stress_simulation",
        "av_dataset",
    }
    present_lanes = {row.get("lane", "") for row in source_rows}
    missing_lanes = sorted(required_source_lanes - present_lanes)
    if missing_lanes:
        findings.append(f"source positioning missing lanes: {missing_lanes}")
    for row in source_rows:
        url = row.get("url", "")
        if not url.startswith("https://"):
            findings.append(f"source row {row.get('source_name', 'unknown')} must use an https source URL")
        if not row.get("orius_gap") or not row.get("boundary"):
            findings.append(
                f"source row {row.get('source_name', 'unknown')} must state ORIUS gap and boundary"
            )

    required_scorecard_dimensions = {
        "core_idea_novelty",
        "theory",
        "three_domain_runtime_evidence",
        "external_validation_depth",
        "reproducibility_and_freeze",
        "claim_quality",
    }
    present_dimensions = {row.get("dimension", "") for row in scorecard_rows}
    missing_dimensions = sorted(required_scorecard_dimensions - present_dimensions)
    if missing_dimensions:
        findings.append(f"95+ scorecard missing dimensions: {missing_dimensions}")
    for row in scorecard_rows:
        if int(float(row.get("target_score", "0"))) < 95:
            findings.append(f"{row.get('dimension', 'unknown')}: target score must be at least 95")
        if row.get("current_status") != "pass" and not row.get("remaining_blocker"):
            findings.append(
                f"{row.get('dimension', 'unknown')}: blocked scorecard rows must name the blocker"
            )
    if uplift_json.get("achieved") and not all(row.get("current_status") == "pass" for row in scorecard_rows):
        findings.append("uplift JSON cannot mark 95+ achieved while scorecard rows remain blocked")

    expected_status = "top_venue_defensible_predeployment_package"
    if package_json.get("status") != expected_status:
        findings.append(f"JSON status must be {expected_status!r}")
    if package_json.get("freeze_status", {}).get("complete") and not _freeze_manifest_exists():
        findings.append("JSON freeze_status.complete is true but no release manifest exists")
    if (
        package_json.get("healthcare_boundary")
        != "retrospective_source_holdout_time_forward_not_live_clinical"
    ):
        findings.append("JSON healthcare boundary must reject live clinical deployment")
    if package_json.get("av_boundary") != "nuplan_replay_surrogate_not_carla_or_road_deployment":
        findings.append("JSON AV boundary must reject CARLA/road deployment completion claims")
    if len(package_json.get("source_anchors", [])) < 12:
        findings.append("JSON must include the source-backed novelty anchors")
    if package_json.get("uplift_95plus_achieved") and not all(
        row.get("current_status") == "pass" for row in scorecard_rows
    ):
        findings.append("package JSON cannot mark 95+ achieved while scorecard rows remain blocked")

    if findings:
        print("[validate_top_venue_research_package] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("[validate_top_venue_research_package] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
