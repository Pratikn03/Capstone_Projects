#!/usr/bin/env python3
"""Validate next-tier validation manifests remain bounded until rerun evidence exists."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "reports" / "predeployment_external_validation"
AV_MANIFEST = OUT_DIR / "nuplan_carla_preparation_manifest.json"
HC_MANIFEST = OUT_DIR / "healthcare_heldout_runtime_preparation_manifest.json"
SUMMARY = OUT_DIR / "next_tier_validation_preparation_manifest.json"
EXTERNAL_SUMMARY = OUT_DIR / "external_validation_summary.csv"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _artifact_exists(path: str) -> bool:
    return (REPO_ROOT / path).exists()


def _negative_context(text: str, start: int, end: int) -> bool:
    sentence_start = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    sentence_end_candidates = [idx for idx in (text.find(".", end), text.find("\n", end)) if idx != -1]
    sentence_end = min(sentence_end_candidates) if sentence_end_candidates else len(text)
    sentence = text[sentence_start:sentence_end]
    return any(marker in sentence for marker in ("not ", "not_", "without claiming", "future validation"))


def _json_text_without_keys(payload: object, *, skip_keys: set[str]) -> str:
    if isinstance(payload, dict):
        values = [
            _json_text_without_keys(value, skip_keys=skip_keys)
            for key, value in payload.items()
            if key not in skip_keys
        ]
        return " ".join(value for value in values if value)
    if isinstance(payload, list):
        return " ".join(_json_text_without_keys(value, skip_keys=skip_keys) for value in payload)
    return str(payload).lower()


def _has_required_nonclaim(boundary: str, phrase: str) -> bool:
    phrase = phrase.lower()
    if phrase not in boundary:
        return False
    match = re.search(re.escape(phrase), boundary)
    if match is None:
        return False
    return _negative_context(boundary, match.start(), match.end())


def main() -> int:
    findings: list[str] = []
    for path in (AV_MANIFEST, HC_MANIFEST, SUMMARY):
        if not path.exists():
            findings.append(f"missing manifest: {path}")

    av = _read_json(AV_MANIFEST)
    hc = _read_json(HC_MANIFEST)
    summary = _read_json(SUMMARY)

    if av:
        required = list(av.get("completion_required_artifacts", []))
        nuplan_required = list(av.get("nuplan_completion_artifacts", []))
        carla_required = list(av.get("carla_completion_artifacts", []))
        complete = all(_artifact_exists(path) for path in required)
        nuplan_complete = (
            all(_artifact_exists(path) for path in nuplan_required) if nuplan_required else False
        )
        carla_complete = all(_artifact_exists(path) for path in carla_required) if carla_required else False
        if av.get("completed_evidence") != complete:
            findings.append("AV completion flag does not match required nuPlan/CARLA artifacts")
        if av.get("nuplan_completed_evidence") != nuplan_complete:
            findings.append("AV nuPlan bounded completion flag does not match nuPlan artifacts")
        if av.get("carla_completed_evidence") != carla_complete:
            findings.append("AV CARLA completion flag does not match CARLA artifacts")
        if av.get("completed_evidence") and not (nuplan_complete and carla_complete):
            findings.append("AV completed_evidence requires both nuPlan and CARLA completion")
        if av.get("claim_allowed") and not complete:
            findings.append("AV nuPlan/CARLA claim allowed without runtime artifacts")
        boundary = str(av.get("claim_boundary", "")).lower()
        for phrase in ("completed nuplan", "completed carla", "road deployment"):
            if not _has_required_nonclaim(boundary, phrase):
                findings.append(f"AV boundary missing explicit non-claim phrase: {phrase}")
        non_boundary_text = _json_text_without_keys(av, skip_keys={"claim_boundary"})
        for pattern in (
            r"\bcompleted\s+carla\b",
            r"\broad\s+deployment\b",
            r"\bfull\s+autonomous[- ]driving\s+closure\b",
        ):
            if re.search(pattern, non_boundary_text):
                findings.append(
                    f"AV manifest has positive deployment/completion claim outside boundary: {pattern}"
                )

    if hc:
        required = list(hc.get("required_runtime_artifacts_for_claim", []))
        complete = all(_artifact_exists(path) for path in required)
        if hc.get("completed_evidence") != complete:
            findings.append("Healthcare completion flag does not match required held-out runtime artifacts")
        if hc.get("heldout_claim_allowed") and not complete:
            findings.append("Healthcare held-out runtime claim allowed without runtime artifacts")
        boundary = str(hc.get("claim_boundary", "")).lower()
        for phrase in (
            "not claim live clinical",
            "prospective trial evidence",
            "clinical decision support approval",
        ):
            if phrase not in boundary:
                findings.append(f"Healthcare boundary missing phrase: {phrase}")

    if summary.get("status") != "prepared_not_completed":
        findings.append(
            "Next-tier preparation summary must remain prepared_not_completed until promoted separately"
        )

    if EXTERNAL_SUMMARY.exists():
        text = EXTERNAL_SUMMARY.read_text(encoding="utf-8").lower()
        forbidden = [
            r"prospective_split_pass",
            r"nuplan.*completed",
            r"carla.*completed",
            r"live clinical",
            r"clinical deployment",
        ]
        for pattern in forbidden:
            for match in re.finditer(pattern, text):
                if _negative_context(text, match.start(), match.end()):
                    continue
                findings.append(
                    "external validation summary contains forbidden completed/non-retrospective claim: "
                    f"{pattern}"
                )
                break

    if findings:
        print("[validate_next_tier_validation_boundaries] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_next_tier_validation_boundaries] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
