#!/usr/bin/env python3
"""Validate flagship theorem promotion cards against hard gate requirements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATRIX = REPO / "reports/publication/theorem_promotion_matrix.json"
REQUIRED_KEYS = (
    "theorem_id",
    "title",
    "status",
    "assumptions",
    "proof_file",
    "code_anchor",
    "tests",
    "artifacts",
    "artifact_hashes",
    "claim_boundary",
    "manuscript_anchor",
)


def _nonempty_list(value: object) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(str(item).strip() for item in value)


def _repo_path(reference: str) -> Path:
    path = Path(reference)
    if path.is_absolute():
        return path
    return REPO / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate() -> int:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    errors: list[str] = []
    seen: set[str] = set()
    for row in payload.get("theorems", []):
        theorem_id = str(row.get("theorem_id", "")).strip()
        status = str(row.get("status", "")).strip().lower()
        if "flagship" not in status:
            continue
        if theorem_id in seen:
            errors.append(f"{theorem_id}: duplicate theorem row")
        seen.add(theorem_id)

        card_path = _repo_path(str(row.get("result_card", "")))
        if not card_path.exists():
            errors.append(f"{theorem_id}: missing result card {card_path}")
            continue
        card = json.loads(card_path.read_text(encoding="utf-8"))
        for key in REQUIRED_KEYS:
            if key not in card:
                errors.append(f"{theorem_id}: missing key {key}")
        if not _nonempty_list(card.get("assumptions")):
            errors.append(f"{theorem_id}: assumptions must be non-empty list")
        if not _nonempty_list(card.get("tests")):
            errors.append(f"{theorem_id}: tests must be non-empty list")
        if not _nonempty_list(card.get("artifacts")):
            errors.append(f"{theorem_id}: artifacts must be non-empty list")

        for path_key in ("proof_file", "code_anchor", "manuscript_anchor"):
            reference = str(card.get(path_key, "")).strip()
            if not reference:
                errors.append(f"{theorem_id}: {path_key} must be non-empty")
                continue
            if not _repo_path(reference).exists():
                errors.append(f"{theorem_id}: {path_key} does not exist: {reference}")

        for test in card.get("tests", []):
            reference = str(test)
            if not _repo_path(reference).exists():
                errors.append(f"{theorem_id}: test does not exist: {reference}")

        hashes = card.get("artifact_hashes", {})
        if not isinstance(hashes, dict):
            errors.append(f"{theorem_id}: artifact_hashes must be a dict")
            hashes = {}
        for artifact in card.get("artifacts", []):
            reference = str(artifact)
            path = _repo_path(reference)
            if not path.exists():
                errors.append(f"{theorem_id}: artifact does not exist: {reference}")
                continue
            recorded = str(hashes.get(reference, "")).strip()
            if not recorded:
                errors.append(f"{theorem_id}: missing artifact hash for {reference}")
                continue
            if _sha256(path) != recorded:
                errors.append(f"{theorem_id}: stale artifact hash for {reference}")

        if not str(card.get("claim_boundary", "")).strip():
            errors.append(f"{theorem_id}: claim_boundary missing")

    if errors:
        print("THEOREM PROMOTION VALIDATION FAILED")
        for error in errors:
            print(f" - {error}")
        return 1
    print("THEOREM PROMOTION VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate())
