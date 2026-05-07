#!/usr/bin/env python3
"""Validate flagship theorem promotion cards against minimum gate requirements."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATRIX = REPO / "reports/publication/theorem_promotion_matrix.json"
REQUIRED_KEYS = ("theorem_id", "title", "status", "assumptions", "proof_file", "code_anchor", "tests", "artifacts", "claim_boundary")


def _nonempty_list(v):
    return isinstance(v, list) and len(v) > 0 and all(str(x).strip() for x in v)


def validate() -> int:
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    errors: list[str] = []
    for row in payload.get("theorems", []):
        status = str(row.get("status", "")).strip().lower()
        if "flagship" not in status:
            continue
        card_path = REPO / row["result_card"]
        if not card_path.exists():
            errors.append(f"{row.get('theorem_id')}: missing result card {card_path}")
            continue
        card = json.loads(card_path.read_text(encoding="utf-8"))
        for key in REQUIRED_KEYS:
            if key not in card:
                errors.append(f"{row.get('theorem_id')}: missing key {key}")
        if not _nonempty_list(card.get("assumptions")):
            errors.append(f"{row.get('theorem_id')}: assumptions must be non-empty list")
        if not _nonempty_list(card.get("tests")):
            errors.append(f"{row.get('theorem_id')}: tests must be non-empty list")
        if not _nonempty_list(card.get("artifacts")):
            errors.append(f"{row.get('theorem_id')}: artifacts must be non-empty list")
        for path_key in ("proof_file", "code_anchor"):
            if not str(card.get(path_key, "")).strip():
                errors.append(f"{row.get('theorem_id')}: {path_key} must be non-empty")
        if not str(card.get("claim_boundary", "")).strip():
            errors.append(f"{row.get('theorem_id')}: claim_boundary missing")
    if errors:
        print("THEOREM PROMOTION VALIDATION FAILED")
        for e in errors:
            print(f" - {e}")
        return 1
    print("THEOREM PROMOTION VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(validate())
