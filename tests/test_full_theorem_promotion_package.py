from __future__ import annotations

import json
from pathlib import Path

import scripts.validate_theorem_promotion as validator

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = REPO_ROOT / "reports" / "publication"
ALLOWED_STATUSES = {
    "flagship_theorem",
    "flagship_corollary",
    "flagship_definition",
    "flagship_lemma",
    "archived_only",
}
LEGACY_SHORTHAND_IDS = {"T11Byz", "Tstale", "Tminimax", "Tsensor"}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_promotion_matrix_covers_every_active_theorem_surface() -> None:
    active = _load_json(PUBLICATION / "active_theorem_audit.json")
    matrix = _load_json(PUBLICATION / "theorem_promotion_matrix.json")

    active_ids = {row["theorem_id"] for row in active["theorems"] if row.get("status", "active") == "active"}
    matrix_ids = {row["theorem_id"] for row in matrix["theorems"]}

    assert matrix_ids == active_ids
    assert not (matrix_ids & LEGACY_SHORTHAND_IDS)


def test_promotion_cards_use_only_flagship_or_archived_statuses() -> None:
    matrix = _load_json(PUBLICATION / "theorem_promotion_matrix.json")

    for row in matrix["theorems"]:
        card_path = REPO_ROOT / row["result_card"]
        card = _load_json(card_path)

        assert row["status"] in ALLOWED_STATUSES
        assert card["status"] == row["status"]
        assert card["theorem_id"] == row["theorem_id"]
        assert card["status"] != "draft_non_defended"
        assert card["assumptions"]
        assert card["tests"]
        assert card["artifacts"]
        assert card["claim_boundary"]


def test_no_legacy_shorthand_result_cards_remain() -> None:
    card_dir = PUBLICATION / "theorem_result_cards"

    card_stems = {path.stem for path in card_dir.glob("*.json")}

    assert not (card_stems & LEGACY_SHORTHAND_IDS)


def test_full_theorem_promotion_validator_passes() -> None:
    assert validator.validate() == 0
