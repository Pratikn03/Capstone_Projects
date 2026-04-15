from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_JSON = REPO_ROOT / "reports" / "publication" / "active_theorem_audit.json"


def _load_payload() -> dict:
    return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))


def test_active_theorem_audit_covers_t1_through_t11() -> None:
    payload = _load_payload()
    theorem_ids = [row["theorem_id"] for row in payload["theorems"]]
    assert theorem_ids == [f"T{i}" for i in range(1, 12)]


def test_every_active_theorem_row_has_required_sync_fields() -> None:
    payload = _load_payload()
    for row in payload["theorems"]:
        assert row["statement_location"]
        assert row["proof_location"]
        assert row["assumptions_used"]
        assert row["weakest_step"]
        assert row["rigor_rating"]
        assert row["code_correspondence"]
        assert row["severity_if_broken"]
        assert row["remediation_class"]
        assert row["code_anchors"]
        assert row["test_anchors"]


def test_t3_row_explicitly_disclaims_probability_interpretation_of_w() -> None:
    payload = _load_payload()
    row = next(item for item in payload["theorems"] if item["theorem_id"] == "T3")
    assert "w_t is a runtime reliability score, not a probability by definition." in row["assumptions_used"]
    assert row["rigor_rating"] in {"has-a-hole", "broken"}


def test_t11_row_points_to_typed_contract_surface_and_supporting_harness() -> None:
    payload = _load_payload()
    row = next(item for item in payload["theorems"] if item["theorem_id"] == "T11")
    code_paths = {anchor["path"] for anchor in row["code_anchors"]}
    assert "src/orius/universal_theory/contracts.py" in code_paths
    assert "src/orius/universal/contract.py" in code_paths
    assert row["code_correspondence"] == "partial"


def test_namespace_drift_entries_cover_legacy_numbering_and_mini_harness() -> None:
    payload = _load_payload()
    drift_surfaces = {entry["surface"] for entry in payload["namespace_drift"]}
    assert "src/orius/dc3s/coverage_theorem.py and tests/test_conditional_coverage.py" in drift_surfaces
    assert "src/orius/universal/contract.py, tests/test_universal_contract.py, and tests/test_unification.py" in drift_surfaces
    assert "reports/publication/theorem_surface_register.csv" in drift_surfaces
