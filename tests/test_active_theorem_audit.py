from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_JSON = REPO_ROOT / "reports" / "publication" / "active_theorem_audit.json"


def _load_payload() -> dict:
    return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))


def test_active_theorem_audit_covers_yaml_canonical_surfaces() -> None:
    payload = _load_payload()
    theorem_ids = [row["theorem_id"] for row in payload["theorems"]]
    assert theorem_ids == [
        "T1",
        "T2",
        "T3a",
        "T3b",
        "T4",
        "T5",
        "T6",
        "T7",
        "T8",
        "T9",
        "T10",
        "T11",
        "L1",
        "L2",
        "L3",
        "L4",
        "T11_Byzantine",
        "T_stale_decay",
        "T_minimax",
        "T_sensor_converse",
        "T_trajectory_PAC",
    ]


def test_every_active_theorem_row_has_required_sync_fields() -> None:
    payload = _load_payload()
    for row in payload["theorems"]:
        assert row["defense_tier"]
        assert row["program_role"]
        assert row["scope_note"]
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


def test_t3a_row_keeps_the_reliability_score_scope_explicit() -> None:
    payload = _load_payload()
    row = next(item for item in payload["theorems"] if item["theorem_id"] == "T3a")
    assert "reliability-score interpretation" in row["scope_note"]
    assert row["rigor_rating"] == "paper_rigorous"
    assert row["code_correspondence"] == "matches"


def test_t11_row_points_to_typed_contract_surface_and_supporting_harness() -> None:
    payload = _load_payload()
    row = next(item for item in payload["theorems"] if item["theorem_id"] == "T11")
    code_paths = {anchor["path"] for anchor in row["code_anchors"]}
    assert "src/orius/universal_theory/contracts.py" in code_paths
    assert "src/orius/universal/contract.py" in code_paths
    assert row["code_correspondence"] == "matches"


def test_l1_and_l2_rows_are_explicitly_scoped_as_open_extension_laws() -> None:
    payload = _load_payload()
    l1 = next(item for item in payload["theorems"] if item["theorem_id"] == "L1")
    l2 = next(item for item in payload["theorems"] if item["theorem_id"] == "L2")
    assert l1["rigor_rating"] == "stylized_surrogate"
    assert l2["rigor_rating"] == "proxy_bridge"
    assert l1["remediation_class"] == "future work"
    assert l2["remediation_class"] == "future work"


def test_namespace_drift_entries_cover_legacy_numbering_and_mini_harness() -> None:
    payload = _load_payload()
    drift_surfaces = {entry["surface"] for entry in payload["namespace_drift"]}
    assert "src/orius/dc3s/coverage_theorem.py and tests/test_conditional_coverage.py" in drift_surfaces
    assert "src/orius/universal/contract.py, tests/test_universal_contract.py, and tests/test_unification.py" in drift_surfaces
    assert "reports/publication/theorem_surface_register.csv" in drift_surfaces


def test_defense_tiers_match_the_rebuilt_core() -> None:
    payload = _load_payload()
    flagship_ids = [row["theorem_id"] for row in payload["theorems"] if row["defense_tier"] == "flagship_defended"]
    supporting_ids = [row["theorem_id"] for row in payload["theorems"] if row["defense_tier"] == "supporting_defended"]
    draft_ids = [row["theorem_id"] for row in payload["theorems"] if row["defense_tier"] == "draft_non_defended"]

    assert flagship_ids == ["T1", "T2", "T3a", "T4", "T11", "T_trajectory_PAC"]
    assert supporting_ids == ["T3b", "T6", "T8", "T11_Byzantine", "T_stale_decay"]
    assert "T5" in draft_ids
    assert "T9" in draft_ids
    assert "T_minimax" in draft_ids


def test_flagship_rows_are_bounded_and_not_marked_broken() -> None:
    payload = _load_payload()
    flagship_rows = [row for row in payload["theorems"] if row["defense_tier"] == "flagship_defended"]
    for row in flagship_rows:
        assert row["rigor_rating"] not in {"broken", "has-a-hole"}
        assert row["scope_note"]


def test_summary_exposes_defended_core_counts_and_readiness() -> None:
    payload = _load_payload()
    summary = payload["summary"]
    assert summary["defense_tier_counts"] == {
        "flagship_defended": 6,
        "supporting_defended": 5,
        "draft_non_defended": 10,
    }
    assert summary["flagship_gate_ready"] is True
