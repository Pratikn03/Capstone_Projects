from __future__ import annotations

import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_JSON = REPO_ROOT / "reports" / "publication" / "active_theorem_audit.json"
ASSUMPTION_MAP_CSV = REPO_ROOT / "reports" / "publication" / "defended_assumption_map.csv"
DEFENDED_CORE_JSON = REPO_ROOT / "reports" / "publication" / "defended_theorem_core.json"


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
        "T10_T11_ObservationAmbiguitySandwich",
        "T11_AV_BrakeHold",
        "T11_HC_FailSafeRelease",
        "T6_AV_FallbackValidity",
        "T6_HC_FallbackValidity",
        "T_EQ_Battery_RuntimeArtifactPackage",
        "T_EQ_AV_RuntimeArtifactPackage",
        "T_EQ_HC_RuntimeArtifactPackage",
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
        assert row["assumptions_used"] or row["typed_obligations"] or row["unresolved_assumptions"]
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
    assert row["assumptions_used"] == []
    assert row["unresolved_assumptions"] == []
    assert row["typed_obligations"] == [
        "Coverage obligation for the observation-consistent state set.",
        "Soundness of the tightened safe-action set.",
        "Repair membership in the tightened safe-action set.",
        "Fallback admissibility when the tightened set is empty.",
    ]


def test_t11_domain_runtime_lemmas_are_supporting_not_flagship() -> None:
    payload = _load_payload()
    rows = {
        item["theorem_id"]: item
        for item in payload["theorems"]
        if item["theorem_id"]
        in {
            "T11_AV_BrakeHold",
            "T11_HC_FailSafeRelease",
            "T6_AV_FallbackValidity",
            "T6_HC_FallbackValidity",
        }
    }

    assert set(rows) == {
        "T11_AV_BrakeHold",
        "T11_HC_FailSafeRelease",
        "T6_AV_FallbackValidity",
        "T6_HC_FallbackValidity",
    }
    for row in rows.values():
        assert row["defense_tier"] == "supporting_defended"
        assert row["rigor_rating"] == "proof_runtime_linked"
        assert row["code_correspondence"] == "matches"
        assert row["assumptions_used"] == []
        assert row["typed_obligations"]
        assert row["unresolved_assumptions"] == []
        assert "T11" in row["dependencies"]
        assert row["code_anchors"]
        assert row["test_anchors"]
        assert "full autonomous" not in row["scope_note"].lower()
        assert "regulated clinical" not in row["scope_note"].lower()


def test_observation_ambiguity_sandwich_is_supporting_not_flagship() -> None:
    payload = _load_payload()
    row = next(item for item in payload["theorems"] if item["theorem_id"] == "T10_T11_ObservationAmbiguitySandwich")
    assert row["defense_tier"] == "supporting_defended"
    assert row["rigor_rating"] == "proof_runtime_linked"
    assert row["code_correspondence"] == "matches"
    assert row["assumptions_used"] == []
    assert row["typed_obligations"]
    assert row["unresolved_assumptions"] == []
    assert row["dependencies"] == ["T10", "T11", "Common safe-core witness"]
    assert "global optimality" in row["scope_note"]
    assert "ObservationAmbiguitySandwich" in row["title"]


def test_equal_domain_artifact_package_rows_are_supporting_not_flagship() -> None:
    payload = _load_payload()
    rows = {
        item["theorem_id"]: item
        for item in payload["theorems"]
        if item["theorem_id"]
        in {
            "T_EQ_Battery_RuntimeArtifactPackage",
            "T_EQ_AV_RuntimeArtifactPackage",
            "T_EQ_HC_RuntimeArtifactPackage",
        }
    }

    assert set(rows) == {
        "T_EQ_Battery_RuntimeArtifactPackage",
        "T_EQ_AV_RuntimeArtifactPackage",
        "T_EQ_HC_RuntimeArtifactPackage",
    }
    for row in rows.values():
        assert row["defense_tier"] == "supporting_defended"
        assert row["rigor_rating"] == "artifact_runtime_linked"
        assert row["code_correspondence"] == "matches"
        assert row["assumptions_used"] == []
        assert row["typed_obligations"]
        assert row["code_anchors"]
        assert row["test_anchors"]
        if "full autonomous-driving field closure" in row["scope_note"].lower():
            assert "does not assert full autonomous-driving field closure" in row["scope_note"].lower()
        if "regulated clinical deployment readiness" in row["scope_note"].lower():
            assert "does not assert regulated clinical deployment readiness" in row["scope_note"].lower()


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

    assert flagship_ids == ["T1", "T2", "T3a", "T4", "T6", "T7", "T11", "T_trajectory_PAC"]
    assert supporting_ids == [
        "T3b",
        "T8",
        "T10_T11_ObservationAmbiguitySandwich",
        "T11_AV_BrakeHold",
        "T11_HC_FailSafeRelease",
        "T6_AV_FallbackValidity",
        "T6_HC_FallbackValidity",
        "T_EQ_Battery_RuntimeArtifactPackage",
        "T_EQ_AV_RuntimeArtifactPackage",
        "T_EQ_HC_RuntimeArtifactPackage",
        "T11_Byzantine",
        "T_stale_decay",
    ]
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
        "flagship_defended": 8,
        "supporting_defended": 12,
        "draft_non_defended": 9,
    }
    assert summary["flagship_gate_ready"] is True


def test_defended_core_is_generated_from_active_defended_registry_rows() -> None:
    core = json.loads(DEFENDED_CORE_JSON.read_text(encoding="utf-8"))
    row_ids = [row["theorem_id"] for row in core["rows"]]
    assert row_ids == [
        "T1",
        "T2",
        "T3a",
        "T3b",
        "T4",
        "T6",
        "T7",
        "T8",
        "T11",
        "T10_T11_ObservationAmbiguitySandwich",
        "T11_AV_BrakeHold",
        "T11_HC_FailSafeRelease",
        "T6_AV_FallbackValidity",
        "T6_HC_FallbackValidity",
        "T_EQ_Battery_RuntimeArtifactPackage",
        "T_EQ_AV_RuntimeArtifactPackage",
        "T_EQ_HC_RuntimeArtifactPackage",
        "T11_Byzantine",
        "T_stale_decay",
        "T_trajectory_PAC",
    ]
    assert "T9" not in row_ids
    assert "T10" not in row_ids
    assert core["summary"]["flagship_defended_ids"] == [
        "T1",
        "T2",
        "T3a",
        "T4",
        "T6",
        "T7",
        "T11",
        "T_trajectory_PAC",
    ]
    assert "T10_T11_ObservationAmbiguitySandwich" in core["summary"]["supporting_defended_ids"]


def test_assumption_map_separates_t11_typed_obligations_from_unresolved_assumptions() -> None:
    rows = list(csv.DictReader(ASSUMPTION_MAP_CSV.open()))
    t11_rows = [row for row in rows if row["theorem_id"] == "T11"]
    assert t11_rows
    assert {row["item_type"] for row in t11_rows} == {"typed_obligation"}
    assert {row["resolution_status"] for row in t11_rows} == {"runtime_linked"}
    assert [row["item"] for row in t11_rows] == [
        "Coverage obligation for the observation-consistent state set.",
        "Soundness of the tightened safe-action set.",
        "Repair membership in the tightened safe-action set.",
        "Fallback admissibility when the tightened set is empty.",
    ]

    scoped_unresolved = [
        row
        for row in rows
        if row["item_type"] == "theorem_local_assumption"
        and row["resolution_status"] == "scoped_unresolved"
    ]
    assert any(row["theorem_id"] == "T10" and "boundary-mass" in row["item"] for row in scoped_unresolved)
