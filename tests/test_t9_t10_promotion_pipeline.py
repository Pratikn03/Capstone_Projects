from __future__ import annotations

import csv
import json
from pathlib import Path

import scripts.build_t9_t10_assumption_discharge as discharge_builder
import scripts.build_t9_t10_mechanized_status as mechanized_builder
import scripts.build_t9_t10_research_package as research_builder
import scripts.validate_t9_t10_assumption_discharge as discharge_validator
import scripts.validate_t9_t10_research_package as research_validator
from orius.universal_theory.theorem_discharge import (
    DischargeThresholds,
    compute_t9_discharge_from_rows,
    compute_t10_discharge_from_rows,
)

REQUIRED_FAMILIES = {
    "lower_bounds",
    "mixing_processes",
    "runtime_assurance",
    "safety_filters",
    "conformal_shift",
    "av_validation",
    "battery_validation",
    "healthcare_validation",
}
T1_T10_IDS = {"T1", "T2", "T3a", "T3b", "T4", "T5", "T6", "T7", "T8", "T9", "T10"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_mechanized_status_tracks_t1_to_t10_kernels_without_fake_domain_discharge(tmp_path: Path) -> None:
    out_dir = tmp_path / "publication"

    payload = mechanized_builder.build_mechanized_status(out_dir=out_dir, run_lean=False)

    assert set(payload["theorems"]) >= T1_T10_IDS
    assert payload["mechanization_scope"] == "theorem_kernel_not_domain_discharge"
    for theorem in payload["theorems"].values():
        assert theorem["mechanization_scope"] == "theorem_kernel_not_domain_discharge"
        assert theorem["formal_file"] == "formal/Orius.lean"


def test_mechanized_status_tracks_every_active_theorem_program_kernel(tmp_path: Path) -> None:
    out_dir = tmp_path / "publication"
    active = json.loads((REPO_ROOT / "reports" / "publication" / "active_theorem_audit.json").read_text())
    expected_ids = {row["theorem_id"] for row in active["theorems"]}

    payload = mechanized_builder.build_mechanized_status(out_dir=out_dir, run_lean=False)

    assert set(payload["theorems"]) == expected_ids
    assert payload["root_module"] == "formal/Orius.lean"
    assert (REPO_ROOT / "formal" / "Orius" / "Program.lean").exists()
    assert (out_dir / "mechanized_theorem_program_status.json").exists()


def test_research_package_builder_emits_required_theory_surfaces(tmp_path: Path) -> None:
    out_dir = tmp_path / "publication"

    result = research_builder.build_research_package(
        out_dir=out_dir,
        min_sources=80,
        use_network=False,
    )

    matrix = _read_csv(out_dir / "t9_t10_research_source_matrix.csv")
    dependency = _read_csv(out_dir / "t9_t10_proof_dependency_matrix.csv")
    scorecard = json.loads((out_dir / "t9_t10_research_scorecard.json").read_text(encoding="utf-8"))

    assert result["source_count"] >= 80
    assert len(matrix) >= 80
    assert {row["topic_family"] for row in matrix} >= REQUIRED_FAMILIES
    assert {row["theorem_id"] for row in dependency} >= {"T9", "T10"}
    assert scorecard["proof_dependency_complete"] is True
    assert scorecard["source_count"] == len(matrix)
    assert all(row["doi_or_url"] for row in matrix)
    assert all(row["provenance"] for row in matrix)


def test_research_validator_rejects_duplicate_padding_and_missing_provenance(tmp_path: Path) -> None:
    out_dir = tmp_path / "publication"
    research_builder.build_research_package(out_dir=out_dir, min_sources=32, use_network=False)
    matrix_path = out_dir / "t9_t10_research_source_matrix.csv"
    rows = _read_csv(matrix_path)
    duplicate = dict(rows[0])
    duplicate["source_id"] = "DUPLICATE_PAD"
    duplicate["provenance"] = ""
    rows = rows[:20] + [duplicate for _ in range(20)]
    _write_csv(matrix_path, rows)

    result = research_validator.validate_research_package(out_dir=out_dir, min_sources=32)

    assert result["pass"] is False
    joined = "\n".join(result["findings"])
    assert "provenance" in joined
    assert "duplicate" in joined


def test_assumption_discharge_builder_keeps_t9_t10_blocked_without_domain_bridges(tmp_path: Path) -> None:
    out_dir = tmp_path / "publication"

    result = discharge_builder.build_assumption_discharge(
        out_dir=out_dir,
        thresholds=DischargeThresholds(min_rows=8),
        max_rows=64,
    )

    for theorem_id in ("T9", "T10"):
        for domain in ("battery", "av", "healthcare"):
            payload = json.loads(
                (out_dir / "theorem_promotion_evidence" / f"{theorem_id}_{domain}.json").read_text(
                    encoding="utf-8"
                )
            )
            assert isinstance(payload["promotion_ready"], bool)
            assert payload["artifact_source"]
            assert payload["source_trace_path"]
            assert "n_usable_rows" in payload
            if not payload["promotion_ready"]:
                assert payload["blocker"]

    validation = discharge_validator.validate_assumption_discharge(out_dir=out_dir)
    assert validation["pass"] is True
    assert validation["promotion_ready"] == result["promotion_ready"]


def test_assumption_discharge_validator_blocks_false_promotion(tmp_path: Path) -> None:
    out_dir = tmp_path / "publication"
    discharge_builder.build_assumption_discharge(
        out_dir=out_dir,
        thresholds=DischargeThresholds(min_rows=8),
        max_rows=64,
    )
    evidence_path = out_dir / "theorem_promotion_evidence" / "T10_av.json"
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["promotion_ready"] = True
    payload["tv_bridge_status"] = "missing"
    payload["unsafe_boundary_mass_status"] = "missing"
    payload["blocker"] = ""
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    result = discharge_validator.validate_assumption_discharge(out_dir=out_dir)

    assert result["pass"] is False
    assert any("T10_av" in finding and "promotion_ready" in finding for finding in result["findings"])


def test_t9_discharge_computes_real_positive_witness_and_mixing_proxy() -> None:
    thresholds = DischargeThresholds(min_rows=8, boundary_margin=0.5, min_positive_rate=0.01)
    rows = [
        {
            "reliability_w": 1.0,
            "fault_family": "nominal",
            "true_margin": 5.0,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.72,
            "fault_family": "dropout",
            "true_margin": 0.1,
            "true_constraint_violated": True,
        },
        {
            "reliability_w": 1.0,
            "fault_family": "nominal",
            "true_margin": 6.0,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.66,
            "fault_family": "blackout",
            "true_margin": 0.2,
            "true_constraint_violated": True,
        },
        {
            "reliability_w": 0.91,
            "fault_family": "nominal",
            "true_margin": 0.3,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.58,
            "fault_family": "dropout",
            "true_margin": 0.2,
            "true_constraint_violated": True,
        },
        {
            "reliability_w": 1.0,
            "fault_family": "nominal",
            "true_margin": 5.0,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.75,
            "fault_family": "out_of_order",
            "true_margin": 0.4,
            "true_constraint_violated": True,
        },
    ]

    payload = compute_t9_discharge_from_rows(
        rows,
        domain="battery",
        artifact_source="reports/publication/theorem_promotion_evidence/T9_battery.json",
        thresholds=thresholds,
    )

    assert payload["promotion_ready"] is True
    assert payload["witness_constant"] > 0.0
    assert payload["degradation_rate"] > 0.0
    assert payload["boundary_reachability_rate"] > 0.0
    assert payload["mixing_proxy"]["finite_mixing_proxy"] is True
    assert payload["blocker"] == ""


def test_t9_discharge_accepts_multi_lag_mixing_witness() -> None:
    thresholds = DischargeThresholds(
        min_rows=32,
        boundary_margin=0.5,
        min_positive_rate=0.01,
        mixing_autocorrelation_max=0.10,
        mixing_max_lag=6,
    )
    pattern = [True, True, True, True, False, False, False, False] * 5
    rows = [
        {
            "reliability_w": 0.70 if degraded else 1.0,
            "fault_family": "dropout" if degraded else "nominal",
            "true_margin": 0.1 if degraded else 5.0,
            "true_constraint_violated": degraded,
        }
        for degraded in pattern
    ]

    payload = compute_t9_discharge_from_rows(
        rows,
        domain="battery",
        artifact_source="reports/publication/theorem_promotion_evidence/T9_battery.json",
        thresholds=thresholds,
    )

    assert payload["promotion_ready"] is True
    assert payload["mixing_proxy"]["finite_mixing_proxy"] is True
    assert payload["mixing_proxy"]["selected_lag"] == 2
    assert payload["mixing_proxy"]["lag1_autocorrelation"] > thresholds.mixing_autocorrelation_max
    assert payload["blocker"] == ""


def test_t9_discharge_blocks_zero_witness_constant() -> None:
    thresholds = DischargeThresholds(min_rows=4, boundary_margin=0.5, min_positive_rate=0.01)
    rows = [
        {
            "reliability_w": 1.0,
            "fault_family": "nominal",
            "true_margin": 5.0,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 1.0,
            "fault_family": "nominal",
            "true_margin": 6.0,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 1.0,
            "fault_family": "nominal",
            "true_margin": 7.0,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 1.0,
            "fault_family": "nominal",
            "true_margin": 8.0,
            "true_constraint_violated": False,
        },
    ]

    payload = compute_t9_discharge_from_rows(
        rows,
        domain="av",
        artifact_source="reports/publication/theorem_promotion_evidence/T9_av.json",
        thresholds=thresholds,
    )

    assert payload["promotion_ready"] is False
    assert payload["witness_constant"] == 0.0
    assert "witness_constant" in payload["blocker"]


def test_t10_discharge_computes_tv_bridge_boundary_mass_and_le_cam_inputs() -> None:
    thresholds = DischargeThresholds(
        min_rows=8, boundary_margin=0.5, min_positive_rate=0.01, tv_bridge_epsilon=0.10
    )
    rows = [
        {
            "reliability_w": 0.70,
            "observed_margin": 0.10,
            "true_margin": 0.10,
            "true_constraint_violated": True,
        },
        {
            "reliability_w": 0.68,
            "observed_margin": 0.12,
            "true_margin": 0.20,
            "true_constraint_violated": True,
        },
        {
            "reliability_w": 0.72,
            "observed_margin": 0.14,
            "true_margin": 0.30,
            "true_constraint_violated": True,
        },
        {
            "reliability_w": 0.69,
            "observed_margin": 0.20,
            "true_margin": 0.40,
            "true_constraint_violated": True,
        },
        {
            "reliability_w": 0.74,
            "observed_margin": 0.11,
            "true_margin": 0.10,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.76,
            "observed_margin": 0.15,
            "true_margin": 0.20,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.75,
            "observed_margin": 0.17,
            "true_margin": 0.30,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.77,
            "observed_margin": 0.19,
            "true_margin": 0.40,
            "true_constraint_violated": False,
        },
    ]

    payload = compute_t10_discharge_from_rows(
        rows,
        domain="av",
        artifact_source="reports/publication/theorem_promotion_evidence/T10_av.json",
        thresholds=thresholds,
    )

    assert payload["promotion_ready"] is True
    assert payload["unsafe_boundary_mass"] > 0.0
    assert payload["tv_bridge"]["passed"] is True
    assert payload["le_cam_lower_bound"] >= 0.0
    assert payload["blocker"] == ""


def test_t10_discharge_uses_auxiliary_unsafe_observation_law() -> None:
    thresholds = DischargeThresholds(
        min_rows=10, boundary_margin=0.5, min_positive_rate=0.01, tv_bridge_epsilon=0.10
    )
    safe_rows = [
        {
            "reliability_w": 0.70,
            "observed_margin": 0.10 + 0.01 * idx,
            "true_margin": 1.0,
            "true_constraint_violated": False,
        }
        for idx in range(8)
    ]
    unsafe_rows = [
        {
            "reliability_w": 0.70,
            "observed_margin": 0.10 + 0.01 * idx,
            "true_margin": -0.1,
            "true_constraint_violated": True,
        }
        for idx in range(2)
    ]

    payload = compute_t10_discharge_from_rows(
        safe_rows,
        domain="battery",
        artifact_source="reports/publication/theorem_promotion_evidence/T10_battery.json",
        thresholds=thresholds,
        auxiliary_unsafe_rows=unsafe_rows,
    )

    assert payload["promotion_ready"] is True
    assert payload["n_auxiliary_unsafe_rows"] == 2
    assert payload["unsafe_boundary_mass"] > 0.0
    assert payload["boundary_testing_subproblem_status"] == "boundary_testing_subproblem_constructed"
    assert payload["tv_bridge"]["passed"] is True
    assert payload["blocker"] == ""


def test_t10_discharge_blocks_missing_boundary_mass_and_invalid_tv_bridge() -> None:
    thresholds = DischargeThresholds(
        min_rows=6, boundary_margin=0.5, min_positive_rate=0.01, tv_bridge_epsilon=0.0
    )
    rows = [
        {
            "reliability_w": 0.20,
            "observed_margin": 10.0,
            "true_margin": 10.0,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.20,
            "observed_margin": 11.0,
            "true_margin": 11.0,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.20,
            "observed_margin": 12.0,
            "true_margin": 12.0,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.20,
            "observed_margin": 13.0,
            "true_margin": 13.0,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.20,
            "observed_margin": 14.0,
            "true_margin": 14.0,
            "true_constraint_violated": False,
        },
        {
            "reliability_w": 0.20,
            "observed_margin": 15.0,
            "true_margin": 15.0,
            "true_constraint_violated": False,
        },
    ]

    payload = compute_t10_discharge_from_rows(
        rows,
        domain="healthcare",
        artifact_source="reports/publication/theorem_promotion_evidence/T10_healthcare.json",
        thresholds=thresholds,
    )

    assert payload["promotion_ready"] is False
    assert payload["unsafe_boundary_mass"] == 0.0
    assert "unsafe_boundary_mass" in payload["blocker"]
