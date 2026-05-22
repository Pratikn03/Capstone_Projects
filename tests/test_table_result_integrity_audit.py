from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import audit_table_result_integrity as audit


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_default_audit_blocks_current_surfaces_but_demotes_legacy_noise(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    _write_csv(
        tmp_path / "reports/publication/three_domain_ml_benchmark.csv",
        [{"domain": "Battery Energy Storage", "result": ""}],
    )
    _write_csv(
        tmp_path / "reports/publication/legacy_exploratory_table.csv",
        [{"domain": "archived", "result": ""}],
    )

    findings, summary = audit.run_audit(["reports/publication"])

    by_path = {finding.path: finding for finding in findings}
    assert summary["blocking_count"] == 1
    assert by_path["reports/publication/three_domain_ml_benchmark.csv"].blocking is True
    assert by_path["reports/publication/legacy_exploratory_table.csv"].blocking is False
    assert by_path["reports/publication/legacy_exploratory_table.csv"].severity == "warning"


def test_current_allowlisted_semantic_blanks_do_not_block(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    _write_csv(
        tmp_path / "reports/publication/utility_preserving_safety_scorecard.csv",
        [
            {
                "domain": "Battery Energy Storage",
                "claim_scope": "T8 graceful degradation useful-work frontier",
                "orius_intervention_rate": "",
                "safety_reference_intervention_rate": "",
                "intervention_reduction_vs_safety_reference": "",
                "orius_fallback_rate": "",
                "safety_reference_fallback_rate": "",
                "fallback_reduction_vs_safety_reference": "",
                "claim_boundary": "Battery intervention/fallback rates are reported in the runtime TSVR table.",
            }
        ],
    )
    scorecard_json = {
        "rows": [
            {
                "domain": "Battery Energy Storage",
                "claim_scope": "T8 graceful degradation useful-work frontier",
                "orius_intervention_rate": "",
                "safety_reference_intervention_rate": "",
                "intervention_reduction_vs_safety_reference": "",
                "orius_fallback_rate": "",
                "safety_reference_fallback_rate": "",
                "fallback_reduction_vs_safety_reference": "",
                "claim_boundary": "Battery intervention/fallback rates are reported in the runtime TSVR table.",
            }
        ]
    }
    json_path = tmp_path / "reports/publication/utility_preserving_safety_scorecard.json"
    json_path.write_text(json.dumps(scorecard_json), encoding="utf-8")

    findings, summary = audit.run_audit(
        [
            "reports/publication/utility_preserving_safety_scorecard.csv",
            "reports/publication/utility_preserving_safety_scorecard.json",
        ]
    )

    assert summary["blocking_count"] == 0
    assert findings == []


def test_utility_scorecard_json_companion_rows_allow_comparison_specific_blanks(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    scorecard_json = {
        "claim_comparison_rows": [
            {
                "domain": "Autonomous Vehicles",
                "comparison": "predictor_only_safety",
                "comparability": "comparable_runtime_native",
                "reference_utility": "",
                "orius_utility": "",
                "utility_delta": "",
                "reference_fallback_rate": "",
                "orius_fallback_rate": "",
                "fallback_delta": "",
            }
        ],
        "ablation_surface_rows": [
            {
                "domain": "Cross-domain governance",
                "requested_surface": "no_signature_hash_gate",
                "comparability": "non_comparable_governance_gate",
                "baseline_controller": "",
                "baseline_tsvr": "",
                "orius_tsvr": "",
                "absolute_tsvr_reduction": "",
                "relative_tsvr_reduction": "",
                "baseline_intervention_rate": "",
                "orius_intervention_rate": "",
            }
        ],
    }
    json_path = tmp_path / "reports/publication/utility_preserving_safety_scorecard.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(scorecard_json), encoding="utf-8")

    findings, summary = audit.run_audit(["reports/publication/utility_preserving_safety_scorecard.json"])

    assert summary["blocking_count"] == 0
    assert findings == []


def test_theorem_defensibility_allows_empty_lake_output_only_when_formal_passes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(audit, "REPO_ROOT", tmp_path)
    theorem_gate = {
        "formal": {
            "pass": True,
            "checks": {"formal_core_lake_build": True},
            "lake_output": "",
        }
    }
    json_path = tmp_path / "reports/publication/theorem_defensibility_10.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(theorem_gate), encoding="utf-8")

    findings, summary = audit.run_audit(["reports/publication/theorem_defensibility_10.json"])

    assert summary["blocking_count"] == 0
    assert findings == []
