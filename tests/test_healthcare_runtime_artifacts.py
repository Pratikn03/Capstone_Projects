from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import duckdb

from orius.adapters.healthcare import HealthcareTrackAdapter


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_healthcare_runtime_artifacts.py"
DATASET_PATH = REPO_ROOT / "data" / "healthcare" / "mimic3" / "processed" / "mimic3_healthcare_orius.csv"


def _load_script():
    spec = importlib.util.spec_from_file_location("build_healthcare_runtime_artifacts", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


healthcare_runtime_script = _load_script()


def test_healthcare_track_blackout_observation_is_not_treated_as_safe() -> None:
    track = HealthcareTrackAdapter()
    state = track.reset(seed=42)
    observed = track.observe(state, {"kind": "blackout"})

    assert track.observed_constraint_satisfied(observed) is None
    assert track.constraint_margin(observed) is None


def test_build_healthcare_runtime_artifacts_emits_domain_native_runtime_surfaces(tmp_path: Path) -> None:
    report = healthcare_runtime_script.build_healthcare_runtime_artifacts(
        dataset_path=DATASET_PATH,
        out_dir=tmp_path,
        seeds=2,
        horizon=6,
        start_seed=2000,
    )

    traces_path = Path(report["runtime_traces_csv"])
    summary_path = Path(report["runtime_summary_csv"])
    db_path = Path(report["healthcare_runtime_duckdb"])
    governance_path = Path(report["runtime_governance_summary_csv"])
    certos_path = Path(report["certos_verification_summary_json"])
    comparison_path = Path(report["runtime_comparison_csv"])
    comparator_path = Path(report["runtime_comparator_summary_csv"])
    comparator_traces_path = Path(report["runtime_comparator_traces_csv"])
    ablation_path = Path(report["runtime_ablation_summary_csv"])
    negative_path = Path(report["runtime_negative_controls_csv"])

    assert traces_path.exists()
    assert summary_path.exists()
    assert db_path.exists()
    assert governance_path.exists()
    assert certos_path.exists()
    assert comparison_path.exists()
    assert comparator_path.exists()
    assert comparator_traces_path.exists()
    assert ablation_path.exists()
    assert negative_path.exists()

    with traces_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {
        "controller",
        "patient_id",
        "true_constraint_violated",
        "observed_constraint_satisfied",
        "fallback_used",
        "certificate_valid",
        "certificate_predicted_valid",
        "validity_horizon_H_t",
        "validity_status",
        "validity_scope",
        "validity_theorem_id",
        "validity_theorem_contract",
        "intervention_reason",
        "repair_mode",
        "contract_id",
        "source_theorem",
        "t11_status",
        "t11_failed_obligations",
        "domain_postcondition_passed",
        "domain_postcondition_failure",
        "projected_release",
        "projected_release_margin",
        "runtime_policy_family",
    } <= set(rows[0])
    orius_rows = [row for row in rows if row["controller"] == "orius"]
    assert orius_rows
    assert {row["contract_id"] for row in orius_rows} == {"HC.T11.fail_safe_release_runtime_lemma"}
    assert {row["source_theorem"] for row in orius_rows} == {"T11"}
    assert {row["t11_status"] for row in orius_rows} <= {"runtime_linked", "contract_violation"}
    assert {row["domain_postcondition_passed"] for row in orius_rows} <= {"True", "False"}
    assert all(row["domain_postcondition_failure"] for row in orius_rows)

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = {row["controller"]: row for row in csv.DictReader(handle)}
    assert {
        "baseline",
        "ews_threshold",
        "conformal_alert_only",
        "predictor_only_no_runtime",
        "fixed_conservative_alert",
        "stale_certificate_no_temporal_guard",
        "always_alert",
        "orius",
    } <= set(summary_rows)
    assert float(summary_rows["orius"]["cva"]) >= 0.0
    assert float(summary_rows["orius"]["intervention_rate"]) > 0.0
    assert float(summary_rows["orius"]["tsvr"]) == 0.0
    assert float(summary_rows["always_alert"]["tsvr"]) == 0.0
    assert float(summary_rows["orius"]["max_alert_rate"]) <= 0.50

    comparator_rows = {
        row["baseline_family"]: row
        for row in csv.DictReader(comparator_path.open("r", encoding="utf-8", newline=""))
    }
    assert set(comparator_rows) == {
        "nominal_deterministic_controller",
        "fixed_threshold_or_fixed_inflation_runtime",
        "standard_conformal_nonreliability_runtime",
        "no_quality_signal_runtime",
        "no_adaptive_response_runtime",
        "no_temporal_guard_or_no_certificate_refresh_runtime",
        "orius_full_stack",
        "degenerate_fallback_runtime",
    }
    assert {row["metric_surface"] for row in comparator_rows.values()} == {"runtime_denominator"}
    assert "proxy_current_shared_harness" not in {row["evidence_status"] for row in comparator_rows.values()}
    assert float(comparator_rows["orius_full_stack"]["runtime_witness_pass_rate"]) == 1.0
    assert float(comparator_rows["orius_full_stack"]["fallback_activation_rate"]) <= 0.50
    assert float(comparator_rows["orius_full_stack"]["useful_work_total"]) >= float(
        comparator_rows["degenerate_fallback_runtime"]["useful_work_total"]
    )
    independent_rows = [
        row
        for family, row in comparator_rows.items()
        if family not in {"orius_full_stack", "degenerate_fallback_runtime"}
    ]
    assert {str(row["independent_baseline"]) for row in independent_rows} == {"True"}
    assert len({row["controller"] for row in independent_rows}) == len(independent_rows)
    ablation_rows = list(csv.DictReader(ablation_path.open("r", encoding="utf-8", newline="")))
    negative_rows = list(csv.DictReader(negative_path.open("r", encoding="utf-8", newline="")))
    assert {row["metric_surface"] for row in ablation_rows} == {"runtime_denominator"}
    assert {row["surface"] for row in negative_rows} == {"runtime_denominator"}

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info('dispatch_certificates')").fetchall()
        }
        runtime_trace_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info('healthcare_runtime_traces')").fetchall()
        }
        typed = conn.execute(
            """
            SELECT
                count(*) AS n,
                min(validity_horizon_H_t) AS min_horizon,
                max(validity_horizon_H_t) AS max_horizon,
                min(length(payload_json)) AS min_payload_len
            FROM dispatch_certificates
            """
        ).fetchone()
    finally:
        conn.close()

    assert {"payload_json", "validity_horizon_H_t", "half_life_steps", "expires_at_step"} <= columns
    assert {
        "patient_id",
        "fallback_used",
        "certificate_valid",
        "intervention_reason",
        "contract_id",
        "t11_status",
        "domain_postcondition_passed",
        "domain_postcondition_failure",
        "validity_scope",
        "validity_theorem_id",
        "validity_theorem_contract",
    } <= runtime_trace_columns
    assert int(typed[0]) == 12
    assert int(typed[1]) >= 0
    assert int(typed[2]) >= int(typed[1])
    assert int(typed[3]) > 0

    governance_rows = {
        row["metric"]: row["value"]
        for row in csv.DictReader(governance_path.open("r", encoding="utf-8", newline=""))
    }
    assert governance_rows["chain_valid"] == "1.0"
    assert governance_rows["expiry_metadata_presence_rate"] == "1.0"

    certos_summary = json.loads(certos_path.read_text(encoding="utf-8"))
    assert certos_summary["chain_valid"] is True
    assert certos_summary["expiry_metadata_presence_rate"] == 1.0
