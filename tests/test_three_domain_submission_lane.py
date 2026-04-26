from __future__ import annotations

import csv
import json
from pathlib import Path
import tomllib

from scripts._dataset_registry import DATASET_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMOTED_AV_RUNTIME_DIR = REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
PROMOTED_RUNTIME_MAX_TSVR = 1e-3
PROMOTED_RUNTIME_MIN_PASS_RATE = 1.0 - PROMOTED_RUNTIME_MAX_TSVR


def test_healthcare_registry_uses_promoted_mimic_paths() -> None:
    healthcare = DATASET_REGISTRY["HEALTHCARE"]
    assert healthcare.raw_data_path == "data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv"
    assert healthcare.canonical_runtime_path == "data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv"
    assert healthcare.runtime_provenance_path == "data/healthcare/mimic3/processed/mimic3_manifest.json"


def test_three_domain_scorecard_row_is_green() -> None:
    rows = {
        row["target_tier"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "publication" / "orius_submission_scorecard.csv").open())
    }
    row = rows["three_domain_93_candidate"]
    assert row["meets_93_gate"] == "True"
    assert row["critical_gap_count"] == "0"
    assert row["high_gap_count"] == "0"


def test_promoted_lane_artifacts_are_sanitized_and_healthcare_promoted() -> None:
    release_summary = (REPO_ROOT / "reports" / "battery_av_healthcare" / "overall" / "release_summary.json").read_text(
        encoding="utf-8"
    )
    override = json.loads(
        (REPO_ROOT / "reports" / "battery_av_healthcare" / "overall" / "publication_closure_override.json").read_text(
            encoding="utf-8"
        )
    )
    closure_rows = {
        row["domain"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "publication" / "orius_domain_closure_matrix.csv").open())
    }

    assert "/Users/" not in release_summary
    assert "battery_av_only" not in release_summary
    assert override["vehicle"]["resulting_tier"] == "runtime_contract_closed"
    assert override["healthcare"]["resulting_tier"] == "runtime_contract_closed"
    assert "mimic3_manifest.json" in release_summary
    assert closure_rows["Medical and Healthcare Monitoring"]["tier"] == "runtime_contract_closed"
    assert "promoted source=MIMIC" in closure_rows["Medical and Healthcare Monitoring"]["current_status"]
    assert "Industrial Process Control" not in closure_rows


def test_runtime_denominator_surfaces_are_claim_governing_and_proxy_is_secondary() -> None:
    closure_rows = list(csv.DictReader((REPO_ROOT / "reports" / "publication" / "orius_domain_closure_matrix.csv").open()))
    evidence_rows = {
        row["domain"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "publication" / "chapters40_44_domain_evidence_register.csv").open())
    }
    proxy_rows = {
        row["domain"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "publication" / "three_domain_proxy_runtime_comparison.csv").open())
    }

    by_domain = {row["domain"]: row for row in closure_rows}
    for domain in ("Autonomous Vehicles", "Medical and Healthcare Monitoring"):
        status = by_domain[domain]["current_status"]
        assert "runtime denominator" in status
        assert "secondary proxy" in status
        if domain == "Autonomous Vehicles":
            assert "nuplan_allzip_grouped_runtime_replay_surrogate" in status
            assert "no road deployment" in status
            assert "full autonomous-driving field closure" in status

        evidence = evidence_rows[domain]
        assert evidence["metric_basis"] == "runtime denominator benchmark"
        assert "runtime denominator" in evidence["evidence_volume_note"].lower()

        proxy = proxy_rows[domain]
        assert proxy["claim_governs_from"] == "runtime_denominator"
        assert proxy["proxy_metric_surface"] == "validation_harness"
        assert proxy["diagnostic_only"] == "True"


def test_runtime_governing_metrics_match_runtime_summaries() -> None:
    av_summary_rows = {
        row["controller"]: row
        for row in csv.DictReader((PROMOTED_AV_RUNTIME_DIR / "runtime_summary.csv").open())
    }
    healthcare_summary_rows = {
        row["controller"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "healthcare" / "runtime_summary.csv").open())
    }
    benchmark_rows = {
        row["domain"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "publication" / "three_domain_ml_benchmark.csv").open())
    }
    closure_rows = {
        row["domain"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "publication" / "orius_domain_closure_matrix.csv").open())
    }

    promoted_domains = {
        "Autonomous Vehicles": av_summary_rows,
        "Medical and Healthcare Monitoring": healthcare_summary_rows,
    }
    for display_name, runtime_summary in promoted_domains.items():
        benchmark = benchmark_rows[display_name]
        closure = closure_rows[display_name]

        assert benchmark["metric_surface"] == "runtime_denominator"
        assert benchmark["baseline_tsvr_mean"] == f"{float(runtime_summary['baseline']['tsvr']):.6f}"
        assert benchmark["orius_tsvr_mean"] == f"{float(runtime_summary['orius']['tsvr']):.6f}"
        assert benchmark["strict_runtime_gate"] == "True"
        assert float(benchmark["t11_pass_rate"]) >= PROMOTED_RUNTIME_MIN_PASS_RATE
        assert float(benchmark["postcondition_pass_rate"]) >= PROMOTED_RUNTIME_MIN_PASS_RATE
        assert float(benchmark["runtime_witness_pass_rate"]) >= PROMOTED_RUNTIME_MIN_PASS_RATE
        assert closure["baseline_tsvr"] == f"{float(runtime_summary['baseline']['tsvr']):.4f}"
        assert closure["orius_tsvr"] == f"{float(runtime_summary['orius']['tsvr']):.4f}"
        assert "runtime-governed" in benchmark["note"] or "runtime-governed" in benchmark["note"].replace("runtime-governing", "runtime-governed")


def test_healthcare_benchmark_uses_runtime_emitted_certificate_metrics() -> None:
    benchmark_rows = {
        row["domain"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "publication" / "three_domain_ml_benchmark.csv").open())
    }
    summary_rows = {
        row["controller"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "healthcare" / "runtime_summary.csv").open())
    }
    trace_rows = [
        row
        for row in csv.DictReader((REPO_ROOT / "reports" / "healthcare" / "runtime_traces.csv").open())
        if row["controller"] == "orius"
    ]

    healthcare = benchmark_rows["Medical and Healthcare Monitoring"]
    orius_summary = summary_rows["orius"]
    fallback_rate = sum(row["fallback_used"] == "True" for row in trace_rows) / len(trace_rows)

    witness_summary = json.loads(
        (REPO_ROOT / "reports" / "publication" / "domain_runtime_contract_summary.json").read_text(encoding="utf-8")
    )

    assert healthcare["certificate_valid_release_rate_semantics"] == "runtime_witness_certificate_valid_rate_promoted_healthcare_row"
    assert healthcare["runtime_source"] == "reports/healthcare/runtime_summary.csv"
    assert healthcare["certificate_valid_release_rate"] == f"{float(witness_summary['domains']['healthcare']['certificate_valid_rate']):.6f}"
    assert healthcare["intervention_rate"] == f"{float(orius_summary['intervention_rate']):.6f}"
    assert healthcare["fallback_activation_rate"] == f"{fallback_rate:.6f}"
    assert "secondary proxy" in healthcare["note"]


def test_av_and_healthcare_emit_domain_runtime_contract_witnesses() -> None:
    summary_path = REPO_ROOT / "reports" / "publication" / "domain_runtime_contract_summary.json"
    witnesses_path = REPO_ROOT / "reports" / "publication" / "domain_runtime_contract_witnesses.csv"
    assert summary_path.exists()
    assert witnesses_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["source_theorem"] == "T11"
    assert {"av", "healthcare"} <= set(summary["domains"])

    trace_specs = {
        "av": (
            PROMOTED_AV_RUNTIME_DIR / "runtime_traces.csv",
            "AV.T11.brake_hold_runtime_lemma",
        ),
        "healthcare": (
            REPO_ROOT / "reports" / "healthcare" / "runtime_traces.csv",
            "HC.T11.fail_safe_release_runtime_lemma",
        ),
    }
    required = {
        "contract_id",
        "source_theorem",
        "t11_status",
        "t11_failed_obligations",
        "domain_postcondition_passed",
        "domain_postcondition_failure",
        "validity_scope",
        "validity_theorem_id",
        "validity_theorem_contract",
    }
    for domain, (path, contract_id) in trace_specs.items():
        with path.open() as handle:
            reader = csv.DictReader(handle)
            first = next(reader)
            assert required <= set(first)
            sample = first if first["controller"] == "orius" else next(row for row in reader if row["controller"] == "orius")
        assert sample["contract_id"] == contract_id
        assert sample["source_theorem"] == "T11"
        assert sample["t11_status"]
        if domain == "av":
            runtime_summary = {
                row["controller"]: row
                for row in csv.DictReader((PROMOTED_AV_RUNTIME_DIR / "runtime_summary.csv").open())
            }
        else:
            runtime_summary = {
                row["controller"]: row
                for row in csv.DictReader((REPO_ROOT / "reports" / "healthcare" / "runtime_summary.csv").open())
            }
        assert int(summary["domains"][domain]["n_steps"]) == int(float(runtime_summary["orius"]["n_steps"]))
        assert summary["domains"][domain]["t11_pass_rate"] >= PROMOTED_RUNTIME_MIN_PASS_RATE
        assert summary["domains"][domain]["postcondition_pass_rate"] >= PROMOTED_RUNTIME_MIN_PASS_RATE
        assert summary["domains"][domain]["witness_pass_rate"] >= PROMOTED_RUNTIME_MIN_PASS_RATE


def test_non_degenerate_runtime_comparators_remain_visible() -> None:
    av_summary_rows = {
        row["controller"]: row
        for row in csv.DictReader((PROMOTED_AV_RUNTIME_DIR / "runtime_summary.csv").open())
    }
    healthcare_summary_rows = {
        row["controller"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "healthcare" / "runtime_summary.csv").open())
    }

    assert float(av_summary_rows["orius"]["tsvr"]) <= PROMOTED_RUNTIME_MAX_TSVR
    assert float(av_summary_rows["always_brake"]["tsvr"]) == 0.0
    assert float(av_summary_rows["orius"]["useful_work_total"]) > float(av_summary_rows["always_brake"]["useful_work_total"])

    assert float(healthcare_summary_rows["orius"]["tsvr"]) == 0.0
    assert float(healthcare_summary_rows["always_alert"]["tsvr"]) == 0.0
    assert float(healthcare_summary_rows["orius"]["max_alert_rate"]) < float(healthcare_summary_rows["always_alert"]["max_alert_rate"])


def test_readme_reflects_three_domain_submission_lane() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "battery_av_healthcare" in readme
    assert "mimic3_healthcare_orius.csv" in readme
    assert "equal_domain_93" not in readme


def test_full_check_testclient_dependency_is_locked() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(pyproject["project"]["dependencies"])
    lock_lines = set((REPO_ROOT / "requirements.lock.txt").read_text(encoding="utf-8").splitlines())

    assert "httpx==0.28.1" in dependencies
    assert "httpx==0.28.1" in lock_lines
    assert "httpcore==1.0.9" in lock_lines
