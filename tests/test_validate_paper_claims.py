from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.validate_paper_claims import (
    _check_claim_matrix,
    _check_healthcare_runtime_metric_alignment,
    _check_impact_alignment_with_manifest,
    _check_promoted_validation_surface_alignment,
    _check_reference_domain_validation_alignment,
)


def test_check_claim_matrix_accepts_historical_and_inactive_rows(tmp_path: Path) -> None:
    claim_matrix = tmp_path / "claim_matrix.csv"
    claim_matrix.write_text(
        "\n".join(
            [
                "claim_id,status,category,manuscript_locations,claim_text,canonical_value,source_file,source_locator,run_id,timestamp_utc,rounding_rule,notes",
                "C001,Verified,Metric,paper/paper.tex,Active claim,1.0,source.csv,row,,,n/a,ok",
                "C002,Historical,Metric,historical_prelock_draft_md,Old claim,old,source.csv,row,,,n/a,ok",
                "C003,Inactive,Metric,NOT PRESENT,Future claim,none,None,None,,,n/a,ok",
            ]
        ),
        encoding="utf-8",
    )

    findings = []
    _check_claim_matrix(findings, claim_matrix)

    assert findings == []


def test_check_claim_matrix_flags_invalid_historical_mapping(tmp_path: Path) -> None:
    claim_matrix = tmp_path / "claim_matrix.csv"
    claim_matrix.write_text(
        "\n".join(
            [
                "claim_id,status,category,manuscript_locations,claim_text,canonical_value,source_file,source_locator,run_id,timestamp_utc,rounding_rule,notes",
                "C001,Historical,Metric,paper/paper.tex,Old claim,old,source.csv,row,,,n/a,broken",
            ]
        ),
        encoding="utf-8",
    )

    findings = []
    _check_claim_matrix(findings, claim_matrix)

    assert any("Historical claim C001" in finding.detail for finding in findings)


def test_check_impact_alignment_with_manifest_flags_drift(tmp_path: Path) -> None:
    manifest = {
        "canonical_metrics": {
            "de": {
                "impact": {
                    "cost_savings_pct_raw": 0.07105244430986107,
                    "carbon_reduction_pct_raw": 0.0030350935898837994,
                    "peak_shaving_pct_raw": 0.06125995778380988,
                }
            },
            "us": {
                "impact": {
                    "cost_savings_pct_raw": 0.0011,
                    "carbon_reduction_pct_raw": 0.0002,
                    "peak_shaving_pct_raw": 0.0,
                }
            },
        }
    }
    de_path = tmp_path / "reports" / "impact_summary.csv"
    us_path = tmp_path / "reports" / "eia930" / "impact_summary.csv"
    de_path.parent.mkdir(parents=True, exist_ok=True)
    us_path.parent.mkdir(parents=True, exist_ok=True)
    de_path.write_text(
        "baseline_cost_usd,orius_cost_usd,cost_savings_pct,baseline_carbon_kg,orius_carbon_kg,carbon_reduction_pct,baseline_peak_mw,orius_peak_mw,peak_shaving_pct,oracle_cost_usd,oracle_gap_pct,carbon_source\n"
        "1,1,0.11,1,1,0.22,1,1,0.33,1,0,average\n",
        encoding="utf-8",
    )
    us_path.write_text(
        "baseline_cost_usd,orius_cost_usd,cost_savings_pct,baseline_carbon_kg,orius_carbon_kg,carbon_reduction_pct,baseline_peak_mw,orius_peak_mw,peak_shaving_pct,oracle_cost_usd,oracle_gap_pct,carbon_source\n"
        "1,1,0.0011,1,1,0.0002,1,1,0.0,1,0,average\n",
        encoding="utf-8",
    )

    findings = []
    _check_impact_alignment_with_manifest(findings, manifest, tmp_path)

    assert any(f.check == "impact_alignment" and "cost_savings_pct=0.11" in f.detail for f in findings)


def test_check_reference_domain_validation_alignment_flags_noncanonical_battery_row(tmp_path: Path) -> None:
    publication_path = tmp_path / "reports" / "publication" / "dc3s_main_table.csv"
    publication_path.parent.mkdir(parents=True, exist_ok=True)
    with publication_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "controller", "violation_rate"])
        writer.writeheader()
        for rate in ("0.083333", "0.059524", "0.017857", "0.017857", "0.017857"):
            writer.writerow({"scenario": "nominal", "controller": "deterministic_lp", "violation_rate": rate})
        for _ in range(5):
            writer.writerow({"scenario": "nominal", "controller": "dc3s_ftit", "violation_rate": "0.0"})

    summary_path = tmp_path / "reports" / "universal_orius_validation" / "domain_validation_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["domain", "baseline_tsvr_mean", "orius_tsvr_mean", "metric_surface"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "domain": "battery",
                "baseline_tsvr_mean": "0.7500",
                "orius_tsvr_mean": "0.8750",
                "metric_surface": "validation_harness",
            }
        )

    findings = []
    _check_reference_domain_validation_alignment(findings, tmp_path)

    assert any("battery baseline_tsvr_mean=0.75" in finding.detail for finding in findings)
    assert any("battery metric_surface='validation_harness'" in finding.detail for finding in findings)


def test_check_promoted_validation_surface_alignment_flags_cross_surface_drift(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "reports" / "publication" / "three_domain_ml_benchmark.csv"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["domain", "metric_surface", "baseline_tsvr_mean", "orius_tsvr_mean"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "domain": "Autonomous Vehicles",
                "metric_surface": "runtime_denominator",
                "baseline_tsvr_mean": "0.125000",
                "orius_tsvr_mean": "0.000000",
            }
        )
        writer.writerow(
            {
                "domain": "Medical and Healthcare Monitoring",
                "metric_surface": "runtime_denominator",
                "baseline_tsvr_mean": "0.291667",
                "orius_tsvr_mean": "0.000000",
            }
        )

    closure_path = tmp_path / "reports" / "publication" / "orius_domain_closure_matrix.csv"
    with closure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["domain", "baseline_tsvr", "orius_tsvr", "current_status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "domain": "Autonomous Vehicles",
                "baseline_tsvr": "0.1250",
                "orius_tsvr": "0.0000",
                "current_status": "tier=runtime_contract_closed; blocker=none",
            }
        )
        writer.writerow(
            {
                "domain": "Medical and Healthcare Monitoring",
                "baseline_tsvr": "0.2917",
                "orius_tsvr": "0.0000",
                "current_status": "tier=runtime_contract_closed; blocker=none",
            }
        )

    proxy_runtime_path = tmp_path / "reports" / "publication" / "three_domain_proxy_runtime_comparison.csv"
    with proxy_runtime_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["domain", "claim_governs_from", "proxy_metric_surface", "diagnostic_only"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "domain": "Autonomous Vehicles",
                "claim_governs_from": "validation_harness",
                "proxy_metric_surface": "validation_harness",
                "diagnostic_only": "False",
            }
        )
        writer.writerow(
            {
                "domain": "Medical and Healthcare Monitoring",
                "claim_governs_from": "validation_harness",
                "proxy_metric_surface": "validation_harness",
                "diagnostic_only": "False",
            }
        )

    av_summary_path = (
        tmp_path
        / "reports"
        / "orius_av"
        / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
        / "runtime_summary.csv"
    )
    av_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with av_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["controller", "tsvr"])
        writer.writeheader()
        writer.writerow({"controller": "baseline", "tsvr": "0.125000"})
        writer.writerow({"controller": "orius", "tsvr": "0.041700"})

    healthcare_summary_path = tmp_path / "reports" / "healthcare" / "runtime_summary.csv"
    healthcare_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with healthcare_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["controller", "tsvr"])
        writer.writeheader()
        writer.writerow({"controller": "baseline", "tsvr": "0.291667"})
        writer.writerow({"controller": "orius", "tsvr": "0.041700"})

    findings = []
    _check_promoted_validation_surface_alignment(findings, tmp_path)

    assert any("Autonomous Vehicles benchmark orius_tsvr_mean=0.0" in finding.detail for finding in findings)
    assert any(
        "Medical and Healthcare Monitoring closure orius_tsvr='0.0000'" in finding.detail
        for finding in findings
    )
    assert any(
        "current_status must disclose runtime-denominator governance" in finding.detail
        for finding in findings
    )
    assert any("claim_governs_from=runtime_denominator" in finding.detail for finding in findings)
    assert any("diagnostic_only=True" in finding.detail for finding in findings)


def test_check_healthcare_runtime_metric_alignment_flags_proxy_or_cross_surface_drift(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "reports" / "publication" / "three_domain_ml_benchmark.csv"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "domain",
                "certificate_valid_release_rate",
                "certificate_valid_release_rate_semantics",
                "intervention_rate",
                "fallback_activation_rate",
                "runtime_source",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "domain": "Medical and Healthcare Monitoring",
                "certificate_valid_release_rate": "1.000000",
                "certificate_valid_release_rate_semantics": "certos_required_payload_pass_rate_proxy_current_runtime_surface",
                "intervention_rate": "0.600000",
                "fallback_activation_rate": "0.000000",
                "runtime_source": "reports/healthcare/healthcare_runtime.duckdb",
            }
        )

    summary_path = tmp_path / "reports" / "healthcare" / "runtime_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "controller",
                "cva",
                "intervention_rate",
            ],
        )
        writer.writeheader()
        writer.writerow({"controller": "orius", "cva": "0.875000", "intervention_rate": "0.500000"})

    witness_summary_path = tmp_path / "reports" / "publication" / "domain_runtime_contract_summary.json"
    witness_summary_path.write_text(
        json.dumps({"domains": {"healthcare": {"certificate_valid_rate": 0.750000}}}),
        encoding="utf-8",
    )

    traces_path = tmp_path / "reports" / "healthcare" / "runtime_traces.csv"
    with traces_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["controller", "fallback_used"])
        writer.writeheader()
        writer.writerow({"controller": "orius", "fallback_used": "True"})
        writer.writerow({"controller": "orius", "fallback_used": "False"})

    findings = []
    _check_healthcare_runtime_metric_alignment(findings, tmp_path)

    assert any("certificate_valid_release_rate_semantics must be" in finding.detail for finding in findings)
    assert any("runtime_source must be" in finding.detail for finding in findings)
    assert any(
        "certificate_valid_release_rate=1.0 diverges from domain runtime witness certificate_valid_rate=0.75"
        in finding.detail
        for finding in findings
    )
    assert any(
        "intervention_rate=0.6 diverges from runtime summary intervention_rate=0.5" in finding.detail
        for finding in findings
    )
