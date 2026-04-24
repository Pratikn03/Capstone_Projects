from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from orius.universal_framework.healthcare_adapter import HealthcareDomainAdapter
from orius.universal_theory.domain_runtime_contracts import (
    AV_BRAKE_HOLD_CONTRACT_ID,
    HEALTHCARE_FAIL_SAFE_CONTRACT_ID,
    witness_from_runtime_trace_row,
)
from orius.universal_theory.domain_validity import domain_certificate_validity_semantics


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_domain_runtime_contract_artifacts.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_domain_runtime_contract_artifacts", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_passing_av_row_links_t11_certificate_and_postcondition() -> None:
    witness = witness_from_runtime_trace_row(
        {
            "trace_id": "av-1",
            "controller": "orius",
            "scenario_id": "scenario-a",
            "t11_status": "runtime_linked",
            "t11_failed_obligations": "",
            "certificate_valid": "True",
            "true_constraint_violated": "False",
            "true_margin": "0.25",
        },
        domain="av",
    )

    assert witness.contract_id == AV_BRAKE_HOLD_CONTRACT_ID
    assert witness.passed is True
    assert witness.failure_reason == "none"
    assert witness.post_margin == 0.25


def test_failing_av_postcondition_fails_closed_even_with_certificate() -> None:
    witness = witness_from_runtime_trace_row(
        {
            "trace_id": "av-2",
            "controller": "orius",
            "scenario_id": "scenario-a",
            "t11_status": "runtime_linked",
            "certificate_valid": "True",
            "true_constraint_violated": "True",
            "true_margin": "-0.1",
        },
        domain="av",
    )

    assert witness.passed is False
    assert witness.t11_passed is True
    assert witness.certificate_valid is True
    assert witness.postcondition_passed is False
    assert "postcondition_failed" in witness.failure_reason


def test_passing_healthcare_row_links_t11_certificate_and_postcondition() -> None:
    witness = witness_from_runtime_trace_row(
        {
            "trace_id": "healthcare-orius-patient-1-0",
            "controller": "orius",
            "patient_id": "patient-1",
            "t11_status": "runtime_linked",
            "t11_failed_obligations": "[]",
            "certificate_valid": "1",
            "domain_postcondition_passed": "true",
            "true_margin": "2.0",
        },
        domain="healthcare",
    )

    assert witness.contract_id == HEALTHCARE_FAIL_SAFE_CONTRACT_ID
    assert witness.passed is True
    assert witness.failure_reason == "none"


def test_missing_t11_status_fails_closed() -> None:
    witness = witness_from_runtime_trace_row(
        {
            "trace_id": "healthcare-orius-patient-1-1",
            "controller": "orius",
            "patient_id": "patient-1",
            "certificate_valid": "True",
            "true_constraint_violated": "False",
        },
        domain="healthcare",
    )

    assert witness.passed is False
    assert witness.t11_status == "missing"
    assert "t11_not_runtime_linked" in witness.failure_reason


def test_av_fallback_certificate_validity_is_one_step_only() -> None:
    validity = domain_certificate_validity_semantics(
        domain="av",
        safe_action={"acceleration_mps2": -6.0},
        uncertainty={"meta": {"validity_status": "invalid"}},
        reliability_w=0.05,
        validity_status="invalid",
        step_index=7,
        repair_meta={"mode": "fallback", "fallback_required": True},
        cfg={"fallback_accel_mps2": -6.0},
    )

    assert validity.validity_horizon_H_t == 1
    assert validity.half_life_steps == 1
    assert validity.expires_at_step == 8
    assert validity.validity_scope == "single_step_fallback"
    assert validity.guarantee_checks_passed is True


def test_healthcare_fallback_certificate_validity_is_one_step_only() -> None:
    validity = domain_certificate_validity_semantics(
        domain="healthcare",
        safe_action={"alert_level": 1.0},
        uncertainty={"meta": {"validity_status": "degraded"}},
        reliability_w=0.1,
        validity_status="degraded",
        step_index=3,
        repair_meta={"mode": "fallback", "fallback_required": True},
        cfg={},
    )

    assert validity.validity_horizon_H_t == 1
    assert validity.half_life_steps == 1
    assert validity.expires_at_step == 4
    assert validity.validity_scope == "single_step_fallback"
    assert validity.guarantee_checks_passed is True


def test_non_fail_safe_fallback_certificate_fails_closed() -> None:
    validity = domain_certificate_validity_semantics(
        domain="healthcare",
        safe_action={"alert_level": 0.0},
        uncertainty={"meta": {"validity_status": "invalid"}},
        reliability_w=0.05,
        validity_status="invalid",
        step_index=0,
        repair_meta={"mode": "fallback", "fallback_required": True},
        cfg={},
    )

    assert validity.validity_horizon_H_t == 0
    assert validity.guarantee_checks_passed is False
    assert "fallback_action_not_fail_safe" in validity.guarantee_fail_reasons


def test_healthcare_nonpositive_validity_margin_routes_to_fallback() -> None:
    adapter = HealthcareDomainAdapter({"expected_cadence_s": 1.0})
    tightened = adapter.tighten_action_set(
        uncertainty={
            "spo2_lower_pct": 94.0,
            "spo2_upper_pct": 99.0,
            "forecast_spo2_lower_pct": 94.0,
            "hr_lower_bpm": 50.0,
            "hr_upper_bpm": 120.0,
            "hr_bpm": 130.0,
            "rr_lower": 12.0,
            "rr_upper": 20.0,
            "meta": {"validity_status": "watch", "w_t": 0.5},
        },
        constraints={"spo2_min_pct": 90.0, "hr_min_bpm": 40.0, "hr_max_bpm": 120.0, "rr_min": 8.0, "rr_max": 30.0},
        cfg={},
    )
    action, meta = adapter.repair_action(
        {"alert_level": 0.0},
        tightened,
        state={},
        uncertainty={},
        constraints={},
        cfg={},
    )

    assert tightened["validity_margin"] == 0.0
    assert tightened["fallback_required"] is True
    assert tightened["fallback_reason"] == "unsafe_current_vitals"
    assert action["alert_level"] == 1.0
    assert meta["mode"] == "fallback"


def test_generated_summary_only_reports_full_witness_pass_when_every_orius_row_passes(tmp_path: Path) -> None:
    builder = _load_builder()
    av_trace = tmp_path / "av_runtime_traces.csv"
    hc_trace = tmp_path / "healthcare_runtime_traces.csv"

    with av_trace.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trace_id",
                "controller",
                "scenario_id",
                "t11_status",
                "t11_failed_obligations",
                "certificate_valid",
                "true_constraint_violated",
                "true_margin",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trace_id": "av-pass",
                "controller": "orius",
                "scenario_id": "s1",
                "t11_status": "runtime_linked",
                "t11_failed_obligations": "",
                "certificate_valid": "True",
                "true_constraint_violated": "False",
                "true_margin": "1.0",
            }
        )

    with hc_trace.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trace_id",
                "controller",
                "patient_id",
                "t11_status",
                "t11_failed_obligations",
                "certificate_valid",
                "true_constraint_violated",
                "true_margin",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "trace_id": "hc-fail",
                "controller": "orius",
                "patient_id": "p1",
                "t11_status": "runtime_linked",
                "t11_failed_obligations": "",
                "certificate_valid": "False",
                "true_constraint_violated": "False",
                "true_margin": "1.0",
            }
        )

    report = builder.build_domain_runtime_contract_artifacts(
        av_trace=av_trace,
        healthcare_trace=hc_trace,
        out_dir=tmp_path / "publication",
        normalize_traces=True,
        recover_t11_from_certificates=False,
    )
    summary = json.loads(Path(report["domain_runtime_contract_summary_json"]).read_text(encoding="utf-8"))

    assert summary["domains"]["av"]["witness_pass_rate"] == 1.0
    assert summary["domains"]["healthcare"]["witness_pass_rate"] == 0.0
    assert summary["domains"]["healthcare"]["certificate_valid_rate"] == 0.0
    with hc_trace.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["contract_id"] == HEALTHCARE_FAIL_SAFE_CONTRACT_ID
    assert row["domain_postcondition_failure"] == "certificate_invalid"
