"""Regression tests for the universal validation evidence gate.

Multi-Domain Universal Framework maturity model:
  reference         → battery
  proof_validated   → industrial, healthcare, vehicle
  shadow_synthetic  → navigation
  experimental      → aerospace
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from orius.orius_bench.controller_api import DC3SController, NominalController
from orius.orius_bench.metrics_engine import compute_all_metrics
from orius.adapters.vehicle import VehicleTrackAdapter


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_universal_orius_validation.py"


def _load_validation_script():
    spec = importlib.util.spec_from_file_location("run_universal_orius_validation", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation_script = _load_validation_script()


# ---------------------------------------------------------------------------
# Unit tests for the defended-domain evidence gate
# ---------------------------------------------------------------------------

def test_evaluate_proof_domain_rejects_trivial_baseline() -> None:
    summary = {
        "tsvr_nominal": [0.0, 0.01, 0.0],
        "tsvr_dc3s": [0.0, 0.0, 0.0],
    }
    report = validation_script._evaluate_proof_domain(summary)
    assert report["evidence_pass"] is False
    assert "baseline_gap_too_small" in report["failure_reasons"]


def test_evaluate_proof_domain_rejects_non_improving_orius() -> None:
    summary = {
        "tsvr_nominal": [0.18, 0.20, 0.22],
        "tsvr_dc3s": [0.24, 0.26, 0.25],
    }
    report = validation_script._evaluate_proof_domain(summary)
    assert report["evidence_pass"] is False
    assert "orius_did_not_improve" in report["failure_reasons"]


def test_evaluate_proof_domain_rejects_unstable_results() -> None:
    summary = {
        "tsvr_nominal": [0.08, 0.30, 0.02],
        "tsvr_dc3s": [0.01, 0.05, 0.01],
    }
    report = validation_script._evaluate_proof_domain(summary)
    assert report["evidence_pass"] is False
    assert "proof_domain_unstable" in report["failure_reasons"]


def test_evaluate_proof_domain_accepts_stable_improvement() -> None:
    summary = {
        "tsvr_nominal": [0.18, 0.20, 0.22],
        "tsvr_dc3s": [0.04, 0.05, 0.03],
    }
    report = validation_script._evaluate_proof_domain(summary)
    assert report["evidence_pass"] is True
    assert report["failure_reasons"] == []


# ---------------------------------------------------------------------------
# Unit tests for the portability_validated soft gate
# ---------------------------------------------------------------------------

def test_evaluate_portability_domain_passes_no_regression() -> None:
    summary = {
        "tsvr_nominal": [0.10, 0.12, 0.08],
        "tsvr_dc3s":    [0.09, 0.10, 0.07],
        "harness_status": "pass",
    }
    report = validation_script._evaluate_portability_domain("healthcare", summary)
    assert report["portability_pass"] is True
    assert report["failure_reasons"] == []


def test_evaluate_portability_domain_fails_on_regression() -> None:
    summary = {
        "tsvr_nominal": [0.10, 0.10, 0.10],
        "tsvr_dc3s":    [0.20, 0.22, 0.21],   # DC3S makes things worse
        "harness_status": "pass",
    }
    report = validation_script._evaluate_portability_domain("industrial", summary)
    assert report["portability_pass"] is False
    assert "dc3s_regression_on_tsvr" in report["failure_reasons"]


def test_evaluate_portability_domain_fails_on_harness_error() -> None:
    summary = {
        "tsvr_nominal": [0.10],
        "tsvr_dc3s":    [0.05],
        "harness_status": "fail",
    }
    report = validation_script._evaluate_portability_domain("aerospace", summary)
    assert report["portability_pass"] is False
    assert "harness_failed" in report["failure_reasons"]


def test_evaluate_portability_domain_passes_with_zero_tsvr() -> None:
    """If neither baseline nor DC3S has violations, no regression → pass."""
    summary = {
        "tsvr_nominal": [0.0, 0.0, 0.0],
        "tsvr_dc3s":    [0.0, 0.0, 0.0],
        "harness_status": "pass",
    }
    report = validation_script._evaluate_portability_domain("industrial", summary)
    assert report["portability_pass"] is True


# ---------------------------------------------------------------------------
# Integration: vehicle proof episode beats nominal
# ---------------------------------------------------------------------------

def test_vehicle_proof_episode_beats_nominal_on_locked_protocol() -> None:
    nominal = validation_script._run_episode(
        VehicleTrackAdapter(), NominalController(), seed=2000, horizon=24
    )
    repaired = validation_script._run_vehicle_proof_episode(
        DC3SController(), seed=2000, horizon=24
    )
    nominal_metrics  = compute_all_metrics(nominal)
    repaired_metrics = compute_all_metrics(repaired)
    assert nominal_metrics.tsvr > 0.0
    assert repaired_metrics.tsvr < nominal_metrics.tsvr


# ---------------------------------------------------------------------------
# End-to-end CLI test
# ---------------------------------------------------------------------------

def test_validation_cli_reports_all_domain_tiers(tmp_path: Path) -> None:
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--seeds", "1",
            "--horizon", "24",
            "--out", str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Harness pass:" in run.stdout
    assert "Evidence pass (defended):" in run.stdout
    assert "All defended domains pass: True" in run.stdout

    report       = json.loads((tmp_path / "validation_report.json").read_text())
    proof_report = json.loads((tmp_path / "proof_domain_report.json").read_text())
    port_report  = json.loads((tmp_path / "portability_validation_report.json").read_text())

    # ---- Reference and primary proof domain ----
    assert report["reference_domain"]     == "battery"
    assert report["proof_domain"]         == "vehicle"
    assert report["harness_pass"]         is True
    assert report["evidence_pass"]        is True
    assert report["all_proof_domains_pass"] is True
    assert report["defended_domains"]     == ["industrial", "healthcare", "vehicle"]
    assert report["proof_candidate_domains"] == []
    assert report["shadow_synthetic_domains"] == ["navigation"]
    assert report["bounded_universal_target_ready"] is False

    # ---- Canonical claim tiers ----
    domain_rows = {row["domain"]: row for row in report["domain_results"]}

    assert domain_rows["battery"]["validation_status"]     == "reference_validated"
    assert domain_rows["industrial"]["validation_status"] == "proof_validated"
    assert domain_rows["healthcare"]["validation_status"] == "proof_validated"
    assert domain_rows["vehicle"]["validation_status"]    == "proof_validated"
    assert domain_rows["navigation"]["validation_status"] == "shadow_synthetic"
    assert domain_rows["aerospace"]["validation_status"]  == "experimental"
    assert domain_rows["vehicle"]["closure_target_tier"] == "defended_bounded_row"
    assert domain_rows["navigation"]["closure_blocker"] == "navigation_real_data_row_missing"
    assert domain_rows["aerospace"]["closure_blocker"] == "real_multi_flight_safety_task_missing"

    # validated_domains contains battery + defended domains that passed
    for d in ("battery", "healthcare", "industrial", "vehicle"):
        assert d in report["validated_domains"], f"{d} not in validated_domains"
    assert "navigation" not in report["validated_domains"]
    assert "aerospace" not in report["validated_domains"]

    # ---- Strong-gate reports ----
    for proof_d in ("vehicle", "healthcare", "industrial"):
        dr = report["domain_proof_reports"][proof_d]
        if proof_d in {"industrial", "healthcare"}:
            assert dr["evidence_pass"] is True, (
                f"{proof_d} evidence gate failed: {dr.get('failure_reasons')}"
            )
        else:
            assert "evidence_pass" in dr, (
                f"{proof_d} candidate report missing evidence flag: {dr}"
            )

    # ---- Support-tier reports ----
    for support_d in ("navigation", "aerospace"):
        sr = report["domain_support_reports"][support_d]
        assert "portability_pass" in sr, (
            f"{support_d} support report missing portability flag: {sr}"
        )

    # ---- Proof-domain report (vehicle candidate) ----
    assert proof_report["proof_validated_domains"] == ["industrial", "healthcare", "vehicle"]
    assert proof_report["evaluated_proof_candidates"] == []
    assert proof_report["proof_domain"]   == "vehicle"

    # ---- Portability report ----
    assert port_report["shadow_synthetic_domains"] == ["navigation"]
    assert port_report["experimental_domains"] == ["aerospace"]
    assert port_report["portability_all_pass"] is True
