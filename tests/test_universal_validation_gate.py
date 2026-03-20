"""Regression tests for the universal validation evidence gate.

Multi-Domain Universal Framework maturity model:
  reference     → battery   (locked PhD-thesis reference domain)
  proof_domain  → vehicle, healthcare, industrial, aerospace, navigation
                  (full evidence gate: TSVR reduction ≥ 25 %)
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
# Unit tests for the proof-domain evidence gate
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
    assert "Evidence pass" in run.stdout
    assert "All proof domains pass:  True" in run.stdout

    report       = json.loads((tmp_path / "validation_report.json").read_text())
    proof_report = json.loads((tmp_path / "proof_domain_report.json").read_text())
    port_report  = json.loads((tmp_path / "portability_validation_report.json").read_text())

    # ---- Reference and primary proof domain ----
    assert report["reference_domain"]     == "battery"
    assert report["proof_domain"]         == "vehicle"
    assert report["harness_pass"]         is True
    assert report["evidence_pass"]        is True
    assert report["all_proof_domains_pass"] is True

    # ---- All non-battery domains are now proof_validated ----
    domain_rows = {row["domain"]: row for row in report["domain_results"]}

    assert domain_rows["battery"]["validation_status"]     == "reference_validated"
    for proof_d in ("vehicle", "healthcare", "industrial", "aerospace", "navigation"):
        assert domain_rows[proof_d]["validation_status"]   == "proof_validated", (
            f"{proof_d} expected proof_validated, got "
            f"{domain_rows[proof_d]['validation_status']}: "
            f"{report['domain_proof_reports'].get(proof_d, {}).get('failure_reasons')}"
        )

    # validated_domains contains battery + all proof domains that passed
    for d in ("battery", "vehicle", "healthcare", "industrial", "aerospace", "navigation"):
        assert d in report["validated_domains"], f"{d} not in validated_domains"

    # ---- Domain proof reports ----
    for proof_d in ("vehicle", "healthcare", "industrial", "aerospace", "navigation"):
        dr = report["domain_proof_reports"][proof_d]
        assert dr["evidence_pass"] is True, (
            f"{proof_d} evidence gate failed: {dr.get('failure_reasons')}"
        )

    # ---- Proof-domain report (vehicle primary) ----
    assert proof_report["evidence_pass"]  is True
    assert proof_report["proof_domain"]   == "vehicle"

    # ---- Portability report (backward-compat; now empty) ----
    assert port_report["portability_all_pass"] is True
