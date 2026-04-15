"""Regression tests for the universal validation evidence gate.

Strict mode now requires defended runtime surfaces for all promoted
non-battery domains. Legacy support-tier behavior remains available only
behind ``--allow-support-tier``.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from orius.orius_bench.controller_api import DC3SController, NominalController
from orius.orius_bench.metrics_engine import compute_all_metrics
from orius.adapters.vehicle import VehicleTrackAdapter
from orius.orius_bench import real_data_loader


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


def _write_csv(path: Path, header: list[str], row: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ",".join(header) + "\n" + ",".join(str(item) for item in row) + "\n",
        encoding="utf-8",
    )


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

def test_build_tracks_requires_missing_defended_surface_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    av_path = tmp_path / "av.csv"
    industrial_path = tmp_path / "industrial.csv"
    healthcare_path = tmp_path / "healthcare.csv"
    navigation_path = tmp_path / "missing_navigation.csv"
    aerospace_path = tmp_path / "aerospace.csv"

    _write_csv(
        av_path,
        ["vehicle_id", "step", "position_m", "speed_mps", "speed_limit_mps", "lead_position_m", "ts_utc", "source_split"],
        ["veh-1", 0, 10.0, 8.0, 13.4, 35.0, "2026-01-01T00:00:00Z", "train"],
    )
    _write_csv(
        industrial_path,
        ["sensor_id", "step", "temp_c", "vacuum_cmhg", "pressure_mbar", "humidity_pct", "power_mw", "ts_utc"],
        ["sensor-1", 0, 20.0, 40.0, 1010.0, 50.0, 450.0, "2026-01-01T00:00:00Z"],
    )
    _write_csv(
        healthcare_path,
        ["patient_id", "step", "hr_bpm", "spo2_pct", "respiratory_rate", "ts_utc"],
        ["patient-1", 0, 72.0, 97.0, 14.0, "2026-01-01T00:00:00Z"],
    )
    _write_csv(
        aerospace_path,
        ["flight_id", "step", "altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct", "ts_utc"],
        ["flight-1", 0, 3000.0, 180.0, 5.0, 80.0, "2026-01-01T00:00:00Z"],
    )

    monkeypatch.setattr(real_data_loader, "AV_PATH", av_path)
    monkeypatch.setattr(real_data_loader, "INDUSTRIAL_RUNTIME_PATH", industrial_path)
    monkeypatch.setattr(real_data_loader, "HEALTHCARE_RUNTIME_PATH", healthcare_path)
    monkeypatch.setattr(real_data_loader, "NAVIGATION_PATH", navigation_path)
    monkeypatch.setattr(real_data_loader, "AEROSPACE_RUNTIME_PATH", aerospace_path)
    monkeypatch.setattr(real_data_loader, "AEROSPACE_REALFLIGHT_PATH", tmp_path / "missing_realflight.csv")

    tracks, domain_sources, missing = validation_script._build_tracks(allow_support_tier=False)

    assert tracks == []
    assert domain_sources["navigation"] == str(navigation_path)
    assert f"navigation={navigation_path}" in missing


def test_validation_cli_requires_support_tier_when_navigation_surface_missing(tmp_path: Path) -> None:
    navigation_path = REPO_ROOT / "data" / "navigation" / "processed" / "navigation_orius.csv"
    aerospace_path = REPO_ROOT / "data" / "aerospace" / "processed" / "aerospace_realflight_runtime.csv"
    if navigation_path.exists() and aerospace_path.exists():
        pytest.skip("Repository already has both strict lower-tier runtime surfaces staged.")

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--seeds", "1",
            "--horizon", "24",
            "--out", str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1
    assert "Strict defended validation requires staged canonical runtime surfaces" in run.stdout
    if not navigation_path.exists():
        assert "navigation=" in run.stdout
    if not aerospace_path.exists():
        assert "aerospace=" in run.stdout
    assert "--allow-support-tier" in run.stdout


def test_validation_cli_reports_all_domain_tiers_under_explicit_support_tier(tmp_path: Path) -> None:
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--seeds", "1",
            "--horizon", "24",
            "--allow-support-tier",
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
    assert domain_rows["navigation"]["closure_blocker"] == "navigation_kitti_runtime_missing"
    assert domain_rows["aerospace"]["closure_blocker"] == "aerospace_realflight_runtime_missing"
    assert domain_rows["battery"]["metric_surface"] == "locked_publication_nominal"
    assert float(domain_rows["battery"]["baseline_tsvr_mean"]) == pytest.approx(0.0393, abs=1e-4)
    assert float(domain_rows["battery"]["orius_tsvr_mean"]) == pytest.approx(0.0, abs=1e-9)
    assert float(domain_rows["battery"]["orius_reduction_pct"]) == pytest.approx(100.0, abs=1e-3)
    assert report["reference_domain_metric_surface"] == "locked_publication_nominal"
    assert report["reference_domain_metrics"]["baseline_tsvr_mean"] == pytest.approx(0.0392856, abs=1e-7)
    assert report["reference_domain_metrics"]["orius_tsvr_mean"] == pytest.approx(0.0, abs=1e-9)

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

    oasg_rows = {
        row["domain"]: row
        for row in csv.DictReader((tmp_path / "cross_domain_oasg_table.csv").open())
    }
    per_controller_rows = list(csv.DictReader((tmp_path / "per_controller_tsvr.csv").open()))
    vehicle_nominal_oasg = [
        float(row["oasg"])
        for row in per_controller_rows
        if row["domain"] == "vehicle" and row["controller"] == "nominal"
    ]
    vehicle_dc3s_oasg = [
        float(row["oasg"])
        for row in per_controller_rows
        if row["domain"] == "vehicle" and row["controller"] == "dc3s"
    ]
    assert float(oasg_rows["vehicle"]["oasg_rate_baseline"]) == pytest.approx(
        sum(vehicle_nominal_oasg) / len(vehicle_nominal_oasg),
        abs=1e-4,
    )
    assert float(oasg_rows["vehicle"]["oasg_rate_orius"]) == pytest.approx(
        sum(vehicle_dc3s_oasg) / len(vehicle_dc3s_oasg),
        abs=1e-4,
    )
