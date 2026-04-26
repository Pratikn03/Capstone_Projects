"""Regression tests for the three-domain universal validation evidence gate."""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from orius.adapters.vehicle import VehicleTrackAdapter
from orius.orius_bench.controller_api import DC3SController, NominalController
from orius.orius_bench.metrics_engine import compute_all_metrics
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


def test_evaluate_proof_domain_rejects_trivial_baseline() -> None:
    summary = {"tsvr_nominal": [0.0, 0.01, 0.0], "tsvr_dc3s": [0.0, 0.0, 0.0]}
    report = validation_script._evaluate_proof_domain(summary)
    assert report["evidence_pass"] is False
    assert "baseline_gap_too_small" in report["failure_reasons"]


def test_evaluate_proof_domain_rejects_non_improving_orius() -> None:
    summary = {"tsvr_nominal": [0.18, 0.20, 0.22], "tsvr_dc3s": [0.24, 0.26, 0.25]}
    report = validation_script._evaluate_proof_domain(summary)
    assert report["evidence_pass"] is False
    assert "orius_did_not_improve" in report["failure_reasons"]


def test_evaluate_proof_domain_rejects_unstable_results() -> None:
    summary = {"tsvr_nominal": [0.08, 0.30, 0.02], "tsvr_dc3s": [0.01, 0.05, 0.01]}
    report = validation_script._evaluate_proof_domain(summary)
    assert report["evidence_pass"] is False
    assert "proof_domain_unstable" in report["failure_reasons"]


def test_evaluate_proof_domain_accepts_stable_improvement() -> None:
    summary = {"tsvr_nominal": [0.18, 0.20, 0.22], "tsvr_dc3s": [0.04, 0.05, 0.03]}
    report = validation_script._evaluate_proof_domain(summary)
    assert report["evidence_pass"] is True
    assert report["failure_reasons"] == []


def test_evaluate_portability_domain_passes_no_regression() -> None:
    summary = {"tsvr_nominal": [0.10, 0.12, 0.08], "tsvr_dc3s": [0.09, 0.10, 0.07], "harness_status": "pass"}
    report = validation_script._evaluate_portability_domain("healthcare", summary)
    assert report["portability_pass"] is True
    assert report["failure_reasons"] == []


def test_evaluate_portability_domain_fails_on_regression() -> None:
    summary = {"tsvr_nominal": [0.10, 0.10, 0.10], "tsvr_dc3s": [0.20, 0.22, 0.21], "harness_status": "pass"}
    report = validation_script._evaluate_portability_domain("healthcare", summary)
    assert report["portability_pass"] is False
    assert "dc3s_regression_on_tsvr" in report["failure_reasons"]


def test_evaluate_portability_domain_fails_on_harness_error() -> None:
    summary = {"tsvr_nominal": [0.10], "tsvr_dc3s": [0.05], "harness_status": "fail"}
    report = validation_script._evaluate_portability_domain("vehicle", summary)
    assert report["portability_pass"] is False
    assert "harness_failed" in report["failure_reasons"]


def test_vehicle_proof_episode_beats_nominal_on_locked_protocol() -> None:
    nominal = validation_script._run_episode(VehicleTrackAdapter(), NominalController(), seed=2000, horizon=24)
    repaired = validation_script._run_vehicle_proof_episode(DC3SController(), seed=2000, horizon=24)
    nominal_metrics = compute_all_metrics(nominal)
    repaired_metrics = compute_all_metrics(repaired)
    assert nominal_metrics.tsvr > 0.0
    assert repaired_metrics.tsvr < nominal_metrics.tsvr


def test_build_tracks_requires_missing_surface_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    av_path = tmp_path / "av.csv"
    healthcare_path = tmp_path / "missing_healthcare.csv"
    _write_csv(
        av_path,
        ["vehicle_id", "step", "position_m", "speed_mps", "speed_limit_mps", "lead_position_m", "ts_utc", "source_split"],
        ["veh-1", 0, 10.0, 8.0, 13.4, 35.0, "2026-01-01T00:00:00Z", "train"],
    )

    monkeypatch.setattr(real_data_loader, "AV_PATH", av_path)
    monkeypatch.setattr(real_data_loader, "HEALTHCARE_RUNTIME_PATH", healthcare_path)

    tracks, domain_sources, missing = validation_script._build_tracks()

    assert tracks == []
    assert domain_sources["healthcare"] == str(healthcare_path)
    assert f"healthcare={healthcare_path}" in missing


def test_validation_cli_reports_only_three_domains(tmp_path: Path) -> None:
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--seeds",
            "1",
            "--horizon",
            "24",
            "--out",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode in {0, 1}

    report = json.loads((tmp_path / "validation_report.json").read_text())
    proof_report = json.loads((tmp_path / "proof_domain_report.json").read_text())
    port_report = json.loads((tmp_path / "portability_validation_report.json").read_text())
    domain_rows = {
        row["domain"]: row
        for row in csv.DictReader((tmp_path / "domain_validation_summary.csv").open())
    }
    per_controller_rows = list(csv.DictReader((tmp_path / "per_controller_tsvr.csv").open()))
    diagnostic_rows = list(csv.DictReader((tmp_path / "diagnostic_validation_harness_tsvr.csv").open()))

    assert report["reference_domain"] == "battery"
    assert report["proof_domain"] == "vehicle"
    assert report["defended_domains"] == ["healthcare", "vehicle"]
    assert report["shadow_synthetic_domains"] == []
    assert report["experimental_domains"] == []
    assert set(report["domain_results"]) == {"battery", "healthcare", "vehicle"}
    assert set(domain_rows) == {"battery", "healthcare", "vehicle"}

    assert proof_report["proof_validated_domains"] == ["healthcare", "vehicle"]
    assert port_report["shadow_synthetic_domains"] == []
    assert port_report["experimental_domains"] == []
    for domain in ("healthcare", "vehicle"):
        row = domain_rows[domain]
        assert row["metric_surface"] == "runtime_denominator"
        assert row["runtime_source"]
        assert int(row["n_steps"]) > 0
        assert row["strict_runtime_gate"] == "True"
        assert row["certificate_valid_rate"] == "1.000000"
        assert row["t11_pass_rate"] == "1.000000"
        assert row["postcondition_pass_rate"] == "1.000000"
        assert row["runtime_witness_pass_rate"] == "1.000000"

    promoted_rows = [
        row
        for row in per_controller_rows
        if row["domain"] in {"healthcare", "vehicle"}
    ]
    assert promoted_rows
    assert {row["seed"] for row in promoted_rows} == {"runtime_denominator"}
    assert all(row["tsvr"] != "" and row["oasg"] != "" and row["cva"] != "" for row in promoted_rows)

    assert diagnostic_rows
    assert {row["metric_surface"] for row in diagnostic_rows} == {"validation_harness"}
    assert {row["diagnostic_only"] for row in diagnostic_rows} == {"True"}
    assert {row["claim_governs_from"] for row in diagnostic_rows} == {"runtime_denominator"}
