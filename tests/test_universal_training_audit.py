"""Regression tests for the canonical-domain training audit."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_universal_training_audit.py"


def test_universal_training_audit_reports_verified_domains(tmp_path: Path) -> None:
    assert not (REPO_ROOT / "artifacts" / "uncertainty" / "gbm_price_eur_mwh_conformal.json").exists()
    assert not (REPO_ROOT / "artifacts" / "backtests" / "price_eur_mwh_calibration.npz").exists()
    assert not (REPO_ROOT / "artifacts" / "backtests" / "price_eur_mwh_test.npz").exists()

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--out",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert "Universal Training Audit" in run.stdout
    assert run.returncode == 0

    report = json.loads((tmp_path / "training_audit_report.json").read_text())
    assert report["domains"] == ["battery", "av", "healthcare"]
    assert report["all_passed"] is True
    assert report["failed_domains"] == []
    assert "battery" in report["training_verified_domains"]
    assert "real_data_backed_domains" in report
    assert "training_surface_closed_domains" in report
    assert "battery" in report["training_surface_closed_domains"]

    summary_rows = {
        row["domain"]: row
        for row in csv.DictReader((tmp_path / "domain_training_summary.csv").open())
    }
    assert summary_rows["battery"]["training_verified"] == "True"
    assert summary_rows["battery"]["training_surface_closed"] == "True"
    assert summary_rows["battery"]["note"] == "verified"
    assert "healthcare" in summary_rows
    assert float(summary_rows["healthcare"]["picp_90"]) >= 0.90
    assert "healthcare_calibration_repaired" in summary_rows["healthcare"]["note"]
    assert (tmp_path / "healthcare_calibration_repair.json").exists()
    assert "av" in summary_rows

    battery_log = (tmp_path / "logs" / "battery_verify.log").read_text(encoding="utf-8")
    assert "✓ All checks PASSED" in battery_log
    assert "Missing GBM conformal JSON for target 'price_eur_mwh'" not in battery_log
