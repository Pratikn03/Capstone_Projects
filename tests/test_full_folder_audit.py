from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import scripts.full_folder_audit as audit


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def test_full_folder_audit_writes_outputs_and_detects_key_findings(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "audit@example.com")
    _git(repo, "config", "user.name", "Audit Bot")

    (repo / "scripts").mkdir()
    (repo / "paper").mkdir()
    (repo / "reports" / "publication").mkdir(parents=True)
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / "frontend" / "node_modules" / ".bin").mkdir(parents=True)

    (repo / "Makefile").write_text(
        "camera-ready-freeze:\n\tpython scripts/run.py --external-root /Users/pratik_n/orius_external_data\n",
        encoding="utf-8",
    )
    (repo / "paper" / "paper.aux").write_text("aux output", encoding="utf-8")
    (repo / "reports" / "real_data_preflight.json").write_text(
        json.dumps(
            {
                "all_domains_present": False,
                "disk": {"free_gib": 5.0, "min_free_gib": 250.0, "passes_threshold": False},
            }
        ),
        encoding="utf-8",
    )
    (repo / "reports" / "publication" / "orius_camera_ready_package_manifest.json").write_text(
        json.dumps({"status": "failed", "failure_step": "real_data_preflight"}),
        encoding="utf-8",
    )
    (repo / "reports" / "publication" / "orius_submission_scorecard.csv").write_text(
        "target_tier,readiness_score_100,critical_gap_count,high_gap_count,meets_93_gate\n"
        "equal_domain_93,78.8,2,1,False\n",
        encoding="utf-8",
    )
    (repo / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / "frontend" / "node_modules" / ".bin" / "next").write_text("#!/bin/sh\n", encoding="utf-8")

    _git(repo, "add", "Makefile", "paper/paper.aux", "reports/publication/orius_submission_scorecard.csv")
    _git(repo, "commit", "-m", "seed")

    monkeypatch.setattr(audit, "REPO_ROOT", repo)
    out_dir = repo / "reports" / "audit"

    summary = audit.run_full_folder_audit(out_dir=out_dir)

    assert summary["files_total"] >= 7
    assert (out_dir / "full_folder_findings_catalog.csv").exists()
    assert (out_dir / "full_folder_coverage_ledger.csv").exists()
    assert (out_dir / "full_folder_synthesis.md").exists()

    with (out_dir / "full_folder_findings_catalog.csv").open(encoding="utf-8", newline="") as handle:
        findings = list(csv.DictReader(handle))
    categories = {row["category"] for row in findings}
    assert "hardcoded_local_path" in categories
    assert "tracked_latex_build_outputs" in categories
    assert "real_data_preflight_failed" in categories
    assert "camera_ready_freeze_failed" in categories
    assert "equal_domain_gate_blocked" in categories


def test_classify_vendor_and_text_modes(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / "frontend" / "node_modules").mkdir(parents=True)
    (repo / "reports").mkdir()

    monkeypatch.setattr(audit, "REPO_ROOT", repo)

    venv_file = repo / ".venv" / "bin" / "python"
    venv_file.write_text("", encoding="utf-8")
    vendor_file = repo / "frontend" / "node_modules" / "pkg.js"
    vendor_file.write_text("module.exports = 1;\n", encoding="utf-8")
    report_file = repo / "reports" / "claim.csv"
    report_file.write_text("a,b\n1,2\n", encoding="utf-8")

    assert audit._classify_file(venv_file, tracked=False) == ("virtualenv", "metadata")
    assert audit._classify_file(vendor_file, tracked=False) == ("vendor", "metadata")
    assert audit._classify_file(report_file, tracked=True) == ("generated_text", "semantic")
