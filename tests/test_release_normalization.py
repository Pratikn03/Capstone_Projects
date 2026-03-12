from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import scripts.build_publication_artifact as pub
import scripts.run_r1_release as r1
import scripts.sync_paper_assets as sync
import scripts.train_dataset as td


def test_write_run_context_emits_normalized_run_manifest(tmp_path: Path) -> None:
    layout = td.RunLayout(
        mode="candidate",
        run_id="R1_TEST_diag",
        dataset="DE",
        artifacts_root=tmp_path / "artifacts",
        models_dir=tmp_path / "artifacts" / "models",
        uncertainty_dir=tmp_path / "artifacts" / "uncertainty",
        backtests_dir=tmp_path / "artifacts" / "backtests",
        registry_dir=tmp_path / "artifacts" / "registry",
        reports_dir=tmp_path / "reports",
        publication_dir=tmp_path / "reports" / "publication",
        validation_report=tmp_path / "reports" / "validation.md",
        data_manifest_output=tmp_path / "artifacts" / "registry" / "data_manifest.json",
        walk_forward_report=tmp_path / "reports" / "walk_forward.json",
        selection_output_dir=tmp_path / "artifacts" / "registry",
    )
    cfg = td.DATASET_REGISTRY["DE"]
    preflight = {
        "expected_targets": ["load_mw", "wind_mw", "solar_mw"],
        "expected_model_types": ["gbm_lightgbm"],
    }

    td._write_run_context(cfg=cfg, run_layout=layout, preflight=preflight, profile="standard")

    manifest = json.loads((layout.registry_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_id"] == "R1_TEST"
    assert manifest["run_id"] == "R1_TEST_diag"
    assert manifest["dataset"] == "DE"
    assert manifest["mode"] == "candidate"
    assert manifest["config_path"] == cfg.config_file
    assert manifest["config_hash"]
    assert manifest["feature_manifest_path"] == str(layout.data_manifest_output)
    assert manifest["preflight_path"] == str(layout.reports_dir / "preflight_dataset_analysis.json")
    assert manifest["targets"] == preflight["expected_targets"]
    assert manifest["selection_summary_path"].endswith("tuning_summary_de.json")
    assert manifest["promoted_at"] is None


def test_load_release_source_runs_reads_single_release_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pub, "REPO_ROOT", tmp_path)
    for dataset in pub.RELEASE_DATASETS:
        manifest_path = tmp_path / "artifacts" / "runs" / dataset.lower() / "TEST123" / "registry" / "run_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "release_id": "TEST123",
                    "run_id": "TEST123",
                    "dataset": dataset,
                    "accepted": True,
                    "artifacts": {
                        "reports_dir": f"reports/runs/{dataset.lower()}/TEST123",
                        "models_dir": f"artifacts/runs/{dataset.lower()}/TEST123/models",
                        "uncertainty_dir": f"artifacts/runs/{dataset.lower()}/TEST123/uncertainty",
                        "backtests_dir": f"artifacts/runs/{dataset.lower()}/TEST123/backtests",
                    },
                    "selection_summary_path": f"artifacts/runs/{dataset.lower()}/TEST123/registry/tuning_summary.json",
                }
            ),
            encoding="utf-8",
        )

    source_runs = pub._load_release_source_runs("TEST123")
    assert set(source_runs) == set(pub.RELEASE_DATASETS)
    assert source_runs["DE"]["release_id"] == "TEST123"
    assert source_runs["DE"]["accepted"] is True
    assert source_runs["DE"]["manifest_path"].endswith("run_manifest.json")


def test_run_r1_verify_main_returns_nonzero_on_failed_acceptance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        r1,
        "stage_verify",
        lambda release_id: {"DE": {"passed": False}, "US_MISO": {"passed": True}},
    )
    monkeypatch.setattr(sys, "argv", ["run_r1_release.py", "--stage", "verify", "--release-id", "TEST123"])
    assert r1.main() == 1


def test_check_release_manifest_contract_accepts_single_release_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sync, "REPO_ROOT", tmp_path)
    source_artifact = tmp_path / "reports" / "publication" / "dc3s_main_table.csv"
    source_artifact.parent.mkdir(parents=True, exist_ok=True)
    source_artifact.write_text("controller\n", encoding="utf-8")
    release_manifest = {
        "release_id": "TEST123",
        "source_runs": {
            "DE": {"release_id": "TEST123"},
            "US_MISO": {"release_id": "TEST123"},
        },
        "paper_assets": {
            "TBL01_MAIN_RESULTS": {
                "paper_path": "paper/assets/tables/tbl01_main_results.csv",
                "source_artifact": "reports/publication/dc3s_main_table.csv",
                "build_command": "bash scripts/export_paper_assets.sh",
            }
        },
    }
    manifest_path = tmp_path / "reports" / "publication" / "release_manifest.json"
    manifest_path.write_text(json.dumps(release_manifest), encoding="utf-8")

    assert sync._check_release_manifest_contract() == []
