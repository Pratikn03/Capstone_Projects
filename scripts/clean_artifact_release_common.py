"""Shared constants for the ORIUS clean artifact release bundle."""
from __future__ import annotations

from pathlib import Path


THREE_DOMAIN_CLAIM_BOUNDARY = (
    "Promoted ORIUS evidence is limited to Battery Energy Storage, "
    "nuPlan Autonomous Vehicles replay/surrogate runtime evidence, and "
    "Medical and Healthcare Monitoring retrospective/source-holdout evidence. "
    "The release does not defend additional public domains, CARLA completion, "
    "road deployment, prospective clinical-trial evidence, or live clinical deployment."
)

REPRODUCTION_COMMANDS = [
    ".venv/bin/python scripts/validate_nuplan_freeze_gate.py --manifest-out reports/predeployment_external_validation/nuplan_full_av_gate.json",
    ".venv/bin/python scripts/run_predeployment_external_validation.py --out reports/predeployment_external_validation",
    ".venv/bin/python scripts/validate_metric_consistency.py",
    ".venv/bin/python scripts/validate_paper_claims.py",
    ".venv/bin/python scripts/validate_assumption_consistency.py",
    ".venv/bin/python scripts/validate_theorem_surface.py",
    ".venv/bin/python scripts/validate_top_venue_research_package.py",
    ".venv/bin/python scripts/validate_equal_domain_artifact_discipline.py",
    ".venv/bin/python scripts/build_camera_ready_figure_lineage.py --verify",
    (
        ".venv/bin/pytest -q tests/test_nuplan_av_surface.py "
        "tests/test_battery_av_pipeline.py tests/test_av_waymo_dry_run.py "
        "tests/test_equal_domain_artifact_discipline.py tests/test_three_domain_submission_lane.py "
        "tests/test_three_domain_offline_freeze.py tests/test_predeployment_external_validation.py "
        "tests/test_validate_paper_claims.py tests/test_active_theorem_audit.py "
        "tests/test_camera_ready_figure_lineage.py"
    ),
]

CODE_SCRIPTS = [
    "scripts/build_camera_ready_figure_lineage.py",
    "scripts/build_clean_artifact_release.py",
    "scripts/build_nuplan_av_surface.py",
    "scripts/build_waymo_av_dry_run_report.py",
    "scripts/clean_artifact_release_common.py",
    "scripts/run_battery_av_pipeline.py",
    "scripts/run_predeployment_external_validation.py",
    "scripts/validate_assumption_consistency.py",
    "scripts/validate_clean_artifact_release.py",
    "scripts/validate_equal_domain_artifact_discipline.py",
    "scripts/validate_metric_consistency.py",
    "scripts/validate_nuplan_freeze_gate.py",
    "scripts/validate_paper_claims.py",
    "scripts/validate_theorem_surface.py",
    "scripts/validate_top_venue_research_package.py",
]

CODE_TESTS = [
    "tests/test_active_theorem_audit.py",
    "tests/test_av_waymo_dry_run.py",
    "tests/test_battery_av_pipeline.py",
    "tests/test_camera_ready_figure_lineage.py",
    "tests/test_clean_artifact_release.py",
    "tests/test_equal_domain_artifact_discipline.py",
    "tests/test_nuplan_av_surface.py",
    "tests/test_predeployment_external_validation.py",
    "tests/test_three_domain_offline_freeze.py",
    "tests/test_three_domain_submission_lane.py",
    "tests/test_validate_paper_claims.py",
]

ENVIRONMENT_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "requirements.lock.txt",
]

CONFIG_FILES = [
    "configs/av_datasets.yaml",
    "configs/dc3s_healthcare.yaml",
    "configs/nuplan_runtime_dropout_aligned_m15.json",
    "configs/train_forecast_av.yaml",
    "configs/train_forecast_healthcare.yaml",
    "configs/validation/healthcare_heldout_runtime.yml",
    "configs/validation/nuplan_carla_predeployment.yml",
]

MANIFEST_FILES = [
    "data/healthcare/mimic3/processed/mimic3_manifest.json",
    "data/healthcare/processed/manifest.json",
    "data/orius_av/av/processed_nuplan_allzip_grouped/feature_table_report.json",
    "data/orius_av/av/processed_nuplan_allzip_grouped/nuplan_db_inventory.csv",
    "data/orius_av/av/processed_nuplan_allzip_grouped/nuplan_source_manifest.json",
    "data/orius_av/av/processed_nuplan_allzip_grouped/nuplan_surface_report.json",
    "reports/battery_av_healthcare/overall/domain_summary.csv",
    "reports/battery_av_healthcare/overall/release_summary.json",
    "reports/predeployment_external_validation/external_validation_report.json",
    "reports/predeployment_external_validation/external_validation_summary.csv",
    "reports/predeployment_external_validation/healthcare_retrospective_holdout_split_details.csv",
    "reports/predeployment_external_validation/healthcare_site_splits/manifest.json",
    "reports/predeployment_external_validation/nuplan_closed_loop_manifest.json",
    "reports/predeployment_external_validation/nuplan_closed_loop_summary.csv",
    "reports/predeployment_external_validation/nuplan_full_av_gate.json",
    "reports/publication/camera_ready_figure_lineage.csv",
    "reports/publication/camera_ready_figure_lineage.json",
    "reports/publication/domain_runtime_contract_summary.json",
    "reports/publication/domain_runtime_contract_witnesses.csv",
    "reports/publication/equal_domain_artifact_discipline.csv",
    "reports/publication/orius_domain_closure_matrix.csv",
    "reports/publication/three_domain_ml_benchmark.csv",
]

TABLE_DIRS = [
    "paper/assets/tables/generated",
]

TABLE_FILES = [
    "reports/publication/three_domain_ablation_matrix.csv",
    "reports/publication/three_domain_baseline_suite.csv",
    "reports/publication/three_domain_grouped_coverage.csv",
    "reports/publication/three_domain_grouped_width.csv",
    "reports/publication/three_domain_ml_benchmark.csv",
    "reports/publication/three_domain_negative_controls.csv",
    "reports/publication/three_domain_reliability_calibration.csv",
    "reports/publication/three_domain_runtime_safety_tradeoff.csv",
    "reports/publication/what_orius_is_not_matrix.csv",
]

BATTERY_EVIDENCE_DIRS = [
    "reports/battery_av/battery",
    "reports/hil",
]

AV_EVIDENCE_DIRS = [
    "reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest",
    "artifacts/models_orius_av_nuplan_allzip_grouped",
    "artifacts/uncertainty/orius_av_nuplan_allzip_grouped",
]

HEALTHCARE_EVIDENCE_FILES = [
    "reports/healthcare/certos_verification_summary.json",
    "reports/healthcare/runtime_ablation_summary.csv",
    "reports/healthcare/runtime_comparator_summary.csv",
    "reports/healthcare/runtime_governance_summary.csv",
    "reports/healthcare/runtime_summary.csv",
    "reports/predeployment_external_validation/healthcare_heldout_runtime_preparation_manifest.json",
    "reports/predeployment_external_validation/healthcare_retrospective_holdout_split_details.csv",
    "reports/predeployment_external_validation/healthcare_site_splits/manifest.json",
]

MANUSCRIPT_FILES = [
    "paper.pdf",
    "paper/paper.pdf",
    "paper/paper.docx",
    "paper/ieee/orius_ieee_main.pdf",
    "paper/ieee/orius_ieee_main.docx",
]

REQUIRED_RELEASE_PATHS = [
    "README.md",
    "REPRODUCE.md",
    "manifest.json",
    "MANIFEST.sha256",
    "environment/pyproject.toml",
    "environment/requirements.txt",
    "code/scripts/validate_nuplan_freeze_gate.py",
    "code/scripts/run_predeployment_external_validation.py",
    "code/tests/test_clean_artifact_release.py",
    "configs/nuplan_runtime_dropout_aligned_m15.json",
    "manifests/av_nuplan/nuplan_source_manifest.json",
    "manifests/predeployment/nuplan_full_av_gate.json",
    "tables/generated/tbl_real_data_validation.tex",
    "evidence/battery/battery/runtime_summary.csv",
    "evidence/battery/hil/hil_summary.json",
    "evidence/av_nuplan/runtime/runtime_traces.csv",
    "evidence/av_nuplan/runtime/dc3s_av_waymo_dryrun.duckdb",
    "evidence/av_nuplan/models/nuplan_av_ego_speed_mps_1s_bundle.pkl",
    "evidence/av_nuplan/uncertainty/nuplan_av_ego_speed_mps_1s_conformal.json",
    "evidence/healthcare/runtime_summary.csv",
    "evidence/healthcare/site_splits/manifest.json",
    "manuscripts/paper.pdf",
    "manuscripts/monograph/paper.pdf",
    "manuscripts/monograph/paper.docx",
    "manuscripts/ieee/orius_ieee_main.pdf",
    "manuscripts/ieee/orius_ieee_main.docx",
]

DISALLOWED_PARTS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}

DISALLOWED_SUFFIXES = {
    ".crdownload",
    ".env",
    ".key",
    ".pem",
    ".zip",
}

SENSITIVE_NAME_MARKERS = (
    "apikey",
    "api_key",
    "credentials",
    "secret",
    "token",
)

STALE_REPORT_MARKERS = (
    "reports/legacy_archive/",
    "reports/orius_av/full_corpus/",
    "reports/orius_av/nuplan_bounded/",
    "reports/orius_av/waymo_dataset_audit/",
)

PUBLIC_STALE_TEXT_MARKERS = (
    "six-domain",
    "six defended domains",
    "six row",
    "six-row",
    "all six orius",
    "full autonomous-driving field closure claimed",
)


def posix(path: Path | str) -> str:
    return Path(path).as_posix()
