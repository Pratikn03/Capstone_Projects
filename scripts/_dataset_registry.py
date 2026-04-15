"""Dataset registry for the unified training script.

Single source of truth for all dataset configurations, paths, and defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DatasetConfig:
    """Configuration for a registered dataset."""
    name: str                           # Short name (DE, US, etc.)
    display_name: str                   # Full name for logging
    config_file: str                    # Path to training config YAML
    features_path: str                  # Path to features.parquet
    splits_path: str                    # Path to splits directory
    models_dir: str                     # Path to model artifacts directory
    reports_dir: str                    # Path to reports output
    uncertainty_dir: str                # Path to conformal artifacts directory
    backtests_dir: str                  # Path to calibration/test artifact directory

    # Feature pipeline settings
    raw_data_path: str                  # Path to raw data
    feature_module: str                 # Module to build features

    # Optional settings
    ba_code: Optional[str] = None       # Balancing authority (for US)
    start_date: Optional[str] = None    # Date filter start
    end_date: Optional[str] = None      # Date filter end
    alias_of: Optional[str] = None      # Backward-compatible dataset alias
    provenance_path: Optional[str] = None  # Standardized real-data manifest when available
    canonical_raw_source_path: Optional[str] = None
    runtime_domain: Optional[str] = None
    publication_label: Optional[str] = None
    closure_target_tier: Optional[str] = None
    maturity_tier: Optional[str] = None
    canonical_runtime_path: Optional[str] = None
    support_runtime_path: Optional[str] = None
    runtime_provenance_path: Optional[str] = None
    support_runtime_provenance_path: Optional[str] = None
    fallback_policy: Optional[str] = None
    exact_blocker: Optional[str] = None
    strict_runtime_required: bool = False


def _us_dataset(
    *,
    key: str,
    display_name: str,
    config_file: str,
    processed_dir: str,
    models_dir: str,
    reports_dir: str,
    uncertainty_dir: str,
    backtests_dir: str,
    ba_code: str,
    alias_of: str | None = None,
) -> DatasetConfig:
    return DatasetConfig(
        name=key,
        display_name=display_name,
        config_file=config_file,
        features_path=f"{processed_dir}/features.parquet",
        splits_path=f"{processed_dir}/splits",
        models_dir=models_dir,
        reports_dir=reports_dir,
        uncertainty_dir=uncertainty_dir,
        backtests_dir=backtests_dir,
        raw_data_path="data/raw/us_eia930",
        feature_module="orius.data_pipeline.build_features_eia930",
        ba_code=ba_code,
        alias_of=alias_of,
        provenance_path=f"{processed_dir}/dataset_provenance.json",
        canonical_raw_source_path="data/raw/us_eia930",
    )


# Dataset registry - single source of truth for all datasets
DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "DE": DatasetConfig(
        name="DE",
        display_name="German OPSD",
        config_file="configs/train_forecast.yaml",
        features_path="data/processed/features.parquet",
        splits_path="data/processed/splits",
        models_dir="artifacts/models",
        reports_dir="reports",
        uncertainty_dir="artifacts/uncertainty",
        backtests_dir="artifacts/backtests",
        raw_data_path="data/raw",
        feature_module="orius.data_pipeline.build_features",
        provenance_path="data/raw/opsd_germany_provenance.json",
        canonical_raw_source_path="data/raw/time_series_60min_singleindex.csv",
        runtime_domain="battery",
        publication_label="Battery Energy Storage",
        closure_target_tier="witness_row",
        maturity_tier="reference",
        fallback_policy="paper6_runtime",
        exact_blocker="battery_reference_witness",
    ),
    "US_MISO": _us_dataset(
        key="US_MISO",
        display_name="US EIA-930 (MISO)",
        config_file="configs/train_forecast_eia930.yaml",
        processed_dir="data/processed/us_eia930",
        models_dir="artifacts/models_eia930",
        reports_dir="reports/eia930",
        uncertainty_dir="artifacts/uncertainty/eia930",
        backtests_dir="artifacts/backtests/eia930",
        ba_code="MISO",
    ),
    "US_PJM": _us_dataset(
        key="US_PJM",
        display_name="US EIA-930 (PJM)",
        config_file="configs/train_forecast_eia930_pjm.yaml",
        processed_dir="data/processed/us_eia930_pjm",
        models_dir="artifacts/models_eia930_pjm",
        reports_dir="reports/eia930_pjm",
        uncertainty_dir="artifacts/uncertainty/eia930_pjm",
        backtests_dir="artifacts/backtests/eia930_pjm",
        ba_code="PJM",
    ),
    "US_ERCOT": _us_dataset(
        key="US_ERCOT",
        display_name="US EIA-930 (ERCOT)",
        config_file="configs/train_forecast_eia930_ercot.yaml",
        processed_dir="data/processed/us_eia930_ercot",
        models_dir="artifacts/models_eia930_ercot",
        reports_dir="reports/eia930_ercot",
        uncertainty_dir="artifacts/uncertainty/eia930_ercot",
        backtests_dir="artifacts/backtests/eia930_ercot",
        ba_code="ERCO",
    ),
    # Backward-compatible alias for the historical single-region US pipeline.
    "US": _us_dataset(
        key="US_MISO",
        display_name="US EIA-930 (MISO)",
        config_file="configs/train_forecast_eia930.yaml",
        processed_dir="data/processed/us_eia930",
        models_dir="artifacts/models_eia930",
        reports_dir="reports/eia930",
        uncertainty_dir="artifacts/uncertainty/eia930",
        backtests_dir="artifacts/backtests/eia930",
        ba_code="MISO",
        alias_of="US_MISO",
    ),
    # Multi-domain (AV, Industrial, Healthcare, Aerospace)
    "AV": DatasetConfig(
        name="AV",
        display_name="AV Trajectories",
        config_file="configs/train_forecast_av.yaml",
        features_path="data/av/processed/features.parquet",
        splits_path="data/av/processed/splits",
        models_dir="artifacts/models_av",
        reports_dir="reports/av",
        uncertainty_dir="artifacts/uncertainty/av",
        backtests_dir="artifacts/backtests/av",
        raw_data_path="data/av/processed/av_trajectories_orius.csv",
        feature_module="orius.data_pipeline.build_features_av",
        provenance_path="data/av/raw/waymo_open_motion_provenance.json",
        canonical_raw_source_path="data/av/raw/waymo_open_motion",
        runtime_domain="vehicle",
        publication_label="Autonomous Vehicles",
        closure_target_tier="defended_bounded_row",
        maturity_tier="proof_validated",
        canonical_runtime_path="data/av/processed/av_trajectories_orius.csv",
        runtime_provenance_path="data/av/raw/waymo_open_motion_provenance.json",
        fallback_policy="bounded_runtime_pass",
        exact_blocker="av_real_row_present",
    ),
    "INDUSTRIAL": DatasetConfig(
        name="INDUSTRIAL",
        display_name="Industrial Process",
        config_file="configs/train_forecast_industrial.yaml",
        features_path="data/industrial/processed/features.parquet",
        splits_path="data/industrial/processed/splits",
        models_dir="artifacts/models_industrial",
        reports_dir="reports/industrial",
        uncertainty_dir="artifacts/uncertainty/industrial",
        backtests_dir="artifacts/backtests/industrial",
        raw_data_path="data/industrial/processed/industrial_orius.csv",
        feature_module="orius.data_pipeline.build_features_industrial",
        provenance_path="data/industrial/raw/ccpp_provenance.json",
        canonical_raw_source_path="data/industrial/raw/CCPP.csv",
        runtime_domain="industrial",
        publication_label="Industrial Process Control",
        closure_target_tier="defended_bounded_row",
        maturity_tier="proof_validated",
        canonical_runtime_path="data/industrial/processed/industrial_orius.csv",
        runtime_provenance_path="data/industrial/raw/ccpp_provenance.json",
        fallback_policy="bounded_runtime_pass",
        exact_blocker="industrial_train_validation_chain_complete",
    ),
    "HEALTHCARE": DatasetConfig(
        name="HEALTHCARE",
        display_name="Healthcare Vital Signs",
        config_file="configs/train_forecast_healthcare.yaml",
        features_path="data/healthcare/processed/features.parquet",
        splits_path="data/healthcare/processed/splits",
        models_dir="artifacts/models_healthcare",
        reports_dir="reports/healthcare",
        uncertainty_dir="artifacts/uncertainty/healthcare",
        backtests_dir="artifacts/backtests/healthcare",
        raw_data_path="data/healthcare/processed/healthcare_orius.csv",
        feature_module="orius.data_pipeline.build_features_healthcare",
        provenance_path="data/healthcare/raw/bidmc_provenance.json",
        canonical_raw_source_path="data/healthcare/raw/bidmc_csv",
        runtime_domain="healthcare",
        publication_label="Medical and Healthcare Monitoring",
        closure_target_tier="defended_bounded_row",
        maturity_tier="proof_validated",
        canonical_runtime_path="data/healthcare/processed/healthcare_orius.csv",
        runtime_provenance_path="data/healthcare/raw/bidmc_provenance.json",
        fallback_policy="bounded_runtime_pass",
        exact_blocker="healthcare_train_validation_chain_complete",
    ),
    "AEROSPACE": DatasetConfig(
        name="AEROSPACE",
        display_name="Aerospace Flight",
        config_file="configs/train_forecast_aerospace.yaml",
        features_path="data/aerospace/processed/features.parquet",
        splits_path="data/aerospace/processed/splits",
        models_dir="artifacts/models_aerospace",
        reports_dir="reports/aerospace",
        uncertainty_dir="artifacts/uncertainty/aerospace",
        backtests_dir="artifacts/backtests/aerospace",
        raw_data_path="data/aerospace/processed/aerospace_orius.csv",
        feature_module="orius.data_pipeline.build_features_aerospace",
        provenance_path="data/aerospace/raw/cmapss_provenance.json",
        canonical_raw_source_path="data/aerospace/raw",
        runtime_domain="aerospace",
        publication_label="Aerospace Control",
        closure_target_tier="defended_bounded_row",
        maturity_tier="experimental",
        canonical_runtime_path="data/aerospace/processed/aerospace_realflight_runtime.csv",
        support_runtime_path="data/aerospace/processed/aerospace_public_adsb_runtime.csv",
        runtime_provenance_path="data/aerospace/raw/aerospace_realflight_provenance.json",
        support_runtime_provenance_path="data/aerospace/raw/public_adsb_proxy_provenance.json",
        fallback_policy="experimental_support_lane",
        exact_blocker="aerospace_realflight_runtime_missing",
        strict_runtime_required=True,
    ),
    "NAVIGATION": DatasetConfig(
        name="NAVIGATION",
        display_name="Navigation Trajectories",
        config_file="configs/train_forecast_navigation.yaml",
        features_path="data/navigation/processed/features.parquet",
        splits_path="data/navigation/processed/splits",
        models_dir="artifacts/models_navigation",
        reports_dir="reports/navigation",
        uncertainty_dir="artifacts/uncertainty/navigation",
        backtests_dir="artifacts/backtests/navigation",
        raw_data_path="data/navigation/processed/navigation_orius.csv",
        feature_module="orius.data_pipeline.build_features_navigation",
        provenance_path="data/navigation/raw/kitti_odometry_provenance.json",
        canonical_raw_source_path="data/navigation/raw/kitti_odometry",
        runtime_domain="navigation",
        publication_label="Navigation and Guidance",
        closure_target_tier="defended_bounded_row",
        maturity_tier="shadow_synthetic",
        canonical_runtime_path="data/navigation/processed/navigation_orius.csv",
        runtime_provenance_path="data/navigation/raw/kitti_odometry_provenance.json",
        fallback_policy="shadow_synthetic_support_tier",
        exact_blocker="navigation_kitti_runtime_missing",
        strict_runtime_required=True,
    ),
}

AGGRESSIVE_DEFAULTS = {
    "DE": {"n_trials": 220, "top_pct": 0.20, "max_seeds": 8},
    "US_MISO": {"n_trials": 260, "top_pct": 0.20, "max_seeds": 5},
    "US_PJM": {"n_trials": 260, "top_pct": 0.20, "max_seeds": 5},
    "US_ERCOT": {"n_trials": 260, "top_pct": 0.20, "max_seeds": 5},
    "AV": {"n_trials": 50, "top_pct": 0.20, "max_seeds": 2},
    "INDUSTRIAL": {"n_trials": 50, "top_pct": 0.20, "max_seeds": 2},
    "HEALTHCARE": {"n_trials": 50, "top_pct": 0.20, "max_seeds": 2},
    "AEROSPACE": {"n_trials": 50, "top_pct": 0.20, "max_seeds": 2},
    "NAVIGATION": {"n_trials": 50, "top_pct": 0.20, "max_seeds": 2},
}


@dataclass
class RunLayout:
    """Resolved output paths for a canonical or candidate training run."""

    mode: str
    run_id: str
    dataset: str
    artifacts_root: Path
    models_dir: Path
    uncertainty_dir: Path
    backtests_dir: Path
    registry_dir: Path
    reports_dir: Path
    publication_dir: Path
    validation_report: Path
    data_manifest_output: Path
    walk_forward_report: Path
    selection_output_dir: Path

    @property
    def is_candidate(self) -> bool:
        return self.mode == "candidate"


def iter_trainable_dataset_keys() -> list[str]:
    """Return canonical (non-alias) dataset keys in registry order."""
    keys: list[str] = []
    seen: set[str] = set()
    for registry_key, cfg in DATASET_REGISTRY.items():
        canonical = cfg.alias_of or cfg.name
        if canonical in seen:
            continue
        seen.add(canonical)
        keys.append(registry_key)
    return keys


RUNTIME_DOMAIN_DATASET_KEYS: dict[str, str] = {
    "battery": "DE",
    "vehicle": "AV",
    "industrial": "INDUSTRIAL",
    "healthcare": "HEALTHCARE",
    "navigation": "NAVIGATION",
    "aerospace": "AEROSPACE",
}


def repo_path(relative_path: str | None) -> Path | None:
    if not relative_path:
        return None
    return REPO_ROOT / relative_path


def get_runtime_dataset_config(domain: str) -> DatasetConfig:
    try:
        return DATASET_REGISTRY[RUNTIME_DOMAIN_DATASET_KEYS[domain]]
    except KeyError as exc:
        raise KeyError(f"No runtime dataset config registered for domain '{domain}'.") from exc


def runtime_domain_configs() -> dict[str, DatasetConfig]:
    return {
        domain: DATASET_REGISTRY[key]
        for domain, key in RUNTIME_DOMAIN_DATASET_KEYS.items()
    }


def get_runtime_dataset_path(domain: str, *, allow_support_tier: bool = False) -> Path | None:
    cfg = get_runtime_dataset_config(domain)
    for relative_path in (
        cfg.canonical_runtime_path,
        cfg.support_runtime_path if allow_support_tier else None,
    ):
        path = repo_path(relative_path)
        if path is not None and path.exists():
            return path
    return None


def get_runtime_source_label(domain: str, *, allow_support_tier: bool = False) -> str:
    cfg = get_runtime_dataset_config(domain)
    canonical_path = repo_path(cfg.canonical_runtime_path)
    if canonical_path is not None and canonical_path.exists():
        return "canonical"
    if allow_support_tier:
        support_path = repo_path(cfg.support_runtime_path)
        if support_path is not None and support_path.exists():
            return "support"
    return "missing"
