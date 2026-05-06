"""Dataset registry for the unified training script.

Single source of truth for all dataset configurations, paths, and defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DatasetConfig:
    """Configuration for a registered dataset."""

    name: str  # Short name (DE, US, etc.)
    display_name: str  # Full name for logging
    config_file: str  # Path to training config YAML
    features_path: str  # Path to features.parquet
    splits_path: str  # Path to splits directory
    models_dir: str  # Path to model artifacts directory
    reports_dir: str  # Path to reports output
    uncertainty_dir: str  # Path to conformal artifacts directory
    backtests_dir: str  # Path to calibration/test artifact directory

    # Feature pipeline settings
    raw_data_path: str  # Path to raw data
    feature_module: str  # Module to build features

    # Optional settings
    ba_code: str | None = None  # Balancing authority (for US)
    start_date: str | None = None  # Date filter start
    end_date: str | None = None  # Date filter end
    alias_of: str | None = None  # Backward-compatible dataset alias
    provenance_path: str | None = None  # Standardized real-data manifest when available
    canonical_raw_source_path: str | None = None
    runtime_domain: str | None = None
    publication_label: str | None = None
    closure_target_tier: str | None = None
    maturity_tier: str | None = None
    canonical_runtime_path: str | None = None
    support_runtime_path: str | None = None
    runtime_provenance_path: str | None = None
    support_runtime_provenance_path: str | None = None
    fallback_policy: str | None = None
    exact_blocker: str | None = None
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
    # Multi-domain promoted program (AV, Healthcare)
    # AV: nuPlan all-zip grouped runtime replay is the promoted AV surface.
    # Legacy Waymo/HEE compatibility paths remain elsewhere for reversibility.
    "AV": DatasetConfig(
        name="AV",
        display_name="nuPlan All-Zip Grouped AV Replay",
        config_file="configs/train_forecast_av.yaml",
        features_path="data/orius_av/av/processed_nuplan_allzip_grouped/anchor_features.parquet",
        splits_path="data/orius_av/av/processed_nuplan_allzip_grouped/splits",
        models_dir="artifacts/models_orius_av_nuplan_allzip_grouped",
        reports_dir="reports/orius_av/nuplan_allzip_grouped",
        uncertainty_dir="artifacts/uncertainty/orius_av_nuplan_allzip_grouped",
        backtests_dir="reports/orius_av/nuplan_allzip_grouped",
        raw_data_path="data/orius_av/av/processed_nuplan_allzip_grouped/replay_windows.parquet",
        feature_module="orius.data_pipeline.build_features_av",
        provenance_path="data/orius_av/av/processed_nuplan_allzip_grouped/nuplan_source_manifest.json",
        canonical_raw_source_path="data/orius_av/raw",
        runtime_domain="vehicle",
        publication_label="Autonomous Vehicles",
        closure_target_tier="defended_bounded_row",
        maturity_tier="proof_validated",
        canonical_runtime_path="data/orius_av/av/processed_nuplan_allzip_grouped/anchor_features.parquet",
        runtime_provenance_path="reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/runtime_report.json",
        fallback_policy="bounded_runtime_pass",
        exact_blocker="nuplan_allzip_grouped_row_present",
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
        raw_data_path="data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv",
        feature_module="orius.data_pipeline.build_features_healthcare",
        provenance_path="data/healthcare/mimic3/processed/mimic3_manifest.json",
        canonical_raw_source_path="data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv",
        runtime_domain="healthcare",
        publication_label="Medical and Healthcare Monitoring",
        closure_target_tier="defended_bounded_row",
        maturity_tier="proof_validated",
        canonical_runtime_path="data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv",
        runtime_provenance_path="data/healthcare/mimic3/processed/mimic3_manifest.json",
        fallback_policy="bounded_runtime_pass",
        exact_blocker="healthcare_train_validation_chain_complete",
    ),
}

AGGRESSIVE_DEFAULTS = {
    "DE": {"n_trials": 220, "top_pct": 0.20, "max_seeds": 8},
    "US_MISO": {"n_trials": 260, "top_pct": 0.20, "max_seeds": 5},
    "US_PJM": {"n_trials": 260, "top_pct": 0.20, "max_seeds": 5},
    "US_ERCOT": {"n_trials": 260, "top_pct": 0.20, "max_seeds": 5},
    "AV": {"n_trials": 50, "top_pct": 0.20, "max_seeds": 2},
    "HEALTHCARE": {"n_trials": 50, "top_pct": 0.20, "max_seeds": 2},
}

MAX_QUALITY_DEFAULTS = {
    "DE": {"n_trials": 420, "top_pct": 0.10, "max_seeds": 12},
    "US_MISO": {"n_trials": 480, "top_pct": 0.10, "max_seeds": 8},
    "US_PJM": {"n_trials": 480, "top_pct": 0.10, "max_seeds": 8},
    "US_ERCOT": {"n_trials": 480, "top_pct": 0.10, "max_seeds": 8},
    "AV": {"n_trials": 140, "top_pct": 0.15, "max_seeds": 4},
    "HEALTHCARE": {"n_trials": 140, "top_pct": 0.15, "max_seeds": 4},
}

PRODUCTION_MAX_FAST_DEFAULTS = {
    "DE": {
        "n_trials": 64,
        "top_pct": 0.20,
        "max_seeds": 4,
        "tuning_n_jobs": 3,
        "gbm_threads": 2,
        "max_deep_epochs": 16,
        "deep_patience": 4,
        "deep_warmup_epochs": 2,
        "reuse_best_gbm": True,
    },
    "US_MISO": {
        "n_trials": 80,
        "top_pct": 0.20,
        "max_seeds": 4,
        "tuning_n_jobs": 2,
        "gbm_threads": 2,
        "max_deep_epochs": 16,
        "deep_patience": 4,
        "deep_warmup_epochs": 2,
        "reuse_best_gbm": True,
    },
    "US_PJM": {
        "n_trials": 80,
        "top_pct": 0.20,
        "max_seeds": 4,
        "tuning_n_jobs": 2,
        "gbm_threads": 2,
        "max_deep_epochs": 16,
        "deep_patience": 4,
        "deep_warmup_epochs": 2,
        "reuse_best_gbm": True,
    },
    "US_ERCOT": {
        "n_trials": 80,
        "top_pct": 0.20,
        "max_seeds": 4,
        "tuning_n_jobs": 2,
        "gbm_threads": 2,
        "max_deep_epochs": 16,
        "deep_patience": 4,
        "deep_warmup_epochs": 2,
        "reuse_best_gbm": True,
    },
    "AV": {
        "n_trials": 48,
        "top_pct": 0.20,
        "max_seeds": 3,
        "tuning_n_jobs": 2,
        "gbm_threads": 2,
        "max_deep_epochs": 12,
        "deep_patience": 3,
        "deep_warmup_epochs": 1,
        "reuse_best_gbm": False,
    },
    "HEALTHCARE": {
        "n_trials": 48,
        "top_pct": 0.20,
        "max_seeds": 3,
        "tuning_n_jobs": 2,
        "gbm_threads": 2,
        "max_deep_epochs": 12,
        "deep_patience": 3,
        "deep_warmup_epochs": 1,
        "reuse_best_gbm": True,
    },
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
    "healthcare": "HEALTHCARE",
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
    return {domain: DATASET_REGISTRY[key] for domain, key in RUNTIME_DOMAIN_DATASET_KEYS.items()}


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
