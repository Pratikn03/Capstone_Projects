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
        feature_module="gridpulse.data_pipeline.build_features_eia930",
        ba_code=ba_code,
        alias_of=alias_of,
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
        feature_module="gridpulse.data_pipeline.build_features",
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
}

AGGRESSIVE_DEFAULTS = {
    "DE": {"n_trials": 220, "top_pct": 0.20, "max_seeds": 8},
    "US_MISO": {"n_trials": 260, "top_pct": 0.20, "max_seeds": 5},
    "US_PJM": {"n_trials": 260, "top_pct": 0.20, "max_seeds": 5},
    "US_ERCOT": {"n_trials": 260, "top_pct": 0.20, "max_seeds": 5},
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
