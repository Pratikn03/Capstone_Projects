"""Utilities: config validation models and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SignalsConfig(BaseModel):
    """Signals config in data.yaml (price/carbon files)."""

    enabled: bool = False
    file: str | None = None

    model_config = ConfigDict(extra="allow")


class DataConfig(BaseModel):
    """Top-level data config schema."""

    signals: SignalsConfig | None = None

    model_config = ConfigDict(extra="allow")


class ObjectiveConfig(BaseModel):
    """Optimization objective weights."""

    cost_weight: float = 1.0
    carbon_weight: float = 0.0

    model_config = ConfigDict(extra="allow")


class CarbonConfig(BaseModel):
    """Carbon signal options for optimization."""

    source: str = "average"
    budget_reduction_pct: float | None = None
    budget_kg: float | None = None

    model_config = ConfigDict(extra="allow")


class BatteryConfig(BaseModel):
    """Battery constraint defaults for optimization."""

    capacity_mwh: float = 10.0
    max_power_mw: float = 2.0
    efficiency: float = 0.9
    efficiency_regime_a: float | None = None
    efficiency_regime_b: float | None = None
    efficiency_soc_split: float = 0.80
    degradation_cost_per_mwh: float = 10.0
    min_soc_mwh: float = 0.0
    initial_soc_mwh: float = 0.0

    model_config = ConfigDict(extra="allow")


class GridConfig(BaseModel):
    """Grid constraint defaults for optimization."""

    max_import_mw: float = 50.0
    price_per_mwh: float = 70.0
    carbon_cost_per_mwh: float = 20.0
    carbon_kg_per_mwh: float = 400.0

    model_config = ConfigDict(extra="allow")


class OptimizationConfig(BaseModel):
    """Schema for configs/optimization.yaml."""

    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    carbon: CarbonConfig = Field(default_factory=CarbonConfig)
    battery: BatteryConfig = Field(default_factory=BatteryConfig)
    grid: GridConfig = Field(default_factory=GridConfig)

    model_config = ConfigDict(extra="allow")


class TaskConfig(BaseModel):
    """Forecasting task configuration."""

    horizon_hours: int = 24
    lookback_hours: int = 168
    targets: list[str] = Field(default_factory=list)
    quantiles: list[float] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class TrainDataConfig(BaseModel):
    """Train-time data paths."""

    processed_path: str = "data/processed/features.parquet"
    timestamp_col: str = "timestamp"

    model_config = ConfigDict(extra="allow")


class TrainForecastConfig(BaseModel):
    """Schema for configs/train_forecast.yaml."""

    task: TaskConfig
    data: TrainDataConfig
    seed: int = 42
    models: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    reports: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class MonitoringConfig(BaseModel):
    """Schema for configs/monitoring.yaml."""

    data_drift: dict[str, Any] = Field(default_factory=dict)
    model_drift: dict[str, Any] = Field(default_factory=dict)
    retraining: dict[str, Any] = Field(default_factory=dict)
    dc3s_health: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class PublishAuditConfig(BaseModel):
    """Schema for configs/publish_audit.yaml."""

    publish_audit: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class ForecastConfig(BaseModel):
    """Schema for configs/forecast.yaml."""

    data: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    fallback_order: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


class UncertaintyConformalConfig(BaseModel):
    alpha: float = 0.10
    horizon_wise: bool = True
    rolling: bool = True
    rolling_window: int = 720
    eps: float = 1e-6

    model_config = ConfigDict(extra="allow")


class UncertaintyConfig(BaseModel):
    enabled: bool = True
    target: str = "load_mw"
    targets: list[str] | None = None
    calibration_split: str = "val"
    artifacts_dir: str = "artifacts/uncertainty"
    calibration_npz: str = "artifacts/backtests/calibration.npz"
    test_npz: str = "artifacts/backtests/test.npz"
    conformal: UncertaintyConformalConfig = Field(default_factory=UncertaintyConformalConfig)

    model_config = ConfigDict(extra="allow")


class StreamingKafkaConfig(BaseModel):
    bootstrap_servers: str = "localhost:9092"
    topic: str = "orius.opsd.v1"
    group_id: str = "orius-consumer"
    auto_offset_reset: str = "earliest"

    model_config = ConfigDict(extra="allow")


class StreamingStorageConfig(BaseModel):
    mode: str = "duckdb"
    duckdb_path: str = "data/interim/streaming.duckdb"
    table_name: str = "telemetry_events"
    parquet_dir: str = "data/interim/streaming_parquet"

    model_config = ConfigDict(extra="allow")


class StreamingCheckpointConfig(BaseModel):
    path: str = "artifacts/checkpoints/streaming_checkpoint.json"

    model_config = ConfigDict(extra="allow")


class StreamingValidationConfig(BaseModel):
    strict: bool = True
    cadence_seconds: int = 3600
    cadence_tolerance_seconds: int = 120
    min_mw: float = 0.0
    max_mw: float = 200000.0
    max_delta_mw: float | None = None

    model_config = ConfigDict(extra="allow")


class StreamingConfig(BaseModel):
    kafka: StreamingKafkaConfig = Field(default_factory=StreamingKafkaConfig)
    storage: StreamingStorageConfig = Field(default_factory=StreamingStorageConfig)
    checkpoint: StreamingCheckpointConfig = Field(default_factory=StreamingCheckpointConfig)
    validation: StreamingValidationConfig = Field(default_factory=StreamingValidationConfig)

    model_config = ConfigDict(extra="allow")


class ShiftAwareUncertaintyConfig(BaseModel):
    enable: bool = False
    policy_mode: str = "legacy_rac_cert"
    aci_mode: str = "fixed"
    adaptation_step_size: float = 0.01
    reliability_bin_count: int = 5
    volatility_bin_count: int = 5
    subgroup_definitions: dict[str, Any] = Field(default_factory=dict)
    validity_score_weights: dict[str, float] = Field(default_factory=dict)
    drift_detector_params: dict[str, Any] = Field(default_factory=dict)
    widening_caps: dict[str, float] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    artifact_output_toggles: dict[str, bool] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


CONFIG_MODELS: dict[str, type[BaseModel]] = {
    "data.yaml": DataConfig,
    "optimization.yaml": OptimizationConfig,
    "train_forecast.yaml": TrainForecastConfig,
    "monitoring.yaml": MonitoringConfig,
    "publish_audit.yaml": PublishAuditConfig,
    "forecast.yaml": ForecastConfig,
    "uncertainty.yaml": UncertaintyConfig,
    "streaming.yaml": StreamingConfig,
    "shift_aware_uncertainty.yaml": ShiftAwareUncertaintyConfig,
}


class _AttrDict(dict):
    """Dict subclass that allows attribute access for convenience."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return _AttrDict({k: _to_attrdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attrdict(item) for item in obj]
    return obj


def _load_yaml(path: Path) -> dict:
    """Read a YAML file into a dict, defaulting to empty."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def load_config(path: str | Path) -> _AttrDict:
    """Load a YAML config file and return an attribute-accessible dict."""
    data = _load_yaml(Path(path))
    return _to_attrdict(data)


def validate_config(path: Path) -> None:
    """Validate a config file if a schema is registered."""
    model = CONFIG_MODELS.get(path.name)
    if not model:
        return
    payload = _load_yaml(path)
    model.model_validate(payload)
