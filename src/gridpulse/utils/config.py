"""Utilities: config validation models and helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Type

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

    model_config = ConfigDict(extra="allow")


class ForecastConfig(BaseModel):
    """Schema for configs/forecast.yaml."""
    data: dict[str, Any] = Field(default_factory=dict)
    models: dict[str, Any] = Field(default_factory=dict)
    fallback_order: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")


CONFIG_MODELS: dict[str, Type[BaseModel]] = {
    "data.yaml": DataConfig,
    "optimization.yaml": OptimizationConfig,
    "train_forecast.yaml": TrainForecastConfig,
    "monitoring.yaml": MonitoringConfig,
    "forecast.yaml": ForecastConfig,
}


def _load_yaml(path: Path) -> dict:
    """Read a YAML file into a dict, defaulting to empty."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload or {}


def validate_config(path: Path) -> None:
    """Validate a config file if a schema is registered."""
    model = CONFIG_MODELS.get(path.name)
    if not model:
        return
    payload = _load_yaml(path)
    model.model_validate(payload)
