"""
PyTest Configuration and Fixtures for GridPulse Tests.

This module provides comprehensive test fixtures for unit tests,
integration tests, and performance tests.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Return the test data directory."""
    return project_root / "tests" / "data"


@pytest.fixture(scope="session")
def artifacts_dir(project_root: Path) -> Path:
    """Return the artifacts directory."""
    return project_root / "artifacts"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_timestamp_range() -> pd.DatetimeIndex:
    """Create a sample timestamp range (1 week hourly)."""
    return pd.date_range(
        start="2024-01-01",
        periods=168,  # 1 week
        freq="h",
    )


@pytest.fixture
def sample_load_data(sample_timestamp_range: pd.DatetimeIndex) -> pd.DataFrame:
    """Create sample load data for testing."""
    np.random.seed(42)
    n = len(sample_timestamp_range)
    
    # Realistic load pattern with daily seasonality
    hours = np.arange(n)
    daily_pattern = 5000 * np.sin(2 * np.pi * hours / 24 - np.pi / 2)
    base_load = 45000
    noise = np.random.normal(0, 1000, n)
    
    return pd.DataFrame({
        "timestamp": sample_timestamp_range,
        "load_mw": base_load + daily_pattern + noise,
        "wind_mw": np.abs(np.random.normal(3000, 1500, n)),
        "solar_mw": np.maximum(0, 2000 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 500, n)),
        "price_eur_mwh": 50 + 20 * np.random.random(n),
    })


@pytest.fixture
def sample_features_df(sample_load_data: pd.DataFrame) -> pd.DataFrame:
    """Create sample feature-engineered dataframe."""
    df = sample_load_data.copy()
    
    # Add lag features
    for lag in [1, 24, 168]:
        df[f"load_mw_lag_{lag}"] = df["load_mw"].shift(lag)
        df[f"wind_mw_lag_{lag}"] = df["wind_mw"].shift(lag)
    
    # Add rolling features
    df["load_mw_roll_24_mean"] = df["load_mw"].rolling(24).mean()
    df["load_mw_roll_24_std"] = df["load_mw"].rolling(24).std()
    
    # Add calendar features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    
    return df.dropna()


@pytest.fixture
def sample_forecast_output() -> Dict[str, np.ndarray]:
    """Create sample forecast output."""
    np.random.seed(42)
    horizon = 24
    
    return {
        "predictions": np.random.normal(45000, 1000, horizon),
        "lower_bound": np.random.normal(43000, 1000, horizon),
        "upper_bound": np.random.normal(47000, 1000, horizon),
        "timestamps": pd.date_range("2024-01-08", periods=horizon, freq="h"),
    }


@pytest.fixture
def sample_optimization_input() -> Dict[str, Any]:
    """Create sample optimization input."""
    horizon = 24
    np.random.seed(42)
    
    return {
        "load_forecast": (45000 + np.random.normal(0, 1000, horizon)).tolist(),
        "wind_forecast": np.abs(np.random.normal(3000, 1000, horizon)).tolist(),
        "solar_forecast": np.maximum(0, np.random.normal(2000, 500, horizon)).tolist(),
        "price_forecast": (50 + np.random.random(horizon) * 30).tolist(),
        "battery_capacity_mwh": 100.0,
        "battery_max_power_mw": 50.0,
        "initial_soc_mwh": 50.0,
    }


# =============================================================================
# MODEL FIXTURES
# =============================================================================

@pytest.fixture
def mock_gbm_model() -> MagicMock:
    """Create a mock GBM model."""
    model = MagicMock()
    model.predict.return_value = np.random.normal(45000, 500, 24)
    model.feature_names_ = ["load_mw_lag_1", "load_mw_lag_24", "hour", "dayofweek"]
    return model


@pytest.fixture
def mock_forecast_service(mock_gbm_model: MagicMock) -> MagicMock:
    """Create a mock forecast service."""
    service = MagicMock()
    service.predict.return_value = {
        "predictions": np.random.normal(45000, 500, 24).tolist(),
        "model": "gbm",
        "target": "load_mw",
    }
    return service


# =============================================================================
# API FIXTURES
# =============================================================================

@pytest.fixture
def api_client():
    """Create a test client for the FastAPI application."""
    from fastapi.testclient import TestClient
    
    # Mock environment variables
    with patch.dict(os.environ, {"GRIDPULSE_API_KEY": "test-key"}):
        from services.api.main import app
        yield TestClient(app)


@pytest.fixture
def api_headers() -> Dict[str, str]:
    """Return standard API headers for testing."""
    return {
        "Content-Type": "application/json",
        "X-API-Key": "test-key",
    }


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def sample_battery_config() -> Dict[str, Any]:
    """Create sample battery configuration."""
    return {
        "capacity_mwh": 100.0,
        "max_charge_mw": 50.0,
        "max_discharge_mw": 50.0,
        "efficiency_charge": 0.95,
        "efficiency_discharge": 0.95,
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "initial_soc_mwh": 50.0,
        "degradation_cost_per_mwh": 5.0,
    }


@pytest.fixture
def sample_optimization_config(sample_battery_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create sample optimization configuration."""
    return {
        "battery": sample_battery_config,
        "grid": {
            "max_import_mw": 100000.0,
            "price_per_mwh": 60.0,
        },
        "risk": {
            "enabled": True,
            "weight_worst_case": 1.0,
        },
        "solver": "highs",
    }


# =============================================================================
# STREAMING FIXTURES
# =============================================================================

@pytest.fixture
def sample_kafka_message() -> Dict[str, Any]:
    """Create a sample Kafka message."""
    return {
        "utc_timestamp": datetime.utcnow().isoformat(),
        "load_mw": 45000.0,
        "wind_mw": 3000.0,
        "solar_mw": 2000.0,
        "price_eur_mwh": 55.0,
    }


@pytest.fixture
def mock_kafka_consumer() -> MagicMock:
    """Create a mock Kafka consumer."""
    consumer = MagicMock()
    consumer.__iter__ = MagicMock(return_value=iter([]))
    return consumer


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture
def mock_duckdb_connection(temp_dir: Path) -> Generator[Any, None, None]:
    """Create a temporary DuckDB connection."""
    import duckdb
    
    db_path = temp_dir / "test.duckdb"
    conn = duckdb.connect(str(db_path))
    yield conn
    conn.close()


# =============================================================================
# METRIC FIXTURES
# =============================================================================

@pytest.fixture
def sample_evaluation_results() -> Dict[str, float]:
    """Create sample evaluation results."""
    return {
        "rmse": 271.2,
        "mae": 161.1,
        "mape": 0.0035,
        "r2": 0.9991,
        "smape": 0.0035,
    }


@pytest.fixture
def sample_coverage_results() -> Dict[str, Any]:
    """Create sample prediction interval coverage results."""
    return {
        "picp": 0.952,
        "mpiw": 742.7,
        "n_test": 1739,
        "nominal_coverage": 0.90,
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, rtol: float = 1e-5):
    """Assert two DataFrames are approximately equal."""
    pd.testing.assert_frame_equal(df1, df2, rtol=rtol)


def assert_array_almost_equal(arr1: np.ndarray, arr2: np.ndarray, decimal: int = 5):
    """Assert two arrays are approximately equal."""
    np.testing.assert_array_almost_equal(arr1, arr2, decimal=decimal)


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring external services"
    )
    config.addinivalue_line(
        "markers", "load: marks load/performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if config.getoption("-m"):
        return
    
    # Skip slow tests by default in CI
    if os.environ.get("CI"):
        skip_slow = pytest.mark.skip(reason="Skipping slow tests in CI")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
