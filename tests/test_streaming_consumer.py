import sys
from unittest.mock import MagicMock
sys.modules['kafka'] = MagicMock()

import pytest
from orius.streaming.consumer import (
    StreamingIngestConsumer,
    AppConfig,
    ConsumerConfig,
    StorageConfig,
    ValidationConfig,
)

@pytest.fixture
def mock_app_config(tmp_path):
    return AppConfig(
        kafka=ConsumerConfig(bootstrap_servers="localhost:9092", topic="test", group_id="test"),
        storage=StorageConfig(mode="duckdb", duckdb_path=":memory:", table_name="test_table", parquet_dir=str(tmp_path)),
        checkpoint_path=str(tmp_path / "ckpt.json"),
        validation=ValidationConfig(strict=True, cadence_seconds=3600, min_mw=0.0, max_mw=10000.0),
    )

def test_consumer_initialization(mock_app_config):
    consumer = StreamingIngestConsumer(mock_app_config)
    assert consumer.con is not None
    
def test_consumer_validation_success(mock_app_config):
    consumer = StreamingIngestConsumer(mock_app_config)
    evt = {
        "utc_timestamp": "2023-01-01T12:00:00Z",
        "DE_load_actual_entsoe_transparency": 5000.0,
        "DE_wind_generation_actual": 1000.0,
        "DE_solar_generation_actual": 500.0,
        "device_id": "d1",
        "zone_id": "DE"
    }
    validated = consumer._validate(evt)
    assert validated is not None
    assert validated["device_id"] == "d1"

def test_consumer_validation_bounds_failure(mock_app_config):
    consumer = StreamingIngestConsumer(mock_app_config)
    evt = {
        "utc_timestamp": "2023-01-01T12:00:00Z",
        "DE_load_actual_entsoe_transparency": 50000.0, 
        "device_id": "d1",
        "zone_id": "DE"
    }
    with pytest.raises(ValueError, match="out of bounds"):
        consumer._validate(evt)

def test_consumer_validation_cadence_failure(mock_app_config):
    consumer = StreamingIngestConsumer(mock_app_config)
    evt1 = {
        "utc_timestamp": "2023-01-01T12:00:00Z",
        "DE_load_actual_entsoe_transparency": 5000.0,
        "device_id": "d1",
        "zone_id": "DE"
    }
    consumer._validate(evt1)
    evt2 = {
        "utc_timestamp": "2023-01-01T12:01:00Z",
        "DE_load_actual_entsoe_transparency": 5000.0,
        "device_id": "d1",
        "zone_id": "DE"
    }
    with pytest.raises(ValueError, match="Cadence violation"):
        consumer._validate(evt2)
