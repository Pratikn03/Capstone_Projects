# GridPulse Tests

Pytest test suite for the GridPulse forecasting platform.

## Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_features.py` | Feature engineering pipeline |
| `test_splits.py` | Time series train/val/test splitting |
| `test_predict.py` | Forecast model predictions |
| `test_conformal.py` | Conformal prediction intervals |
| `test_anomaly.py` | Anomaly detection logic |
| `test_optimizer.py` | Battery dispatch optimization |
| `test_optimizer_constraints.py` | SoC and power constraints |
| `test_api_health.py` | API health endpoints |
| `test_api_intervals.py` | Forecast interval endpoints |
| `test_api_security.py` | API authentication |
| `test_streaming_contract.py` | Kafka message schemas |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/gridpulse --cov-report=html

# Run specific test file
pytest tests/test_optimizer.py -v

# Run tests matching pattern
pytest tests/ -k "anomaly" -v
```

## Test Configuration

See `pytest.ini` for test configuration:
- Test discovery patterns
- Markers definition
- Coverage settings

## Fixtures

Common test fixtures are defined in `conftest.py` (if present) or inline within test files.

## CI Integration

Tests run automatically via GitHub Actions on:
- Push to main/develop branches
- Pull request creation
- Scheduled nightly runs
