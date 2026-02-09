# GridPulse Utilities

Common utility modules shared across the GridPulse forecasting system.

## Modules

| Module | Purpose |
|--------|---------|
| `config.py` | YAML configuration loading with environment variable expansion |
| `logging.py` | Structured logging setup using Python's logging module |
| `manifest.py` | Build manifest generation for reproducibility tracking |
| `metrics.py` | Prometheus metric definitions for monitoring |
| `net.py` | Network utilities for API health checks |
| `registry.py` | Function/model registry decorators |
| `scaler.py` | Feature scaling wrappers (StandardScaler persistence) |
| `seed.py` | Random seed management for reproducibility |
| `time.py` | Time-based feature engineering helpers |

## Usage

```python
from gridpulse.utils.config import load_config
from gridpulse.utils.seed import set_seed
from gridpulse.utils.scaler import fit_scaler, transform_features

# Load configuration
cfg = load_config("configs/forecast.yaml")

# Set reproducibility seed
set_seed(42)

# Scale features
scaler = fit_scaler(X_train)
X_scaled = transform_features(X_test, scaler)
```

## Design Principles

- **Single Responsibility**: Each module handles one concern
- **Minimal Dependencies**: Core utilities avoid heavy ML imports
- **Type Hints**: All public functions are fully typed
- **Testable**: Stateless functions where possible
