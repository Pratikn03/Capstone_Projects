"""
ORIUS Utilities Package.

Common utility modules for configuration, logging, seeding, and feature scaling.
These utilities are shared across all ORIUS components (training, serving, etc.)

Key modules:
    - config: YAML configuration loading and Pydantic validation models
    - logging: Structured logging setup with configurable handlers  
    - manifest: Build manifest generation for reproducibility tracking
    - metrics: Prometheus metric definitions for observability
    - net: Network utilities for API health checks
    - registry: Function and model registry decorators
    - scaler: Feature scaling with persistence (StandardScaler)
    - seed: Random seed management for reproducibility
    - time: Time-based feature engineering helpers

Example usage:
    >>> from orius.utils.seed import set_seed
    >>> from orius.utils.config import validate_config
    >>> 
    >>> set_seed(42)
    >>> validate_config(Path("configs/forecast.yaml"))
"""

# Re-export commonly used utilities for convenient imports
from orius.utils.seed import set_seed

__all__ = [
    "set_seed",
]

