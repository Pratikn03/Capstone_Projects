"""
GridPulse: ML-Based Energy Forecasting and Battery Dispatch Optimization.

This package provides an end-to-end machine learning system for:
- Multi-horizon forecasting of load, wind, and solar generation
- Uncertainty quantification via conformal prediction
- Carbon-aware battery dispatch optimization
- Real-time anomaly detection and drift monitoring

Main Modules:
    - forecasting: GBM, LSTM, TCN models with Optuna tuning
    - optimizer: MILP-based dispatch solver with risk constraints
    - anomaly: Isolation Forest + residual z-score detection
    - monitoring: KS-test drift detection and alerting
    - data_pipeline: Feature engineering for OPSD/EIA-930 data
    - evaluation: Statistical testing and ablation studies

Example:
    >>> from gridpulse.forecasting.train import main as train_models
    >>> from gridpulse.optimizer.lp_dispatch import optimize_dispatch

Author: GridPulse Team
Version: 0.1.0
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
