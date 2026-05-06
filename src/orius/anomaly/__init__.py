"""Anomaly detection:   init  ."""

from .detection import MultivariateAnomalyDetector as MultivariateAnomalyDetector

__all__ = ["MultivariateAnomalyDetector"]
# Key: flag anomalies from residuals or isolation forest signals
