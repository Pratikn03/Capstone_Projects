"""Monitoring: model performance drift checks."""
from __future__ import annotations

def metric_drift(current_metric: float, baseline_metric: float, degradation_threshold: float = 0.15):
    """Check whether current metric degraded vs baseline by a threshold."""
    if baseline_metric <= 0:
        return {"drift": False, "ratio": None}
    ratio = (current_metric - baseline_metric) / baseline_metric
    return {"drift": ratio >= degradation_threshold, "ratio": float(ratio)}
