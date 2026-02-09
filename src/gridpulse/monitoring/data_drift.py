"""
Monitoring: Data drift detection using statistical tests.

This module provides the Kolmogorov-Smirnov test for detecting distributional
drift between reference (training) and current (production) data samples.
Drift detection is critical for triggering model retraining.

Typical usage:
    >>> from gridpulse.monitoring.data_drift import ks_drift
    >>> result = ks_drift(train_feature, prod_feature)
    >>> if result['drift']:
    ...     trigger_retraining()
"""
from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp


def ks_drift(reference, current, p_value_threshold: float = 0.01):
    """Kolmogorov-Smirnov drift test between reference and current samples.
    
    The KS test measures the maximum distance between cumulative distribution
    functions. A small p-value indicates the distributions are different.
    
    Args:
        reference: Reference data sample (e.g., training distribution)
        current: Current data sample (e.g., production data window)
        p_value_threshold: Significance level for drift detection (default: 0.01)
    
    Returns:
        Dictionary with:
        - ks_stat: KS test statistic (max CDF distance)
        - p_value: Two-sided p-value
        - drift: Boolean indicating significant drift detected
    """
    ref = np.asarray(reference)
    cur = np.asarray(current)
    stat, p = ks_2samp(ref, cur)
    return {"ks_stat": float(stat), "p_value": float(p), "drift": bool(p < p_value_threshold)}
