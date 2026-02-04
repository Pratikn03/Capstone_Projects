"""Monitoring: data drift tests."""
from __future__ import annotations
import numpy as np
from scipy.stats import ks_2samp

def ks_drift(reference, current, p_value_threshold: float = 0.01):
    """Kolmogorov–Smirnov drift test between reference and current samples."""
    ref = np.asarray(reference)
    cur = np.asarray(current)
    stat, p = ks_2samp(ref, cur)
    return {"ks_stat": float(stat), "p_value": float(p), "drift": bool(p < p_value_threshold)}
