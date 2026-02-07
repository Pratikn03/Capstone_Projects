"""Risk-aware helpers for dispatch optimization inputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class RiskConfig:
    enabled: bool = True
    mode: str = "worst_case_interval"
    load_bound: str = "upper"
    renew_bound: str = "lower"
    reserve_soc_mwh: float = 0.0


def apply_interval_bounds(
    load_point: np.ndarray,
    renew_point: np.ndarray,
    load_lo: Optional[np.ndarray] = None,
    load_hi: Optional[np.ndarray] = None,
    renew_lo: Optional[np.ndarray] = None,
    renew_hi: Optional[np.ndarray] = None,
    cfg: Optional[RiskConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return conservative load/renew arrays based on configured bounds."""
    cfg = cfg or RiskConfig()
    load_used = np.asarray(load_point, dtype=float).copy()
    renew_used = np.asarray(renew_point, dtype=float).copy()

    if not cfg.enabled:
        return load_used, renew_used

    if cfg.load_bound == "upper" and load_hi is not None:
        load_used = np.asarray(load_hi, dtype=float)
    elif cfg.load_bound == "lower" and load_lo is not None:
        load_used = np.asarray(load_lo, dtype=float)

    if cfg.renew_bound == "lower" and renew_lo is not None:
        renew_used = np.asarray(renew_lo, dtype=float)
    elif cfg.renew_bound == "upper" and renew_hi is not None:
        renew_used = np.asarray(renew_hi, dtype=float)

    return load_used, renew_used
