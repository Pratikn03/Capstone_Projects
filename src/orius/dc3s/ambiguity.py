from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AmbiguityConfig:
    lambda_mw: float = 0.0
    min_w: float = 0.05
    max_extra: float = 1.0


def widen_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    w_t: float,
    cfg: AmbiguityConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.size != hi.size:
        raise ValueError("lower and upper must have the same length")
    if np.any(lo > hi):
        raise ValueError("lower cannot exceed upper")

    w_eff = max(float(cfg.min_w), min(float(w_t), 1.0))
    extra = min(float(cfg.max_extra), max(0.0, 1.0 - w_eff))
    delta = float(cfg.lambda_mw) * extra

    lo2 = lo - delta
    hi2 = hi + delta
    return lo2, hi2, {"w_t": float(w_t), "w_used": float(w_eff), "delta_mw": float(delta)}
