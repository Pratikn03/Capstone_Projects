from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class NGBoostConfig:
    n_estimators: int = 600
    learning_rate: float = 0.03
    minibatch_frac: float = 1.0
    random_state: int = 42
    verbose: bool = False


def _require_ngboost():
    try:
        from ngboost import NGBRegressor
        from ngboost.distns import Normal
        from ngboost.scores import LogScore
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "NGBoost is required for distributional forecasting. Install ngboost in the active environment."
        ) from exc
    return NGBRegressor, Normal, LogScore


def train_ngboost_distribution(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: NGBoostConfig | None = None,
):
    config = cfg or NGBoostConfig()
    x = np.asarray(x_train, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("x_train must be a 2D array")
    if len(x) != len(y):
        raise ValueError("x_train and y_train must have matching length")

    NGBRegressor, Normal, LogScore = _require_ngboost()
    model = NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        n_estimators=int(config.n_estimators),
        learning_rate=float(config.learning_rate),
        minibatch_frac=float(config.minibatch_frac),
        verbose=bool(config.verbose),
        random_state=int(config.random_state),
    )
    model.fit(x, y)
    return model


def predict_ngboost_quantiles(
    model,
    x: np.ndarray,
    quantiles: Iterable[float] = (0.1, 0.5, 0.9),
) -> dict[float, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x must be a 2D array")
    q_list = [float(q) for q in quantiles]
    for q in q_list:
        if not (0.0 < q < 1.0):
            raise ValueError("quantiles must be in (0, 1)")

    pred_dist = model.pred_dist(x_arr)
    out: dict[float, np.ndarray] = {}
    for q in q_list:
        vals = np.asarray(pred_dist.ppf(q), dtype=float).reshape(-1)
        out[q] = vals
    return out


def summarize_interval_quality(
    y_true: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    lo = np.asarray(q_lo, dtype=float).reshape(-1)
    hi = np.asarray(q_hi, dtype=float).reshape(-1)
    if not (len(y) == len(lo) == len(hi)):
        raise ValueError("y_true, q_lo, q_hi must have same length")
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    return {
        "picp_90": float(np.mean(covered)) if len(covered) else 0.0,
        "mean_width": float(np.mean(width)) if len(width) else 0.0,
        "width_p95": float(np.quantile(width, 0.95)) if len(width) else 0.0,
    }
