"""CertOS shift engine: uncertainty set inflation."""
from __future__ import annotations

from typing import Any, Mapping


def build_uncertainty_set(
    yhat: float | list[float],
    q: float | list[float],
    w_t: float,
    drift_flag: bool = False,
    cfg: Mapping[str, Any] | None = None,
    prev_inflation: float | None = None,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Delegate to orius.dc3s.calibration.build_uncertainty_set."""
    from orius.dc3s.calibration import build_uncertainty_set as _build
    import numpy as np
    lower, upper, meta = _build(
        yhat, q, w_t, drift_flag, cfg or {}, prev_inflation
    )
    return (
        np.asarray(lower).tolist(),
        np.asarray(upper).tolist(),
        meta,
    )
