"""CertOS shift engine: uncertainty set inflation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_uncertainty_set(
    yhat: float | list[float],
    q: float | list[float],
    w_t: float,
    drift_flag: bool = False,
    cfg: Mapping[str, Any] | None = None,
    prev_inflation: float | None = None,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """CertOS-facing uncertainty wrapper with normalized outputs."""
    if float(w_t) < 0.0:
        raise ValueError("w_t must be non-negative")
    import numpy as np

    from orius.dc3s.calibration import build_uncertainty_set as _build

    lower, upper, meta = _build(yhat, q, w_t, drift_flag, cfg or {}, prev_inflation)
    return (
        np.asarray(lower).tolist(),
        np.asarray(upper).tolist(),
        {"engine": "certos.shift", **dict(meta)},
    )
