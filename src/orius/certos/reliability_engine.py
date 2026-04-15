"""CertOS reliability engine: OQE / quality scoring."""
from __future__ import annotations

from typing import Any, Mapping


def compute_reliability(
    event: Mapping[str, Any],
    last_event: Mapping[str, Any] | None,
    **kwargs: Any,
) -> tuple[float, Mapping[str, Any]]:
    """CertOS-facing reliability wrapper with basic contract checks."""
    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping")
    if last_event is not None and not isinstance(last_event, Mapping):
        raise TypeError("last_event must be a mapping or None")
    if "expected_cadence_s" in kwargs and float(kwargs["expected_cadence_s"]) <= 0.0:
        raise ValueError("expected_cadence_s must be positive")
    from orius.dc3s.quality import compute_reliability as _compute
    weight, flags = _compute(dict(event), None if last_event is None else dict(last_event), **kwargs)
    return float(weight), {"engine": "certos.reliability", **dict(flags)}
