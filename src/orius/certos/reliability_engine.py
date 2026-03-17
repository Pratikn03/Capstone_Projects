"""CertOS reliability engine: OQE / quality scoring."""
from __future__ import annotations

from typing import Any, Mapping


def compute_reliability(
    event: Mapping[str, Any],
    last_event: Mapping[str, Any] | None,
    **kwargs: Any,
) -> tuple[float, Mapping[str, Any]]:
    """Delegate to orius.dc3s.quality.compute_reliability."""
    from orius.dc3s.quality import compute_reliability as _compute
    return _compute(event, last_event, **kwargs)
