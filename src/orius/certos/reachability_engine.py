"""CertOS reachability engine: validity horizon from reachability."""
from __future__ import annotations

from typing import Any, Mapping


def compute_validity_horizon(
    interval_lower_mwh: float,
    interval_upper_mwh: float,
    safe_action: Mapping[str, Any],
    constraints: Mapping[str, Any],
    sigma_d: float,
    max_steps: int = 4096,
) -> dict[str, float | int]:
    """Delegate to orius.dc3s.reachability."""
    from orius.dc3s.reachability import compute_validity_horizon_from_reachability
    return compute_validity_horizon_from_reachability(
        interval_lower_mwh=interval_lower_mwh,
        interval_upper_mwh=interval_upper_mwh,
        safe_action=safe_action,
        constraints=constraints,
        sigma_d=sigma_d,
        max_steps=max_steps,
    )
