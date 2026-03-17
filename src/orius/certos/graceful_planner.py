"""CertOS graceful planner: fallback trajectory."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def plan_fallback(
    last_action: Mapping[str, Any],
    horizon_steps: int,
    soc_mwh: float,
    constraints: Mapping[str, Any],
    sigma_d: float = 50.0,
) -> Sequence[dict[str, float]]:
    """Delegate to orius.dc3s.graceful.optimized_graceful."""
    from orius.dc3s.graceful import optimized_graceful
    return optimized_graceful(
        last_action=last_action,
        horizon_steps=horizon_steps,
        soc_mwh=soc_mwh,
        constraints=constraints,
        sigma_d=sigma_d,
    )
