"""CertOS safe action filter: shield projection."""
from __future__ import annotations

from typing import Any, Mapping


def filter_action(
    a_star: Mapping[str, Any],
    state: Mapping[str, Any],
    uncertainty_set: Mapping[str, Any],
    constraints: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[Mapping[str, float], Mapping[str, Any]]:
    """Delegate to orius.dc3s.shield.repair_action."""
    from orius.dc3s.shield import repair_action
    return repair_action(
        a_star=a_star,
        state=state,
        uncertainty_set=uncertainty_set,
        constraints=constraints,
        cfg=cfg,
    )
