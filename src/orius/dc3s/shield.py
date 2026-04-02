"""Generic safety shield that delegates action repair to a domain-specific adapter."""
from __future__ import annotations

from typing import Any, Mapping, Protocol


Action = Mapping[str, Any]
UncertaintySet = Mapping[str, Any]


class ProjectionAdapter(Protocol):
    """Compatibility surface for legacy projection-based shield adapters.

    The canonical runtime boundary is ``orius.dc3s.domain_adapter.DomainAdapter``.
    This protocol remains only so the battery-oriented shield path can keep
    working without importing the legacy adapter module into active runtime
    code.
    """

    def project_to_safe_set(
        self,
        candidate_action: Action,
        uncertainty_set: UncertaintySet | None,
        state: Any | None = None,
    ) -> tuple[Action, dict[str, Any]]:
        ...


def repair_action(
    a_star: Action,
    state: Any = None,
    uncertainty_set: UncertaintySet | None = None,
    constraints: Mapping[str, Any] | None = None,
    cfg: Mapping[str, Any] | None = None,
    *,
    domain_adapter: ProjectionAdapter | None = None,
) -> tuple[Action, dict[str, Any]]:
    """
    Repairs a potentially unsafe action.

    Supports two call patterns:

    1. Domain-adapter API (pipeline): repair_action(a_star, uncertainty_set=..., domain_adapter=..., state=...)
    2. Legacy battery API (runner, baselines): repair_action(a_star, state=..., uncertainty_set=..., constraints=..., cfg=...)
    """
    if domain_adapter is not None:
        safe_action, meta = domain_adapter.project_to_safe_set(
            candidate_action=a_star,
            uncertainty_set=uncertainty_set,
            state=state,
        )
        return safe_action, meta

    from orius.domain.battery_adapter import repair_battery_action

    return repair_battery_action(
        a_star=a_star,
        state=state or {},
        uncertainty_set=uncertainty_set or {},
        constraints=constraints or {},
        cfg=cfg or {},
    )
