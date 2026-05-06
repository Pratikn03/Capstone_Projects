"""Controller and metric helpers for AV runtime replay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from orius.av_waymo.runtime import (
    _runtime_aligned_longitudinal_controller,
)
from orius.av_waymo.runtime import (
    deterministic_longitudinal_controller as _deterministic_longitudinal_controller,
)

deterministic_longitudinal_controller = _deterministic_longitudinal_controller


def runtime_aligned_longitudinal_controller(
    state: dict[str, Any],
    *,
    constraints: Mapping[str, Any],
    policy_config: Mapping[str, Any],
) -> dict[str, float]:
    """Controller wrapper used by the source-neutral AV runtime surface."""

    return cast(
        dict[str, float],
        _runtime_aligned_longitudinal_controller(
            state,
            constraints=constraints,
            policy_config=policy_config,
        ),
    )
