"""Navigation domain adapter — compatibility entrypoint."""

from __future__ import annotations

from orius.orius_bench.navigation_track import NavigationTrackAdapter
from orius.universal_framework.navigation_adapter import NavigationDomainAdapter

__all__ = ["NavigationDomainAdapter", "NavigationTrackAdapter"]
