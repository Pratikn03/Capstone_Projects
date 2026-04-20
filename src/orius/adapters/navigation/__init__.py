"""Navigation domain adapter — compatibility entrypoint."""
from __future__ import annotations

from orius.universal_framework.navigation_adapter import NavigationDomainAdapter
from orius.orius_bench.navigation_track import NavigationTrackAdapter

__all__ = ["NavigationDomainAdapter", "NavigationTrackAdapter"]
