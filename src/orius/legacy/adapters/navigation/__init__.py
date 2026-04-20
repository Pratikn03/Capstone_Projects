"""Navigation domain adapter — canonical entrypoint.

Exports both the BenchmarkTrack and the full DomainAdapter proof implementation.

    from orius.adapters.navigation import NavigationDomainAdapter, NavigationTrackAdapter

Exports both:
  - NavigationDomainAdapter for the universal runtime path
  - NavigationTrackAdapter for ORIUS-Bench benchmark runs
"""
from orius.orius_bench.navigation_track import NavigationTrackAdapter
from orius.universal_framework.navigation_adapter import NavigationDomainAdapter

__all__ = ["NavigationTrackAdapter", "NavigationDomainAdapter"]
