"""Navigation domain adapter — canonical entrypoint.

Provides both:
  - ``NavigationDomainAdapter`` for runtime ORIUS/DC3S execution, and
  - ``NavigationTrackAdapter`` for ORIUS-Bench evaluation.
"""
from orius.orius_bench.navigation_track import NavigationTrackAdapter
from orius.universal_framework.navigation_adapter import NavigationDomainAdapter

__all__ = ["NavigationDomainAdapter", "NavigationTrackAdapter"]
