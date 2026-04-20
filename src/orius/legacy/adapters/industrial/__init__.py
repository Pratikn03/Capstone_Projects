"""Industrial domain adapter — canonical entrypoint.

Re-exports from current implementations. New code should import from here:

    from orius.adapters.industrial import IndustrialDomainAdapter, IndustrialTrackAdapter
"""
from __future__ import annotations

from orius.universal_framework.industrial_adapter import IndustrialDomainAdapter
from orius.orius_bench.industrial_track import IndustrialTrackAdapter

__all__ = ["IndustrialDomainAdapter", "IndustrialTrackAdapter"]
