"""Industrial domain adapter — compatibility entrypoint."""
from __future__ import annotations

from orius.universal_framework.industrial_adapter import IndustrialDomainAdapter
from orius.orius_bench.industrial_track import IndustrialTrackAdapter

__all__ = ["IndustrialDomainAdapter", "IndustrialTrackAdapter"]
