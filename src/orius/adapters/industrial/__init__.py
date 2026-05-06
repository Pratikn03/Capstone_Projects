"""Industrial domain adapter — compatibility entrypoint."""

from __future__ import annotations

from orius.orius_bench.industrial_track import IndustrialTrackAdapter
from orius.universal_framework.industrial_adapter import IndustrialDomainAdapter

__all__ = ["IndustrialDomainAdapter", "IndustrialTrackAdapter"]
