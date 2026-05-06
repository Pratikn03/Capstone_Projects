"""Aerospace domain adapter — compatibility entrypoint."""

from __future__ import annotations

from orius.orius_bench.aerospace_track import AerospaceTrackAdapter
from orius.universal_framework.aerospace_adapter import AerospaceDomainAdapter

__all__ = ["AerospaceDomainAdapter", "AerospaceTrackAdapter"]
