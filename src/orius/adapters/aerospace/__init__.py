"""Aerospace domain adapter — canonical entrypoint.

Re-exports from current implementations. New code should import from here:

    from orius.adapters.aerospace import AerospaceDomainAdapter, AerospaceTrackAdapter
"""
from __future__ import annotations

from orius.universal_framework.aerospace_adapter import AerospaceDomainAdapter
from orius.orius_bench.aerospace_track import AerospaceTrackAdapter

__all__ = ["AerospaceDomainAdapter", "AerospaceTrackAdapter"]
