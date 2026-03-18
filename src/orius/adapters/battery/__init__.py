"""Battery domain adapter — canonical entrypoint.

Re-exports from current implementations. New code should import from here:

    from orius.adapters.battery import BatteryDomainAdapter, BatteryTrackAdapter
"""
from __future__ import annotations

from orius.dc3s.battery_adapter import BatteryDomainAdapter
from orius.orius_bench.battery_track import BatteryTrackAdapter

__all__ = ["BatteryDomainAdapter", "BatteryTrackAdapter"]
