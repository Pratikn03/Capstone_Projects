"""Healthcare domain adapter — canonical entrypoint.

Re-exports from current implementations. New code should import from here:

    from orius.adapters.healthcare import HealthcareDomainAdapter, HealthcareTrackAdapter
"""

from __future__ import annotations

from orius.orius_bench.healthcare_track import HealthcareTrackAdapter
from orius.universal_framework.healthcare_adapter import HealthcareDomainAdapter

__all__ = ["HealthcareDomainAdapter", "HealthcareTrackAdapter"]
