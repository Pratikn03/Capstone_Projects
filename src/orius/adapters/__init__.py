"""ORIUS canonical adapter layer.

All domain adapters are re-exported from their canonical locations.
New code should import from orius.adapters.* for a single source of truth.

Canonical paths:
  - orius.adapters.battery
  - orius.adapters.vehicle
  - orius.adapters.navigation
  - orius.adapters.industrial
  - orius.adapters.aerospace
  - orius.adapters.healthcare
"""

from orius.adapters.battery import BatteryDomainAdapter
from orius.adapters.battery import BatteryTrackAdapter
from orius.adapters.vehicle import VehicleDomainAdapter
from orius.adapters.vehicle import VehicleTrackAdapter
from orius.adapters.industrial import IndustrialDomainAdapter
from orius.adapters.industrial import IndustrialTrackAdapter
from orius.adapters.healthcare import HealthcareDomainAdapter
from orius.adapters.healthcare import HealthcareTrackAdapter
from orius.adapters.aerospace import AerospaceDomainAdapter
from orius.adapters.aerospace import AerospaceTrackAdapter
from orius.adapters.navigation import NavigationDomainAdapter
from orius.adapters.navigation import NavigationTrackAdapter
from orius.adapters.navigation import NavigationDomainAdapter

__all__ = [
    "BatteryDomainAdapter",
    "BatteryTrackAdapter",
    "VehicleDomainAdapter",
    "VehicleTrackAdapter",
    "IndustrialDomainAdapter",
    "IndustrialTrackAdapter",
    "HealthcareDomainAdapter",
    "HealthcareTrackAdapter",
    "AerospaceDomainAdapter",
    "AerospaceTrackAdapter",
    "NavigationDomainAdapter",
    "NavigationTrackAdapter",
    "NavigationDomainAdapter",
]
