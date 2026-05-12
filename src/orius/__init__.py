"""ORIUS -- Observation-Reality Integrity for Universal Safety.

ORIUS is a typed runtime safety layer that wraps any upstream controller and
closes the Observation--Action Safety Gap through a five-stage kernel
(Detect, Calibrate, Constrain, Shield, Certify; collectively DC3S).

Public API (stable across patch versions)
-----------------------------------------
The following submodules form the public surface that downstream code may
import.  They follow semantic versioning within a 0.x release line:

    orius.adapters          -- canonical typed adapter entry points
    orius.dc3s              -- DC3S kernel, contracts, calibration, drift
    orius.certos            -- certificate runtime + chain verification
    orius.universal_theory  -- T1--T11 theorem-linked runtime helpers

Defended-evidence rows (paper rows whose runtime, calibration, and governance
artifacts back the IEEE manuscript):

    orius.dc3s.battery_adapter           -- battery witness row (full closure)
    orius.av_waymo                       -- AV bounded row (replay surface)
    orius.universal_framework.healthcare_adapter
                                         -- healthcare bounded row

Experimental / unstable (subject to change without deprecation):

    orius.forecasting.*                  -- deep-learning forecaster lab
    orius.multi_agent                    -- multi-agent constraint composition
    orius.orius_bench                    -- benchmark export layer

Not part of the defended evidence package (preserved for backward
compatibility, not exercised by ``make paper-verify``):

    orius.legacy.aerospace_adapter
    orius.legacy.industrial_adapter
    orius.legacy.navigation_adapter
    orius.universal_framework.{aerospace,industrial,navigation}_adapter
                                         -- thin compat re-exports of the above

Import discipline
-----------------
New code should import from ``orius.adapters.*`` and ``orius.dc3s.*``.
Direct imports of legacy modules trigger no warnings, but their inclusion in
publication-facing artifacts is not endorsed; see ``src/orius/legacy/__init__.py``.

See ``src/orius/PUBLIC_API.md`` for the full stability matrix.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
