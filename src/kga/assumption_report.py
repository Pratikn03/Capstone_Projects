"""Stable import path for the assumption-report schema.

The definitions live in :mod:`kga.assumptions` alongside the diagnostics that
populate them; this module only re-exports them, so reporting tools can depend on a
name that will not move if the diagnostics are reorganised.

Importing this pulls in numpy/scipy via :mod:`kga.assumptions`.  Tools that only
need to *read* emitted reports should read the JSON directly -- the schema is fixed
and documented in the paper (Table: assumption-report schema).
"""

from __future__ import annotations

from kga.assumptions import (  # noqa: F401
    AssumptionReport,
    CoverageType,
    FallbackAction,
    GateDecision,
    GateThresholds,
    ProtocolRecord,
    Status,
    run_gate,
    write_report,
)

__all__ = [
    "AssumptionReport",
    "CoverageType",
    "FallbackAction",
    "GateDecision",
    "GateThresholds",
    "ProtocolRecord",
    "Status",
    "run_gate",
    "write_report",
]
