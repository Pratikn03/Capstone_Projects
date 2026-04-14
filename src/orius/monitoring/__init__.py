"""Monitoring package.

Contains drift detection, retraining decisions, and alert/report helpers.
"""

from .report import write_monitoring_report
from .shift_validity import ResidualStreamMonitor

__all__ = ["write_monitoring_report", "ResidualStreamMonitor"]
