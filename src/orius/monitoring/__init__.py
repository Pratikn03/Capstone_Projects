"""Monitoring package.

Contains drift detection, retraining decisions, and alert/report helpers.
"""

from .report import write_monitoring_report

__all__ = ["write_monitoring_report"]
