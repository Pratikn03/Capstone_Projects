"""Monitoring: report."""
from __future__ import annotations
from pathlib import Path
import json

def write_monitoring_report(payload: dict, out_path: str = "reports/monitoring_report.md"):
    # Key: compute drift metrics and retraining signals
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        "# Monitoring Report\n\n```json\n" + json.dumps(payload, indent=2) + "\n```\n",
        encoding="utf-8",
    )
