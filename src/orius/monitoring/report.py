"""Monitoring report writer helpers."""
from __future__ import annotations

import json
from pathlib import Path


def write_monitoring_report(payload: dict, out_path: str = "reports/monitoring_report.md") -> None:
    """Write a simple markdown monitoring report with embedded JSON payload."""
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# Monitoring Report\n\n```json\n" + json.dumps(payload, indent=2) + "\n```\n",
        encoding="utf-8",
    )
