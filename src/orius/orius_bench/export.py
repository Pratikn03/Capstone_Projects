"""ORIUS-Bench export utilities.

Generates leaderboard CSV, JSON artefact bundles, schemas, and
reproducibility digests for Paper 4.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from orius.orius_bench.metrics_engine import BenchmarkMetrics


def leaderboard_row(
    controller_name: str,
    domain: str,
    seed: int,
    metrics: BenchmarkMetrics,
) -> dict[str, Any]:
    """Build a single leaderboard row dict."""
    return {
        "controller": controller_name,
        "domain": domain,
        "seed": seed,
        "tsvr": round(metrics.tsvr, 6),
        "oasg": round(metrics.oasg, 6),
        "cva": round(metrics.cva, 6),
        "gdq": round(metrics.gdq, 6),
        "intervention_rate": round(metrics.intervention_rate, 6),
        "audit_completeness": round(metrics.audit_completeness, 6),
        "recovery_latency": round(metrics.recovery_latency, 6),
        "n_steps": metrics.n_steps,
    }


_LEADERBOARD_COLUMNS = [
    "controller",
    "domain",
    "seed",
    "tsvr",
    "oasg",
    "cva",
    "gdq",
    "intervention_rate",
    "audit_completeness",
    "recovery_latency",
    "n_steps",
]


def write_leaderboard_csv(
    rows: Sequence[dict[str, Any]],
    path: str | Path,
) -> Path:
    """Write leaderboard CSV to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_LEADERBOARD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_bundle_json(
    rows: Sequence[dict[str, Any]],
    fault_digests: dict[int, str],
    path: str | Path,
    *,
    fault_schedules: dict[int, list[dict[str, Any]]] | None = None,
) -> Path:
    """Write full artefact bundle (results + fault digests + optional fault schedules) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle: dict[str, Any] = {
        "schema_version": "2.0.0",
        "fault_digests": {str(k): v for k, v in fault_digests.items()},
        "results": list(rows),
    }
    if fault_schedules is not None:
        bundle["fault_schedules"] = {str(k): v for k, v in fault_schedules.items()}
    with open(path, "w") as f:
        json.dump(bundle, f, indent=2)
    return path


def controller_contract_schema() -> dict[str, Any]:
    """Return JSON Schema for the ControllerAPI contract."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ORIUS-Bench Controller Contract",
        "type": "object",
        "required": ["name", "propose_action"],
        "properties": {
            "name": {"type": "string"},
            "propose_action": {
                "type": "object",
                "properties": {
                    "observed_state": {"type": "object"},
                    "uncertainty": {"type": ["object", "null"]},
                    "certificate_state": {"type": ["object", "null"]},
                },
                "required": ["observed_state"],
            },
        },
    }


def fault_schema() -> dict[str, Any]:
    """Return JSON Schema for FaultSchedule serialisation."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ORIUS-Bench Fault Schedule",
        "type": "object",
        "required": ["seed", "horizon", "events", "digest"],
        "properties": {
            "seed": {"type": "integer"},
            "horizon": {"type": "integer"},
            "digest": {"type": "string"},
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["step", "kind", "duration"],
                    "properties": {
                        "step": {"type": "integer"},
                        "kind": {"type": "string", "enum": ["bias", "blackout", "noise", "stuck_sensor"]},
                        "duration": {"type": "integer", "minimum": 1},
                        "params": {"type": "object"},
                    },
                },
            },
        },
    }


def metrics_schema() -> dict[str, Any]:
    """Return JSON Schema for BenchmarkMetrics."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ORIUS-Bench Metrics",
        "type": "object",
        "required": [
            "tsvr",
            "oasg",
            "cva",
            "gdq",
            "intervention_rate",
            "audit_completeness",
            "recovery_latency",
        ],
        "properties": {
            "tsvr": {"type": "number", "minimum": 0, "maximum": 1},
            "oasg": {"type": "number", "minimum": 0, "maximum": 1},
            "cva": {"type": "number", "minimum": 0, "maximum": 1},
            "gdq": {"type": "number", "minimum": 0, "maximum": 1},
            "intervention_rate": {"type": "number", "minimum": 0, "maximum": 1},
            "audit_completeness": {"type": "number", "minimum": 0, "maximum": 1},
            "recovery_latency": {"type": "number", "minimum": 0},
            "n_steps": {"type": "integer"},
        },
    }


def write_schemas(out_dir: str | Path) -> None:
    """Write controller_contract.json, fault_schema.json, metrics_schema.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, schema in [
        ("controller_contract.json", controller_contract_schema()),
        ("fault_schema.json", fault_schema()),
        ("metrics_schema.json", metrics_schema()),
    ]:
        with open(out_dir / name, "w") as f:
            json.dump(schema, f, indent=2)
