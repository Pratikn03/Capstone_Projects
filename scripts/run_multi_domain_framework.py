#!/usr/bin/env python3
"""Run ORIUS Universal Framework across the registered runtime domains.

Current default surface: battery, AV, navigation, surgical robotics,
aerospace, and industrial.

Each domain runs one step of run_universal_step with synthetic telemetry.
Outputs a JSON report with certificates and safe actions per domain.

Usage:
    python scripts/run_multi_domain_framework.py [--out reports/multi_domain]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from orius.universal_framework import run_universal_step, get_adapter, list_domains


def _json_serializable(obj: object) -> object:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(x) for x in obj]
    return obj


# Runtime domains exercised with synthetic telemetry.
DOMAIN_CONFIGS = [
    {
        "id": "energy",
        "name": "Battery Energy Storage",
        "telemetry": {
            "load_mw": 45.0,
            "renewables_mw": 80.0,
            "current_soc_mwh": 100.0,
            "capacity_mwh": 200.0,
            "yhat_load": 48.0,
            "ts_utc": "2026-03-16T12:00:00Z",
        },
        "candidate": {"charge_mw": 20.0, "discharge_mw": 0.0},
        "constraints": {
            "min_soc_mwh": 20.0,
            "max_soc_mwh": 180.0,
            "capacity_mwh": 200.0,
            "max_power_mw": 100.0,
        },
    },
    {
        "id": "av",
        "name": "Autonomous Vehicles",
        "telemetry": {
            "position_m": 100.0,
            "speed_mps": 8.0,
            "speed_limit_mps": 15.0,
            "lead_position_m": 150.0,
            "ts_utc": "2026-03-16T12:00:00Z",
        },
        "candidate": {"acceleration_mps2": 0.5},
        "constraints": {"speed_max_mps": 15.0},
    },
    {
        "id": "navigation",
        "name": "Navigation",
        "telemetry": {
            "x": 9.95,
            "y": 9.80,
            "vx": 0.0,
            "vy": 0.0,
            "ts_utc": "2026-03-16T12:00:00Z",
        },
        "candidate": {"ax": 4.0, "ay": 4.0},
        "constraints": {
            "arena_min": 0.0,
            "arena_max": 10.0,
            "max_speed": 1.0,
            "dt_s": 0.25,
        },
    },
    {
        "id": "surgical_robotics",
        "name": "Surgical Robotics (Vital Signs)",
        "telemetry": {
            "hr_bpm": 72.0,
            "spo2_pct": 97.0,
            "respiratory_rate": 14.0,
            "ts_utc": "2026-03-16T12:00:00Z",
        },
        "candidate": {"alert_level": 0.2},
        "constraints": {"spo2_min_pct": 90.0},
    },
    {
        "id": "aerospace",
        "name": "Aerospace",
        "telemetry": {
            "altitude_m": 3000.0,
            "airspeed_kt": 180.0,
            "bank_angle_deg": 5.0,
            "fuel_remaining_pct": 65.0,
            "ts_utc": "2026-03-16T12:00:00Z",
        },
        "candidate": {"throttle": 0.7, "bank_deg": 3.0},
        "constraints": {"v_min_kt": 60.0, "v_max_kt": 350.0},
    },
    {
        "id": "industrial",
        "name": "Industrial Process Control",
        "telemetry": {
            "temp_c": 85.0,
            "pressure_mbar": 1010.0,
            "power_mw": 450.0,
            "ts_utc": "2026-03-16T12:00:00Z",
        },
        "candidate": {"power_setpoint_mw": 480.0},
        "constraints": {"power_max_mw": 500.0, "temp_max_c": 120.0},
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ORIUS Universal Framework across the registered runtime domains"
    )
    parser.add_argument("--out", default="reports/multi_domain", help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    available = list_domains()

    for cfg in DOMAIN_CONFIGS:
        domain_id = cfg["id"]
        if domain_id not in available:
            results.append({
                "domain_id": domain_id,
                "domain_name": cfg["name"],
                "status": "skipped",
                "reason": f"Domain not registered: {domain_id}",
            })
            continue

        try:
            adapter = get_adapter(domain_id, {})
            result = run_universal_step(
                domain_adapter=adapter,
                raw_telemetry=cfg["telemetry"],
                history=None,
                candidate_action=cfg["candidate"],
                constraints=cfg["constraints"],
                quantile=50.0,
            )
            safe_action = result.get("safe_action")
            results.append({
                "domain_id": domain_id,
                "domain_name": cfg["name"],
                "status": "ok",
                "safe_action": _json_serializable(safe_action) if safe_action else {},
                "reliability_w": float(result.get("reliability_w", 0.0)),
                "drift_flag": result.get("drift_flag", False),
                "certificate_keys": list(result.get("certificate", {}).keys()),
            })
        except Exception as e:
            results.append({
                "domain_id": domain_id,
                "domain_name": cfg["name"],
                "status": "error",
                "error": str(e),
            })

    report_path = out / "multi_domain_report.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "domains_run": len([r for r in results if r.get("status") == "ok"]),
                "domains_total": len(DOMAIN_CONFIGS),
                "available_domains": available,
                "results": results,
            },
            f,
            indent=2,
        )

    print("=== ORIUS Multi-Domain Framework (Thesis Ch 18) ===")
    print(f"  Available domains: {available}")
    for r in results:
        status = r.get("status", "?")
        name = r.get("domain_name", r.get("domain_id", "?"))
        if status == "ok":
            safe = r.get("safe_action", {})
            w = r.get("reliability_w", 0)
            print(f"  ✓ {name}: reliability={w:.3f} safe_action={list(safe.keys())}")
        else:
            print(f"  ✗ {name}: {status} {r.get('reason', r.get('error', ''))}")
    print(f"\n  Report → {report_path}")


if __name__ == "__main__":
    main()
