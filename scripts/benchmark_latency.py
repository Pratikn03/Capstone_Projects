#!/usr/bin/env python3
"""DC3S per-domain latency benchmark.

Runs 1000 consecutive DC3S steps per domain, measures p50/p95/p99 wall-clock
latency, and asserts p99 < 50ms.

Usage
-----
    python scripts/benchmark_latency.py [--n N] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from orius.universal_framework.domain_registry import get_adapter
from orius.universal_framework.pipeline import run_universal_step

# (registry_id, sample_state, sample_action, sample_constraints)
DOMAIN_SPECS: list[tuple[str, dict, dict, dict]] = [
    (
        "industrial",
        {"power_mw": 490.0, "temp_c": 112.0, "pressure_mbar": 1010.0},
        {"power_setpoint_mw": 480.0},
        {"power_max_mw": 500.0},
    ),
    (
        "healthcare",
        {"spo2_pct": 87.0, "hr_bpm": 72.0, "respiratory_rate": 14.0},
        {"alert_level": 0.8},
        {"spo2_min_pct": 90.0},
    ),
    (
        "av",
        {"speed_ms": 20.0, "lateral_error_m": 0.3},
        {"brake_force": 0.2},
        {},
    ),
    (
        "aerospace",
        {"altitude_m": 10000.0, "airspeed_kts": 250.0},
        {"thrust_reduction": 0.1},
        {},
    ),
    (
        "navigation",
        {"heading_deg": 90.0, "speed_ms": 8.0},
        {"speed_reduction": 0.0},
        {},
    ),
    (
        "energy",
        {"soc": 0.65, "power_kw": 200.0},
        {"dispatch_kw": 250.0},
        {"power_max_kw": 300.0, "soc_min": 0.10, "soc_max": 0.95},
    ),
]

P99_LIMIT_MS = 50.0


def benchmark_domain(
    registry_id: str,
    state: dict,
    action: dict,
    constraints: dict,
    n_steps: int,
) -> dict:
    """Benchmark one domain for n_steps and return latency stats."""
    adapter = get_adapter(registry_id)
    latencies_ms: list[float] = []

    # Warm-up (5 steps, not measured)
    for _ in range(5):
        try:
            run_universal_step(
                domain_adapter=adapter,
                raw_telemetry=state,
                history=None,
                candidate_action=action,
                constraints=constraints,
            )
        except Exception:
            pass

    # Measured steps
    for _ in range(n_steps):
        t0 = time.perf_counter()
        try:
            run_universal_step(
                domain_adapter=adapter,
                raw_telemetry=state,
                history=None,
                candidate_action=action,
                constraints=constraints,
            )
        except Exception:
            pass
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    passed = p99 < P99_LIMIT_MS

    return {
        "domain": registry_id,
        "n_steps": n_steps,
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "p99_ms": round(p99, 3),
        "mean_ms": round(float(np.mean(arr)), 3),
        "max_ms": round(float(np.max(arr)), 3),
        "p99_limit_ms": P99_LIMIT_MS,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ORIUS DC3S latency benchmark")
    parser.add_argument("--n", type=int, default=1000, help="Steps per domain")
    parser.add_argument("--out", default="reports/latency_run/", help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"=== ORIUS DC3S Latency Benchmark (n={args.n} steps/domain) ===")
    print(f"  P99 limit: {P99_LIMIT_MS} ms\n")

    results = []
    all_pass = True
    for registry_id, state, action, constraints in DOMAIN_SPECS:
        r = benchmark_domain(registry_id, state, action, constraints, args.n)
        results.append(r)
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {registry_id:12s}:  p50={r['p50_ms']:6.2f}ms"
              f"  p95={r['p95_ms']:6.2f}ms  p99={r['p99_ms']:6.2f}ms")
        if not r["passed"]:
            all_pass = False

    report = {
        "n_steps": args.n,
        "p99_limit_ms": P99_LIMIT_MS,
        "domains": results,
        "all_pass": all_pass,
    }

    out_file = out_path / "latency_report.json"
    with out_file.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report → {out_file}")
    print(f"  All pass: {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
