#!/usr/bin/env python3
"""Micro-benchmark core DC3S step functions."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gridpulse.dc3s.calibration import build_uncertainty_set
from gridpulse.dc3s.drift import PageHinkleyDetector
from gridpulse.dc3s.quality import compute_reliability
from gridpulse.dc3s.shield import repair_action

try:
    from gridpulse.dc3s.calibration import build_uncertainty_set_kappa
except ImportError:  # pragma: no cover - kept for compatibility
    build_uncertainty_set_kappa = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark core DC3S per-step functions")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--out", default="reports/dc3s_latency_benchmark.json")
    return parser.parse_args()


def _mean_ms(samples_ns: list[int]) -> float:
    return float(np.mean(np.asarray(samples_ns, dtype=np.float64)) / 1_000_000.0)


def _p95_ms(samples_ns: list[int]) -> float:
    return float(np.quantile(np.asarray(samples_ns, dtype=np.float64), 0.95) / 1_000_000.0)


def _time_callable(fn: Callable[[], Any], iterations: int, warmup: int) -> dict[str, float]:
    for _ in range(max(0, int(warmup))):
        fn()

    samples_ns: list[int] = []
    for _ in range(max(1, int(iterations))):
        start = time.perf_counter_ns()
        fn()
        samples_ns.append(time.perf_counter_ns() - start)

    return {
        "mean": _mean_ms(samples_ns),
        "p95": _p95_ms(samples_ns),
    }


def _component_inputs() -> dict[str, Any]:
    last_event = {
        "ts_utc": "2026-01-01T00:00:00Z",
        "load_mw": 100.0,
        "renewables_mw": 10.0,
    }
    event = {
        "ts_utc": "2026-01-01T01:00:00Z",
        "load_mw": 102.0,
        "renewables_mw": 11.0,
    }
    reliability_cfg = {
        "lambda_delay": 0.002,
        "spike_beta": 0.25,
        "ooo_gamma": 0.35,
        "min_w": 0.05,
    }
    ftit_cfg = {
        "law": "linear",
        "decay": 0.98,
        "stale_k": 3,
        "stale_tol": 1.0e-9,
    }
    linear_cfg = {
        "k_q": 0.8,
        "k_drift": 0.6,
        "infl_max": 3.0,
        "cooldown_smoothing": 0.0,
        "reliability": {"min_w": 0.05},
        "shield": {"mode": "projection", "reserve_soc_pct_drift": 0.0},
    }
    kappa_cfg = {
        "reliability": {"min_w": 0.05},
        "infl_max": 3.0,
        "kappa_drift_penalty": 0.5,
        "cooldown_smoothing": 0.0,
    }
    uncertainty_set = {
        "lower": [90.0],
        "upper": [110.0],
        "meta": {"drift_flag": False},
        "renewables_forecast": [10.0],
        "price": [50.0],
    }
    constraints = {
        "capacity_mwh": 100.0,
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "max_power_mw": 20.0,
        "max_charge_mw": 20.0,
        "max_discharge_mw": 20.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
        "last_net_mw": 0.0,
        "ramp_mw": 20.0,
        "time_step_hours": 1.0,
    }
    return {
        "last_event": last_event,
        "event": event,
        "reliability_cfg": reliability_cfg,
        "ftit_cfg": ftit_cfg,
        "linear_cfg": linear_cfg,
        "kappa_cfg": kappa_cfg,
        "uncertainty_set": uncertainty_set,
        "constraints": constraints,
    }


def _benchmark_chain(iterations: int, warmup: int, *, use_kappa: bool) -> dict[str, float]:
    payload = _component_inputs()
    detector = PageHinkleyDetector.from_state(
        None,
        cfg={"ph_delta": 0.01, "ph_lambda": 5.0, "warmup_steps": 48, "cooldown_steps": 24},
    )
    events = [
        {"ts_utc": "2026-01-01T00:00:00Z", "load_mw": 100.0, "renewables_mw": 10.0},
        {"ts_utc": "2026-01-01T01:00:00Z", "load_mw": 102.0, "renewables_mw": 11.0},
        {"ts_utc": "2026-01-01T02:00:00Z", "load_mw": 101.0, "renewables_mw": 12.0},
        {"ts_utc": "2026-01-01T03:00:00Z", "load_mw": 103.0, "renewables_mw": 10.5},
    ]
    yhat = np.asarray([100.0, 102.0, 101.0, 103.0], dtype=float)
    ytrue = np.asarray([101.0, 100.0, 104.0, 102.0], dtype=float)
    q = np.asarray([10.0], dtype=float)
    adaptive_state: dict[str, Any] = {}
    current_soc = 50.0
    last_event: dict[str, Any] | None = payload["last_event"]

    def run_step() -> None:
        nonlocal adaptive_state, current_soc, last_event
        idx = detector.count % len(events)
        event = events[idx]
        w_t, _ = compute_reliability(
            event,
            last_event,
            expected_cadence_s=3600.0,
            reliability_cfg=payload["reliability_cfg"],
            adaptive_state=adaptive_state,
            ftit_cfg=payload["ftit_cfg"],
        )
        residual = abs(float(ytrue[idx]) - float(yhat[idx]))
        drift = detector.update(residual)
        if use_kappa and build_uncertainty_set_kappa is not None:
            lower, upper, meta = build_uncertainty_set_kappa(
                yhat=np.asarray([yhat[idx]], dtype=float),
                q=q,
                w_t=w_t,
                drift_flag=bool(drift["drift"]),
                cfg=payload["kappa_cfg"],
                sigma_sq=4.0,
            )
        else:
            lower, upper, meta = build_uncertainty_set(
                yhat=np.asarray([yhat[idx]], dtype=float),
                q=q,
                w_t=w_t,
                drift_flag=bool(drift["drift"]),
                cfg=payload["linear_cfg"],
            )
        uncertainty_set = dict(payload["uncertainty_set"])
        uncertainty_set["lower"] = lower.tolist()
        uncertainty_set["upper"] = upper.tolist()
        uncertainty_set["meta"] = meta
        safe, _ = repair_action(
            a_star={"charge_mw": 5.0, "discharge_mw": 0.0},
            state={"current_soc_mwh": current_soc},
            uncertainty_set=uncertainty_set,
            constraints={**payload["constraints"], "current_soc_mwh": current_soc},
            cfg=payload["linear_cfg"],
        )
        current_soc = current_soc + safe["charge_mw"] - safe["discharge_mw"]
        current_soc = min(payload["constraints"]["max_soc_mwh"], max(payload["constraints"]["min_soc_mwh"], current_soc))
        adaptive_state = {}
        last_event = event

    return _time_callable(run_step, iterations=iterations, warmup=warmup)


def _render_markdown_table(benchmarks: dict[str, dict[str, Any]]) -> str:
    rows = [
        ("compute_reliability", "compute_reliability_ms"),
        ("page_hinkley_update", "page_hinkley_update_ms"),
        ("build_uncertainty_set", "build_uncertainty_set_ms"),
        ("build_uncertainty_set_kappa", "build_uncertainty_set_kappa_ms"),
        ("repair_action", "repair_action_ms"),
        ("full_step_linear", "full_step_linear_ms"),
        ("full_step_kappa", "full_step_kappa_ms"),
    ]
    lines = [
        "# DC3S Latency Benchmark",
        "",
        "| Component | Mean (ms) | P95 (ms) |",
        "| --- | ---: | ---: |",
    ]
    for label, key in rows:
        data = benchmarks.get(key, {})
        if data.get("available", True) is False:
            lines.append(f"| {label} | N/A | N/A |")
            continue
        lines.append(f"| {label} | {float(data['mean']):.3f} | {float(data['p95']):.3f} |")
    return "\n".join(lines)


def run_benchmark(*, iterations: int, warmup: int, out_path: Path) -> dict[str, Any]:
    payload = _component_inputs()
    detector = PageHinkleyDetector.from_state(
        None,
        cfg={"ph_delta": 0.01, "ph_lambda": 5.0, "warmup_steps": 48, "cooldown_steps": 24},
    )

    benchmarks: dict[str, dict[str, Any]] = {}
    benchmarks["compute_reliability_ms"] = _time_callable(
        lambda: compute_reliability(
            payload["event"],
            payload["last_event"],
            expected_cadence_s=3600.0,
            reliability_cfg=payload["reliability_cfg"],
            adaptive_state={},
            ftit_cfg=payload["ftit_cfg"],
        ),
        iterations=iterations,
        warmup=warmup,
    )
    benchmarks["page_hinkley_update_ms"] = _time_callable(
        lambda: detector.update(2.0),
        iterations=iterations,
        warmup=warmup,
    )
    benchmarks["build_uncertainty_set_ms"] = _time_callable(
        lambda: build_uncertainty_set(
            yhat=np.asarray([100.0], dtype=float),
            q=np.asarray([10.0], dtype=float),
            w_t=0.8,
            drift_flag=False,
            cfg=payload["linear_cfg"],
        ),
        iterations=iterations,
        warmup=warmup,
    )

    if build_uncertainty_set_kappa is not None:
        result = _time_callable(
            lambda: build_uncertainty_set_kappa(
                yhat=np.asarray([100.0], dtype=float),
                q=np.asarray([10.0], dtype=float),
                w_t=0.8,
                drift_flag=False,
                cfg=payload["kappa_cfg"],
                sigma_sq=4.0,
                delta=0.10,
                eps_floor=50.0,
            ),
            iterations=iterations,
            warmup=warmup,
        )
        benchmarks["build_uncertainty_set_kappa_ms"] = {"available": True, **result}
    else:
        benchmarks["build_uncertainty_set_kappa_ms"] = {"available": False}

    benchmarks["repair_action_ms"] = _time_callable(
        lambda: repair_action(
            a_star={"charge_mw": 5.0, "discharge_mw": 0.0},
            state={"current_soc_mwh": 50.0},
            uncertainty_set=payload["uncertainty_set"],
            constraints={**payload["constraints"], "current_soc_mwh": 50.0},
            cfg=payload["linear_cfg"],
        ),
        iterations=iterations,
        warmup=warmup,
    )
    benchmarks["full_step_linear_ms"] = _benchmark_chain(iterations, warmup, use_kappa=False)
    if build_uncertainty_set_kappa is not None:
        result = _benchmark_chain(iterations, warmup, use_kappa=True)
        benchmarks["full_step_kappa_ms"] = {"available": True, **result}
    else:
        benchmarks["full_step_kappa_ms"] = {"available": False}

    payload_json = {
        "iterations": int(iterations),
        "warmup": int(warmup),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "benchmarks": benchmarks,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload_json, indent=2, sort_keys=True), encoding="utf-8")
    return payload_json


def main() -> None:
    args = _parse_args()
    payload = run_benchmark(
        iterations=max(1, int(args.iterations)),
        warmup=max(0, int(args.warmup)),
        out_path=Path(args.out),
    )
    print(_render_markdown_table(payload["benchmarks"]))


if __name__ == "__main__":
    main()
