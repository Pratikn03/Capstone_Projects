"""Tests that evidence artifacts exist and are non-empty.

Paper 2: certificate_half_life_blackout.csv
Paper 3: graceful_degradation_trace.csv
Paper 6: certos_lifecycle.csv, certos_summary.json
ORIUS-Bench: M7 (RL) in metrics_engine
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _artifact_ok(path: Path, min_size: int = 10) -> bool:
    return path.exists() and path.stat().st_size >= min_size


def test_paper2_half_life_artifact() -> None:
    """Paper 2: half-life blackout benchmark artifact."""
    csv = REPO / "reports/publication/certificate_half_life_blackout.csv"
    if not csv.exists():
        pytest.skip("Run: python scripts/run_certificate_half_life_blackout.py")
    assert _artifact_ok(csv), f"Paper 2 artifact empty or missing: {csv}"
    content = csv.read_text()
    assert "blackout_hours" in content or "tsvr" in content.lower()


def test_paper3_graceful_artifact() -> None:
    """Paper 3: graceful degradation trace artifact."""
    csv = REPO / "reports/publication/graceful_degradation_trace.csv"
    if not csv.exists():
        pytest.skip("Run: python scripts/run_graceful_degradation.py or generate_priority2_artifacts.py")
    assert _artifact_ok(csv), f"Paper 3 artifact empty or missing: {csv}"


def test_paper6_certos_artifact() -> None:
    """Paper 6: CertOS lifecycle artifact."""
    csv = REPO / "reports/certos/certos_lifecycle.csv"
    j = REPO / "reports/certos/certos_summary.json"
    if not csv.exists():
        pytest.skip("Run: python scripts/run_certos_lifecycle.py")
    assert _artifact_ok(csv), f"CertOS artifact empty or missing: {csv}"
    assert _artifact_ok(j), f"CertOS summary missing: {j}"


def test_orius_bench_m7_recovery_latency() -> None:
    """ORIUS-Bench M7 (Recovery Latency) is defined and exported."""
    from orius.orius_bench.metrics_engine import (
        BenchmarkMetrics,
        StepRecord,
        compute_all_metrics,
        compute_recovery_latency,
    )

    # M7 exists in schema
    assert hasattr(BenchmarkMetrics, "__annotations__")
    assert "recovery_latency" in BenchmarkMetrics.__annotations__

    # compute_recovery_latency works
    records = [
        StepRecord(
            step=0, true_state={}, observed_state={}, action={}, soc_after=0.5, certificate_valid=True
        ),
        StepRecord(
            step=1, true_state={}, observed_state={}, action={}, soc_after=0.5, certificate_valid=False
        ),
        StepRecord(
            step=2, true_state={}, observed_state={}, action={}, soc_after=0.5, certificate_valid=False
        ),
        StepRecord(
            step=3, true_state={}, observed_state={}, action={}, soc_after=0.5, certificate_valid=True
        ),
    ]
    rl = compute_recovery_latency(records)
    assert rl == 2.0  # 2 steps to recover

    # compute_all_metrics includes M7
    m = compute_all_metrics(records)
    assert m.recovery_latency == 2.0
