from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_dc3s_latency_script_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "dc3s_latency_benchmark.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_dc3s_steps.py",
            "--iterations",
            "200",
            "--warmup",
            "20",
            "--out",
            str(out_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["iterations"] == 200
    assert "benchmarks" in payload

    benchmarks = payload["benchmarks"]
    for key in [
        "compute_reliability_ms",
        "page_hinkley_update_ms",
        "build_uncertainty_set_ms",
        "repair_action_ms",
        "full_step_linear_ms",
    ]:
        assert key in benchmarks
        assert "mean" in benchmarks[key]
        assert "p95" in benchmarks[key]

    if "build_uncertainty_set_kappa_ms" in benchmarks:
        assert "available" in benchmarks["build_uncertainty_set_kappa_ms"]

    assert "| Component | Mean (ms) |" in proc.stdout
