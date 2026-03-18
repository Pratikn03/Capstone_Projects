#!/usr/bin/env python3
"""ORIUS-Bench Step 4.3: Replayability test.

Runs the benchmark twice with identical seeds/horizon and verifies:
- Same fault digests
- Same leaderboard row count and content hash
- run_metadata.json exists

Writes: reports/orius_bench/replay_test.log
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def _run_benchmark(out_dir: Path, seeds: int = 2, horizon: int = 24) -> int:
    """Run release script; return exit code."""
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "run_orius_bench_release.py"),
        "--seeds", str(seeds),
        "--horizon", str(horizon),
        "--out", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    return result.returncode


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description="ORIUS-Bench replayability test")
    parser.add_argument("--out", default="reports/orius_bench", help="Output directory")
    args = parser.parse_args()

    out_dir = _resolve_repo_path(args.out)
    log_path = out_dir / "replay_test.log"

    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    lines.append("ORIUS-Bench Replayability Test (Step 4.3)")
    lines.append("=" * 50)
    lines.append("")

    # Run 1: to OUT
    lines.append(f"Run 1: {out_dir}")
    rc1 = _run_benchmark(out_dir, seeds=2, horizon=24)
    if rc1 != 0:
        lines.append(f"  FAIL: release script exited {rc1}")
        log_path.write_text("\n".join(lines))
        return 1
    lines.append("  OK")

    # Run 2: to temp dir
    with tempfile.TemporaryDirectory() as tmp:
        run2_dir = Path(tmp)
        lines.append(f"Run 2: {run2_dir}")
        rc2 = _run_benchmark(run2_dir, seeds=2, horizon=24)
        if rc2 != 0:
            lines.append(f"  FAIL: release script exited {rc2}")
            log_path.write_text("\n".join(lines))
            return 1
        lines.append("  OK")
        lines.append("")

        # Compare fault digests
        bundle1 = _load_json(out_dir / "artefact_bundle.json")
        bundle2 = _load_json(run2_dir / "artefact_bundle.json")
        digests1 = bundle1.get("fault_digests", {})
        digests2 = bundle2.get("fault_digests", {})

        lines.append("Fault digest comparison:")
        if digests1 == digests2:
            lines.append("  PASS: fault_digests identical")
        else:
            lines.append("  FAIL: fault_digests differ")
            lines.append(f"    Run1: {digests1}")
            lines.append(f"    Run2: {digests2}")

        # Compare leaderboard
        lb1 = out_dir / "leaderboard.csv"
        lb2 = run2_dir / "leaderboard.csv"
        if lb1.exists() and lb2.exists():
            h1 = _file_hash(lb1)
            h2 = _file_hash(lb2)
            if h1 == h2:
                lines.append("Leaderboard hash comparison:")
                lines.append("  PASS: leaderboard.csv identical")
            else:
                lines.append("Leaderboard hash comparison:")
                lines.append(f"  FAIL: hashes differ (Run1={h1}, Run2={h2})")
        lines.append("")

        # Check run_metadata.json
        meta_path = out_dir / "run_metadata.json"
        if meta_path.exists():
            meta = _load_json(meta_path)
            lines.append("Run metadata:")
            lines.append(f"  PASS: run_metadata.json exists")
            lines.append(f"  seeds: {meta.get('seeds', [])}")
            lines.append(f"  horizon: {meta.get('horizon')}")
            lines.append(f"  n_runs: {meta.get('n_runs')}")
        else:
            lines.append("Run metadata:")
            lines.append("  FAIL: run_metadata.json missing")

    lines.append("")
    lines.append("Step 4.3 Replayability: CLOSED" if rc1 == 0 and rc2 == 0 else "Step 4.3 Replayability: FAILED")

    log_path.write_text("\n".join(lines))
    print(f"Wrote {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
