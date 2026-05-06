#!/usr/bin/env python3
"""ORIUS-Bench replayability verification (Step 4.3).

Runs the benchmark twice with identical seeds and compares outputs.
Writes reports/orius_bench/replay_test.log with PASS/FAIL.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def _run_benchmark(out_dir: Path, seeds: int = 2, horizon: int = 24) -> int:
    """Run ORIUS-Bench release; return exit code."""
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "run_orius_bench_release.py"),
        "--seeds",
        str(seeds),
        "--horizon",
        str(horizon),
        "--out",
        str(out_dir),
    ]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return r.returncode


def _load_leaderboard(path: Path) -> list[dict]:
    """Load leaderboard CSV as list of dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _load_metadata(path: Path) -> dict | None:
    """Load run_metadata.json."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ORIUS-Bench replayability")
    parser.add_argument("--out", default="reports/orius_bench", help="Output directory")
    args = parser.parse_args()

    out_dir = _resolve_repo_path(args.out)
    out_log = out_dir / "replay_test.log"

    seeds = 2
    horizon = 24

    with tempfile.TemporaryDirectory() as tmp:
        dir1 = Path(tmp) / "run1"
        dir2 = Path(tmp) / "run2"
        dir1.mkdir()
        dir2.mkdir()

        # Run 1
        code1 = _run_benchmark(dir1, seeds=seeds, horizon=horizon)
        if code1 != 0:
            out_log.parent.mkdir(parents=True, exist_ok=True)
            out_log.write_text(
                f"[{datetime.now(UTC).isoformat()}] REPLAY TEST: FAIL\nFirst run exited with code {code1}\n"
            )
            print("First run failed")
            return 1

        # Run 2
        code2 = _run_benchmark(dir2, seeds=seeds, horizon=horizon)
        if code2 != 0:
            out_log.parent.mkdir(parents=True, exist_ok=True)
            out_log.write_text(
                f"[{datetime.now(UTC).isoformat()}] REPLAY TEST: FAIL\nSecond run exited with code {code2}\n"
            )
            print("Second run failed")
            return 1

        # Compare leaderboards
        lb1 = _load_leaderboard(dir1 / "leaderboard.csv")
        lb2 = _load_leaderboard(dir2 / "leaderboard.csv")

        if len(lb1) != len(lb2):
            mismatch = f"Row count: {len(lb1)} vs {len(lb2)}"
        else:
            mismatch = None
            for i, (r1, r2) in enumerate(zip(lb1, lb2, strict=False)):
                for k in r1:
                    if k not in r2 or str(r1[k]) != str(r2[k]):
                        mismatch = f"Row {i} key={k}: {r1.get(k)} vs {r2.get(k)}"
                        break
                if mismatch:
                    break

        # Compare run_metadata fault_digests
        meta1 = _load_metadata(dir1 / "run_metadata.json")
        meta2 = _load_metadata(dir2 / "run_metadata.json")
        digest_match = True
        if meta1 and meta2:
            d1 = meta1.get("fault_digests", {})
            d2 = meta2.get("fault_digests", {})
            if d1 != d2:
                digest_match = False
                mismatch = mismatch or f"fault_digests differ: {d1} vs {d2}"

        passed = mismatch is None and digest_match

        # Write replay_test.log
        out_log.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"[{datetime.now(UTC).isoformat()}] ORIUS-Bench Replay Test",
            "",
            f"Seeds: {seeds} (1000..{1000 + seeds - 1})",
            f"Horizon: {horizon}",
            f"Runs per execution: {len(lb1)}",
            "",
            f"Result: {'PASS' if passed else 'FAIL'}",
        ]
        if mismatch:
            lines.extend(["", f"Mismatch: {mismatch}"])
        lines.append("")

        out_log.write_text("\n".join(lines))
        print(f"Replay test: {'PASS' if passed else 'FAIL'}")
        print(f"Wrote {out_log}")
        return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
