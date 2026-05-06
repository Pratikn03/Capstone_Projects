#!/usr/bin/env python3
"""Snapshot Paper 1 battery evidence into artifacts/paper1_lock and artifacts/canonical_runs.

Run this once when freezing the battery foundation for ORIUS. The snapshot
enables CI to detect drift from locked values.

Usage:
    python scripts/snapshot_paper1_lock.py
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PAPER1_LOCK_DIR = REPO_ROOT / "artifacts" / "paper1_lock"
CANONICAL_RUNS_DIR = REPO_ROOT / "artifacts" / "canonical_runs"

# Artifacts to snapshot for Paper 1 lock
PAPER1_ARTIFACTS = [
    "paper/metrics_manifest.json",
    "paper/claim_matrix.csv",
    "reports/impact_summary.csv",
    "reports/eia930/impact_summary.csv",
    "reports/research_metrics_de.csv",
    "reports/research_metrics_us.csv",
    "data/dashboard/de_stats.json",
    "data/dashboard/us_stats.json",
    "data/dashboard/manifest.json",
    "reports/publication/release_manifest.json",
    "reports/publication/dc3s_main_table_ci.csv",
    "reports/publication/blackout_half_life.csv",
    "reports/blackout/blackout_study.csv",
]

# Key publication artifacts for canonical runs
CANONICAL_ARTIFACTS = [
    "reports/impact_summary.csv",
    "reports/eia930/impact_summary.csv",
    "reports/research_metrics_de.csv",
    "reports/research_metrics_us.csv",
    "reports/publication/dc3s_main_table_ci.csv",
    "reports/publication/fault_performance_table.csv",
    "reports/publication/dc3s_latency_summary.csv",
]


def main() -> None:
    PAPER1_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    CANONICAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_meta = {
        "snapshot_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Paper 1 battery foundation lock for ORIUS",
        "artifacts_copied": [],
        "artifacts_missing": [],
    }

    for rel_path in PAPER1_ARTIFACTS:
        src = REPO_ROOT / rel_path
        dst = PAPER1_LOCK_DIR / rel_path
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_file():
                shutil.copy2(src, dst)
                snapshot_meta["artifacts_copied"].append(rel_path)
            else:
                shutil.copytree(src, dst, dirs_exist_ok=True)
                snapshot_meta["artifacts_copied"].append(rel_path)
        else:
            snapshot_meta["artifacts_missing"].append(rel_path)

    for rel_path in CANONICAL_ARTIFACTS:
        src = REPO_ROOT / rel_path
        dst = CANONICAL_RUNS_DIR / rel_path
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_file():
                shutil.copy2(src, dst)
                if rel_path not in snapshot_meta["artifacts_copied"]:
                    snapshot_meta["artifacts_copied"].append(rel_path)
        elif rel_path not in snapshot_meta["artifacts_missing"]:
            snapshot_meta["artifacts_missing"].append(rel_path)

    meta_path = PAPER1_LOCK_DIR / "snapshot_meta.json"
    meta_path.write_text(json.dumps(snapshot_meta, indent=2), encoding="utf-8")

    print(f"Snapshot written to {PAPER1_LOCK_DIR}")
    print(f"Canonical runs copied to {CANONICAL_RUNS_DIR}")
    print(f"Copied: {len(snapshot_meta['artifacts_copied'])} artifacts")
    if snapshot_meta["artifacts_missing"]:
        print(f"Missing (optional): {snapshot_meta['artifacts_missing']}")


if __name__ == "__main__":
    main()
