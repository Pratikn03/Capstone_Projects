#!/usr/bin/env python3
"""Check that current battery evidence has not drifted from Paper 1 lock.

Fails if locked canonical values in artifacts/paper1_lock differ from
current reports. Used by CI to protect the battery foundation.

Usage:
    python scripts/check_paper1_lock_drift.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = REPO_ROOT / "artifacts" / "paper1_lock"

# If lock does not exist yet, pass (first-time setup)
if not LOCK_DIR.exists():
    print("Paper 1 lock not yet created. Run scripts/snapshot_paper1_lock.py first.")
    sys.exit(0)

# Key artifacts to compare lock vs current (drift = lock != current)
LOCK_ARTIFACTS = [
    "reports/impact_summary.csv",
    "reports/eia930/impact_summary.csv",
    "paper/metrics_manifest.json",
]


def _csv_key_values(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return out
        for col in ("cost_savings_pct", "carbon_reduction_pct", "peak_shaving_pct"):
            if col in rows[0]:
                try:
                    out[col] = float(rows[0][col])
                except (ValueError, TypeError):
                    pass
    return out


def _manifest_key_values(obj: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in obj.items():
        path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, (int, float)):
            out[path] = float(v)
        elif isinstance(v, dict):
            out.update(_manifest_key_values(v, path))
    return out


def main() -> int:
    errors = []

    for rel_path in LOCK_ARTIFACTS:
        lock_path = LOCK_DIR / rel_path
        current_path = REPO_ROOT / rel_path

        if not lock_path.exists():
            continue
        if not current_path.exists():
            errors.append(f"Current artifact missing: {rel_path}")
            continue

        if rel_path.endswith(".csv"):
            lock_vals = _csv_key_values(lock_path)
            current_vals = _csv_key_values(current_path)
            for k in lock_vals:
                if k in current_vals and abs(lock_vals[k] - current_vals[k]) > 1e-4:
                    errors.append(
                        f"{rel_path} {k}: lock={lock_vals[k]}, current={current_vals[k]}"
                    )
        elif rel_path.endswith(".json"):
            try:
                lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
                current_data = json.loads(current_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                errors.append(f"{rel_path}: read error {e}")
                continue
            lock_vals = _manifest_key_values(lock_data.get("canonical_metrics", {}))
            current_vals = _manifest_key_values(current_data.get("canonical_metrics", {}))
            for k in lock_vals:
                if k in current_vals and abs(lock_vals[k] - current_vals[k]) > 1e-4:
                    errors.append(
                        f"{rel_path} canonical_metrics.{k}: lock={lock_vals[k]}, current={current_vals[k]}"
                    )

    # Check for placeholder tokens in primary manuscript only (not sync_rules, etc.)
    placeholder_tokens = ["[see latest frozen outputs]", "To be assigned", "TBD"]
    manuscript_files = [
        REPO_ROOT / "paper" / "PAPER_DRAFT.md",
        REPO_ROOT / "paper" / "paper.tex",
    ]
    for path in manuscript_files:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            for token in placeholder_tokens:
                if token in text:
                    errors.append(f"Placeholder '{token}' found in {path.relative_to(REPO_ROOT)}")

    if errors:
        print("Paper 1 lock drift detected:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("Paper 1 lock check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
