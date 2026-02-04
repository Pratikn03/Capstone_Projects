"""Run monitoring and retrain only when drift triggers."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Run monitoring before deciding")
    parser.add_argument("--dry-run", action="store_true", help="Print decision without retraining")
    parser.add_argument("--force", action="store_true", help="Force retraining regardless of decision")
    args = parser.parse_args()

    summary_path = Path("reports/monitoring_summary.json")
    if args.refresh or not summary_path.exists():
        subprocess.run([sys.executable, "scripts/run_monitoring.py"], check=True)

    payload = _load_summary(summary_path) or {}
    retraining = payload.get("retraining", {}) or {}
    should_retrain = bool(retraining.get("retrain")) or args.force

    if args.dry_run:
        print(f"[retrain] decision={should_retrain} reasons={retraining.get('reasons')}")
        return

    if not should_retrain:
        print("[retrain] skipped (no drift trigger).")
        return

    print("[retrain] running pipeline (train + reports)...")
    subprocess.run([sys.executable, "-m", "gridpulse.pipeline.run", "--steps", "train,reports", "--force"], check=True)


if __name__ == "__main__":
    main()

