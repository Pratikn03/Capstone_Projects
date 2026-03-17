#!/usr/bin/env python3
"""Generate synthetic Aerospace flight telemetry for ORIUS multi-domain validation.

No real dataset yet; generates placeholder data for training pipeline.
Output: data/aerospace/processed/aerospace_orius.csv

Usage:
  python scripts/download_aerospace_datasets.py
  python scripts/download_aerospace_datasets.py --out data/aerospace/processed/my_aerospace.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "aerospace"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_OUT = PROCESSED_DIR / "aerospace_orius.csv"


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_flight(out_path: Path, n_steps: int = 5000) -> Path:
    """Generate synthetic flight envelope telemetry in ORIUS format."""
    random.seed(42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alt = 3000.0
    v = 150.0
    bank = 0.0
    fuel = 80.0
    dt_s = 1.0
    rows = []
    for step in range(n_steps):
        alt = max(500, alt + random.gauss(0, 50))
        v = max(60, min(350, v + random.gauss(0, 2)))
        bank = max(-30, min(30, bank + random.gauss(0, 1)))
        fuel = max(5, fuel - random.uniform(0.05, 0.15))
        ts = f"2026-01-01T{step // 3600:02d}:{(step % 3600) // 60:02d}:{step % 60:02d}Z"
        rows.append({
            "flight_id": "syn_001",
            "step": step,
            "altitude_m": f"{alt:.1f}",
            "airspeed_kt": f"{v:.1f}",
            "bank_angle_deg": f"{bank:.1f}",
            "fuel_remaining_pct": f"{fuel:.1f}",
            "ts_utc": ts,
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["flight_id", "step", "altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct", "ts_utc"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Synthetic aerospace -> {out_path} ({len(rows)} rows)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic Aerospace datasets")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    parser.add_argument("--steps", type=int, default=5000, help="Number of timesteps")
    args = parser.parse_args()
    ensure_dirs()
    generate_synthetic_flight(args.out, n_steps=args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
