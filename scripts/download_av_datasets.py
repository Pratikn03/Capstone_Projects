#!/usr/bin/env python3
"""Download and convert AV (autonomous vehicle) datasets for ORIUS vehicles extension.

Best datasets for 1D longitudinal control (position, speed, lead vehicle):
- NGSIM: US highway trajectories (US-101, I-80), 0.1s resolution
- highD: German highway, drone-recorded (requires manual request)
- HEE: Bosch Highway Eagle Eye, ~12k trajectories (GitHub)

Output: data/av/processed/ in ORIUS format (position_m, speed_mps, speed_limit_mps, lead_position_m, ts_utc).

Usage:
  python scripts/download_av_datasets.py --source ngsim   # NGSIM via Kaggle (requires kaggle CLI)
  python scripts/download_av_datasets.py --source hee    # HEE from GitHub
  python scripts/download_av_datasets.py --source synthetic  # Generate synthetic (no download)
  python scripts/download_av_datasets.py --convert path/to/ngsim.csv  # Convert existing CSV
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_AV = REPO_ROOT / "data" / "av"
RAW_DIR = DATA_AV / "raw"
PROCESSED_DIR = DATA_AV / "processed"

# NGSIM column names (standard format)
NGSIM_COLS = ["Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time", "Local_X", "Local_Y", "v_Velocity", "v_Acceleration", "Lane_ID", "Preceding", "Following", "Spacing", "Headway"]


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_ngsim_via_kaggle() -> Path | None:
    """Download NGSIM US-101 from Kaggle. Requires: pip install kaggle, ~/.kaggle/kaggle.json."""
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "nigelwilliams/ngsim-vehicle-trajectory-data-us-101", "-p", str(RAW_DIR)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print("Kaggle download failed. Install: pip install kaggle; configure ~/.kaggle/kaggle.json")
            print(result.stderr or result.stdout)
            return None
        zip_path = RAW_DIR / "ngsim-vehicle-trajectory-data-us-101.zip"
        if zip_path.exists():
            subprocess.run(["unzip", "-o", str(zip_path), "-d", str(RAW_DIR / "ngsim")], check=False)
            return RAW_DIR / "ngsim"
        return None
    except FileNotFoundError:
        print("Kaggle CLI not found. Install: pip install kaggle")
        return None


def download_hee_from_github() -> Path | None:
    """Clone HEE dataset from Bosch GitHub."""
    hee_dir = RAW_DIR / "hee_dataset"
    if hee_dir.exists():
        print(f"HEE already at {hee_dir}")
        return hee_dir
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/boschresearch/hee_dataset.git", str(hee_dir)],
            check=True,
            capture_output=True,
            timeout=60,
        )
        return hee_dir
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"HEE clone failed: {e}")
        return None


def generate_synthetic_trajectories(out_path: Path, n_vehicles: int = 50, steps_per_vehicle: int = 200) -> Path:
    """Generate synthetic longitudinal trajectories in ORIUS format."""
    import random
    random.seed(42)
    speed_limit = 30.0  # m/s
    dt = 0.1  # 10 Hz like NGSIM
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for vid in range(n_vehicles):
        x, v = float(vid * 50), 5.0 + random.uniform(0, 10)
        for step in range(steps_per_vehicle):
            a = random.gauss(0, 0.5)
            v = max(0, min(speed_limit, v + a * dt))
            x = x + v * dt
            lead = x + 20 + random.uniform(5, 30) if random.random() < 0.7 else None
            rows.append({
                "vehicle_id": vid,
                "step": step,
                "position_m": f"{x:.2f}",
                "speed_mps": f"{v:.2f}",
                "speed_limit_mps": str(speed_limit),
                "lead_position_m": f"{lead:.2f}" if lead is not None else "",
                "ts_utc": f"2026-01-01T{step // 3600:02d}:{(step % 3600) // 60:02d}:{(step % 60):02d}Z",
            })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["vehicle_id", "step", "position_m", "speed_mps", "speed_limit_mps", "lead_position_m", "ts_utc"])
        w.writeheader()
        w.writerows(rows)
    print(f"Synthetic trajectories -> {out_path} ({len(rows)} rows)")
    return out_path


def convert_ngsim_to_orius(csv_path: Path, out_path: Path) -> Path:
    """Convert NGSIM-format CSV to ORIUS vehicles format."""
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=50000)
    if "Local_X" not in df.columns or "v_Velocity" not in df.columns:
        raise ValueError(f"Need Local_X and v_Velocity. Found: {list(df.columns)}")
    out = df.rename(columns={"Local_X": "position_m", "v_Velocity": "speed_mps", "Vehicle_ID": "vehicle_id", "Frame_ID": "step"})
    out["speed_limit_mps"] = 30.0
    if "Spacing" in df.columns:
        out["lead_position_m"] = out["position_m"] + df["Spacing"]
    else:
        out["lead_position_m"] = None
    if "Global_Time" in df.columns:
        out["ts_utc"] = pd.to_datetime(df["Global_Time"], unit="s", errors="coerce").astype(str)
    else:
        out["ts_utc"] = pd.Series(range(len(out))).apply(lambda i: f"2026-01-01T{i // 36000:02d}:{(i % 36000) // 600:02d}:{(i % 600) // 10:02d}Z")
    out_cols = ["vehicle_id", "step", "position_m", "speed_mps", "speed_limit_mps", "lead_position_m", "ts_utc"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[out_cols].to_csv(out_path, index=False)
    print(f"Converted -> {out_path} ({len(out)} rows)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AV datasets for ORIUS vehicles")
    parser.add_argument("--source", choices=["ngsim", "hee", "synthetic"], default="synthetic", help="Dataset source")
    parser.add_argument("--convert", type=Path, help="Convert existing NGSIM-format CSV to ORIUS format")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "av_trajectories_orius.csv", help="Output path")
    args = parser.parse_args()
    ensure_dirs()

    if args.convert:
        convert_ngsim_to_orius(args.convert, args.out)
        return 0

    if args.source == "synthetic":
        generate_synthetic_trajectories(args.out)
        return 0

    if args.source == "ngsim":
        ng_dir = download_ngsim_via_kaggle()
        if ng_dir:
            csv_files = list(ng_dir.rglob("*.csv"))
            if csv_files:
                convert_ngsim_to_orius(csv_files[0], args.out)
                return 0
        print("Falling back to synthetic. Run with --source synthetic for immediate use.")
        generate_synthetic_trajectories(args.out)
        return 0

    if args.source == "hee":
        hee_dir = download_hee_from_github()
        if hee_dir:
            csv_files = list(hee_dir.rglob("*.csv"))
            if csv_files:
                convert_ngsim_to_orius(csv_files[0], args.out)
                return 0
        print("HEE structure may differ. Try --convert path/to/hee_file.csv")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
