#!/usr/bin/env python3
"""Download and convert Healthcare vital-signs datasets for ORIUS multi-domain validation.

Best datasets for vital signs monitoring (HR, SpO2, respiratory rate):
- BIDMC: PhysioNet PPG and Respiration (53 recordings, 1 Hz numerics)
- Synthetic: Generated vital signs for immediate use

Output: data/healthcare/processed/ in ORIUS format (patient_id, step, hr_bpm, spo2_pct, respiratory_rate, ts_utc).

Usage:
  python scripts/download_healthcare_datasets.py --source bidmc   # PhysioNet BIDMC (requires wfdb)
  python scripts/download_healthcare_datasets.py --source synthetic  # Generate synthetic (no download)
  python scripts/download_healthcare_datasets.py --convert path/to/numerics.csv  # Convert existing CSV
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "healthcare"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

BIDMC_BASE = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_bidmc_numerics() -> Path | None:
    """Download BIDMC Numerics CSVs from PhysioNet (first 5 recordings)."""
    bidmc_dir = RAW_DIR / "bidmc"
    bidmc_dir.mkdir(parents=True, exist_ok=True)
    fetched = []
    for i in range(1, 6):
        fname = f"bidmc_{i:02d}_Numerics.csv"
        url = f"{BIDMC_BASE}{fname}"
        try:
            with urlopen(url, timeout=30) as resp:
                data = resp.read().decode("utf-8")
            out_path = bidmc_dir / fname
            out_path.write_text(data, encoding="utf-8")
            fetched.append(out_path)
        except Exception as e:
            print(f"BIDMC {fname} failed: {e}")
    return bidmc_dir if fetched else None


def convert_bidmc_to_orius(csv_path: Path, out_path: Path) -> Path:
    """Convert BIDMC Numerics CSV to ORIUS healthcare format."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    col_map = {}
    if "HR" in df.columns:
        col_map["HR"] = "hr_bpm"
    elif "hr" in df.columns:
        col_map["hr"] = "hr_bpm"
    if "SpO2" in df.columns:
        col_map["SpO2"] = "spo2_pct"
    elif "spo2" in df.columns:
        col_map["spo2"] = "spo2_pct"
    if "RESP" in df.columns:
        col_map["RESP"] = "respiratory_rate"
    elif "RR" in df.columns:
        col_map["RR"] = "respiratory_rate"
    elif "resp" in df.columns:
        col_map["resp"] = "respiratory_rate"
    df = df.rename(columns=col_map)
    out = df.copy()
    out["patient_id"] = csv_path.stem.split("_")[1] if "_" in csv_path.stem else "0"
    out["step"] = range(len(out))
    if "Time [s]" in df.columns:
        out["ts_utc"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(df["Time [s]"], unit="s")
    else:
        out["ts_utc"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(out["step"], unit="s")
    out["ts_utc"] = out["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_cols = ["patient_id", "step", "hr_bpm", "spo2_pct", "respiratory_rate", "ts_utc"]
    out_cols = [c for c in out_cols if c in out.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[out_cols].to_csv(out_path, index=False)
    print(f"Converted -> {out_path} ({len(out)} rows)")
    return out_path


def generate_synthetic_vital_signs(out_path: Path, n_patients: int = 10, steps_per_patient: int = 500) -> Path:
    """Generate synthetic vital signs in ORIUS format."""
    import random
    random.seed(42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in range(n_patients):
        hr_base = 70 + random.uniform(-10, 15)
        spo2_base = 96 + random.uniform(-2, 4)
        rr_base = 14 + random.uniform(-2, 4)
        for step in range(steps_per_patient):
            hr = hr_base + random.gauss(0, 3)
            spo2 = spo2_base + random.gauss(0, 1)
            rr = rr_base + random.gauss(0, 1)
            rows.append({
                "patient_id": pid,
                "step": step,
                "hr_bpm": f"{max(40, min(120, hr)):.1f}",
                "spo2_pct": f"{max(85, min(100, spo2)):.1f}",
                "respiratory_rate": f"{max(8, min(30, rr)):.1f}",
                "ts_utc": f"2026-01-01T{step // 3600:02d}:{(step % 3600) // 60:02d}:{(step % 60):02d}Z",
            })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "step", "hr_bpm", "spo2_pct", "respiratory_rate", "ts_utc"])
        w.writeheader()
        w.writerows(rows)
    print(f"Synthetic vital signs -> {out_path} ({len(rows)} rows)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Healthcare datasets for ORIUS")
    parser.add_argument("--source", choices=["bidmc", "synthetic"], default="synthetic", help="Dataset source")
    parser.add_argument("--convert", type=Path, help="Convert existing BIDMC-format CSV to ORIUS format")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "healthcare_orius.csv", help="Output path")
    args = parser.parse_args()
    ensure_dirs()

    if args.convert:
        convert_bidmc_to_orius(args.convert, args.out)
        return 0

    if args.source == "synthetic":
        generate_synthetic_vital_signs(args.out)
        return 0

    if args.source == "bidmc":
        bidmc_dir = download_bidmc_numerics()
        if bidmc_dir:
            csv_files = sorted(bidmc_dir.glob("bidmc_*_Numerics.csv"))
            if csv_files:
                import pandas as pd
                dfs = []
                for p in csv_files:
                    df = pd.read_csv(p)
                    col_map = {}
                    if "HR" in df.columns:
                        col_map["HR"] = "hr_bpm"
                    if "SpO2" in df.columns:
                        col_map["SpO2"] = "spo2_pct"
                    if "RESP" in df.columns:
                        col_map["RESP"] = "respiratory_rate"
                    elif "RR" in df.columns:
                        col_map["RR"] = "respiratory_rate"
                    df = df.rename(columns=col_map)
                    df["patient_id"] = p.stem.split("_")[1]
                    df["step"] = range(len(df))
                    if "Time [s]" in df.columns:
                        df["ts_utc"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(df["Time [s]"], unit="s")
                    else:
                        df["ts_utc"] = pd.to_datetime("2026-01-01") + pd.to_timedelta(df["step"], unit="s")
                    df["ts_utc"] = df["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    dfs.append(df)
                out = pd.concat(dfs, ignore_index=True)
                out_cols = ["patient_id", "step", "hr_bpm", "spo2_pct", "respiratory_rate", "ts_utc"]
                out_cols = [c for c in out_cols if c in out.columns]
                args.out.parent.mkdir(parents=True, exist_ok=True)
                out[out_cols].to_csv(args.out, index=False)
                print(f"Converted -> {args.out} ({len(out)} rows)")
                return 0
        print("Falling back to synthetic. Run with --source synthetic for immediate use.")
        generate_synthetic_vital_signs(args.out)
        return 0

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
