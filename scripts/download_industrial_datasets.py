#!/usr/bin/env python3
"""Download and convert Industrial process datasets for ORIUS multi-domain validation.

Best datasets for process control (temperature, pressure, power):
- CCPP: UCI Combined Cycle Power Plant (AT, V, AP, RH → PE)
- Hydraulic: UCI Condition Monitoring (pressure, temperature, flow) as companion real-data evidence

Output: data/industrial/processed/ in ORIUS format (sensor_id, step, temp_c, pressure_mbar, humidity_pct, power_mw, ts_utc).

Usage:
  python scripts/download_industrial_datasets.py --source ccpp
  python scripts/download_industrial_datasets.py --source synthetic  # Generate synthetic (no download)
  python scripts/download_industrial_datasets.py --convert path/to/ccpp.csv  # Convert existing CSV
"""
from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path
from urllib.request import urlopen

from orius.data_pipeline.real_data_contract import (
    ResolvedRawSource,
    build_provenance_manifest,
    summarize_csv_output,
    summarize_files,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "industrial"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROVENANCE_PATH = RAW_DIR / "ccpp_provenance.json"

UCI_CCPP_URL = "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _local_ccpp_xlsx_files() -> list[Path]:
    return list((RAW_DIR / "ccpp").rglob("*.xlsx"))


def _row_summary_path(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.stem}_row_summary.json"


def _ccpp_raw_source() -> ResolvedRawSource:
    ccpp_dir = RAW_DIR / "ccpp"
    return ResolvedRawSource(path=ccpp_dir, source_kind="repo_local", checked_locations=(str(ccpp_dir),))


def download_ccpp_from_uci() -> Path | None:
    """Download UCI Combined Cycle Power Plant dataset."""
    try:
        with urlopen(UCI_CCPP_URL, timeout=60) as resp:
            data = resp.read()
        zip_path = RAW_DIR / "ccpp.zip"
        zip_path.write_bytes(data)
        ccpp_dir = RAW_DIR / "ccpp"
        ccpp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(ccpp_dir)
        xlsx_files = list(ccpp_dir.rglob("*.xlsx"))
        return ccpp_dir if xlsx_files else None
    except Exception as e:
        print(f"CCPP download failed: {e}")
        return None


def convert_ccpp_to_orius(csv_path: Path, out_path: Path) -> Path:
    """Convert CCPP-style CSV to ORIUS industrial format."""
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=10000)
    if "AT" not in df.columns or "PE" not in df.columns:
        raise ValueError(f"Need AT, V, AP, RH, PE. Found: {list(df.columns)}")
    out = df.rename(columns={
        "AT": "temp_c",
        "V": "vacuum_cmhg",
        "AP": "pressure_mbar",
        "RH": "humidity_pct",
        "PE": "power_mw",
    })
    out["sensor_id"] = 0
    out["step"] = range(len(out))
    out["ts_utc"] = pd.to_datetime("2006-01-01") + pd.to_timedelta(out["step"], unit="h")
    out["ts_utc"] = out["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_cols = ["sensor_id", "step", "temp_c", "vacuum_cmhg", "pressure_mbar", "humidity_pct", "power_mw", "ts_utc"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[out_cols].to_csv(out_path, index=False)
    write_json(_row_summary_path(out_path), summarize_csv_output(out_path))
    print(f"Converted -> {out_path} ({len(out)} rows)")
    return out_path


def convert_ccpp_xlsx_to_orius(xlsx_path: Path, out_path: Path) -> Path:
    """Convert CCPP xlsx to ORIUS format. Requires: pip install openpyxl"""
    import pandas as pd
    try:
        df = pd.read_excel(xlsx_path, engine="openpyxl", nrows=10000)
    except ImportError:
        raise ImportError("pip install openpyxl required for CCPP xlsx")
    if "AT" not in df.columns or "PE" not in df.columns:
        raise ValueError(f"Need AT, V, AP, RH, PE. Found: {list(df.columns)}")
    out = df.rename(columns={
        "AT": "temp_c",
        "V": "vacuum_cmhg",
        "AP": "pressure_mbar",
        "RH": "humidity_pct",
        "PE": "power_mw",
    })
    out["sensor_id"] = 0
    out["step"] = range(len(out))
    out["ts_utc"] = pd.to_datetime("2006-01-01") + pd.to_timedelta(out["step"], unit="h")
    out["ts_utc"] = out["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out_cols = ["sensor_id", "step", "temp_c", "vacuum_cmhg", "pressure_mbar", "humidity_pct", "power_mw", "ts_utc"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[out_cols].to_csv(out_path, index=False)
    output_summary = summarize_csv_output(out_path)
    write_json(_row_summary_path(out_path), output_summary)
    raw_source = _ccpp_raw_source()
    zema_present = (RAW_DIR / "zema_hydraulic").exists()
    manifest = build_provenance_manifest(
        domain="industrial",
        dataset_key="ccpp",
        provider="UCI Combined Cycle Power Plant",
        version="Folds5x2_pp.xlsx",
        raw_source=raw_source,
        processed_output=out_path,
        output_summary=output_summary,
        raw_inventory=summarize_files(raw_source.path),
        source_urls=[
            "https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant",
            "https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems",
        ],
        license_notes="Follow UCI dataset terms for both CCPP and ZeMA assets.",
        access_notes="CCPP is the primary trainable industrial surface here; ZeMA is tracked as a companion real-data corpus.",
        canonical_source=True,
        used_fallback=False,
        notes=["industrial_orius.csv is built from CCPP", "ZeMA is tracked separately as companion evidence"],
        extras={
            "companion_datasets": {
                "zema_hydraulic_present": zema_present,
                "zema_hydraulic_dir": str(RAW_DIR / "zema_hydraulic"),
            }
        },
    )
    write_json(PROVENANCE_PATH, manifest)
    print(f"Converted -> {out_path} ({len(out)} rows)")
    return out_path


def generate_synthetic_industrial(out_path: Path, n_steps: int = 5000) -> Path:
    """Generate synthetic process control data in ORIUS format."""
    import random
    random.seed(42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in range(n_steps):
        t = 15 + 10 * (step % 100) / 100 + random.gauss(0, 2)
        p = 1010 + random.gauss(0, 5)
        h = 50 + 30 * (step % 200) / 200 + random.gauss(0, 3)
        v = 40 + random.gauss(0, 2)
        power = 450 + 0.5 * t - 0.3 * v + random.gauss(0, 3)
        rows.append({
            "sensor_id": 0,
            "step": step,
            "temp_c": f"{t:.2f}",
            "vacuum_cmhg": f"{v:.2f}",
            "pressure_mbar": f"{p:.2f}",
            "humidity_pct": f"{h:.2f}",
            "power_mw": f"{power:.2f}",
            "ts_utc": f"2026-01-01T{step // 3600:02d}:{(step % 3600) // 60:02d}:{(step % 60):02d}Z",
        })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sensor_id", "step", "temp_c", "vacuum_cmhg", "pressure_mbar", "humidity_pct", "power_mw", "ts_utc"])
        w.writeheader()
        w.writerows(rows)
    output_summary = summarize_csv_output(out_path)
    write_json(_row_summary_path(out_path), output_summary)
    write_json(
        RAW_DIR / "synthetic_provenance.json",
        {
            "generated_at_utc": output_summary["generated_at_utc"],
            "domain": "industrial",
            "dataset_key": "synthetic",
            "provider": "generated",
            "version": "seed-42",
            "source_kind": "synthetic",
            "processed_output": str(out_path),
            "canonical_source": False,
            "used_fallback": True,
            "output_summary": output_summary,
            "notes": [
                "synthetic industrial output is opt-in only",
                "synthetic industrial output is not eligible for strict all-domain real-data closure",
            ],
        },
    )
    print(f"Synthetic industrial -> {out_path} ({len(rows)} rows)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Industrial datasets for ORIUS")
    parser.add_argument("--source", choices=["ccpp", "synthetic"], default="ccpp", help="Dataset source")
    parser.add_argument("--convert", type=Path, help="Convert existing CCPP-format CSV to ORIUS format")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "industrial_orius.csv", help="Output path")
    args = parser.parse_args()
    ensure_dirs()

    if args.convert:
        convert_ccpp_to_orius(args.convert, args.out)
        return 0

    if args.source == "synthetic":
        generate_synthetic_industrial(args.out)
        return 0

    if args.source == "ccpp":
        xlsx_files = _local_ccpp_xlsx_files()
        if xlsx_files:
            convert_ccpp_xlsx_to_orius(xlsx_files[0], args.out)
            return 0
        ccpp_dir = download_ccpp_from_uci()
        if ccpp_dir:
            xlsx_files = list(ccpp_dir.rglob("*.xlsx"))
            if xlsx_files:
                convert_ccpp_xlsx_to_orius(xlsx_files[0], args.out)
                return 0
        print("Unable to build the CCPP industrial surface. Download or place Folds5x2_pp.xlsx under data/industrial/raw/ccpp/.")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
