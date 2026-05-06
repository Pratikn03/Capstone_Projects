#!/usr/bin/env python3
"""Build the aerospace trainable surface from real C-MAPSS raw files by default.

The aerospace closure path is intentionally two-surface:

- C-MAPSS is the trainable degradation companion surface.
- Provider-approved multi-flight telemetry is the canonical runtime replay and
  defended safety-validation surface.

Usage:
  python scripts/download_aerospace_datasets.py
  python scripts/download_aerospace_datasets.py --source synthetic
  python scripts/download_aerospace_datasets.py --out data/aerospace/processed/my_aerospace.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pandas as pd

from orius.data_pipeline.real_data_contract import (
    ResolvedRawSource,
    build_provenance_manifest,
    resolve_repo_or_external_raw_dir,
    summarize_csv_output,
    summarize_files,
    utc_now_iso,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "aerospace"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEFAULT_OUT = PROCESSED_DIR / "aerospace_orius.csv"
PROVENANCE_PATH = RAW_DIR / "cmapss_provenance.json"
RUNTIME_REPLAY_DIR = RAW_DIR / "aerospace_flight_telemetry"
RUNTIME_CONTRACT_PATH = RAW_DIR / "multi_flight_runtime_contract.json"
CMAPSS_TRAIN_FILES = ("train_FD001.txt", "train_FD002.txt", "train_FD003.txt", "train_FD004.txt")


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _row_summary_path(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.stem}_row_summary.json"


def _resolve_cmapss_raw_source() -> ResolvedRawSource:
    missing = [name for name in CMAPSS_TRAIN_FILES if not (RAW_DIR / name).exists()]
    if missing:
        raise FileNotFoundError("Missing C-MAPSS train files under data/aerospace/raw: " + ", ".join(missing))
    return ResolvedRawSource(
        path=RAW_DIR,
        source_kind="repo_local",
        checked_locations=(str(RAW_DIR),),
    )


def write_runtime_replay_contract(*, external_root: Path | None = None) -> Path:
    """Record the current availability of the defended aerospace replay surface."""
    runtime_surface = resolve_repo_or_external_raw_dir(
        RUNTIME_REPLAY_DIR,
        external_dataset_key="aerospace_flight_telemetry",
        explicit_root=external_root,
        required=False,
    )
    payload = {
        "generated_at_utc": utc_now_iso(),
        "domain": "aerospace",
        "contract": "runtime_replay_surface",
        "canonical_source": "provider_approved_multi_flight_telemetry",
        "present": runtime_surface is not None,
        "checked_locations": [] if runtime_surface is None else list(runtime_surface.checked_locations),
        "raw_root": None if runtime_surface is None else str(runtime_surface.path),
        "notes": [
            "C-MAPSS remains the trainable degradation companion surface only.",
            "Defended bounded-universal aerospace closure requires a separate multi-flight runtime replay surface.",
        ],
    }
    return write_json(RUNTIME_CONTRACT_PATH, payload)


def _cmapss_columns() -> list[str]:
    return ["unit_number", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + [
        f"sensor_{idx}" for idx in range(1, 22)
    ]


def _load_cmapss_subset(path: Path, subset_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :26].copy()
    df.columns = _cmapss_columns()
    df["dataset_subset"] = subset_name
    return df


def convert_cmapss_to_orius(out_path: Path) -> Path:
    """Convert NASA C-MAPSS raw files into the ORIUS aerospace contract."""
    raw_source = _resolve_cmapss_raw_source()
    frames: list[pd.DataFrame] = []
    subset_rows: dict[str, int] = {}

    for subset_index, filename in enumerate(CMAPSS_TRAIN_FILES):
        subset_name = filename.replace("train_", "").replace(".txt", "")
        raw = _load_cmapss_subset(RAW_DIR / filename, subset_name)
        raw["flight_id"] = raw["dataset_subset"] + "_unit_" + raw["unit_number"].astype(str).str.zfill(3)
        raw["step"] = raw.groupby("flight_id").cumcount()

        max_cycle = raw.groupby("flight_id")["cycle"].transform("max").clip(lower=1)
        remaining = 1.0 - ((raw["cycle"] - 1) / max_cycle)
        raw["fuel_remaining_pct"] = (remaining * 100.0).clip(lower=0.0, upper=100.0)

        # These columns are derived proxies from the engine simulator signals.
        raw["altitude_m"] = pd.to_numeric(raw["sensor_4"], errors="coerce")
        raw["airspeed_kt"] = pd.to_numeric(raw["sensor_20"], errors="coerce") * 10.0
        raw["bank_angle_deg"] = pd.to_numeric(raw["op_setting_2"], errors="coerce").fillna(0.0) * 50.0

        subset_base = pd.Timestamp("2010-01-01T00:00:00Z") + pd.to_timedelta(subset_index * 365, unit="D")
        raw["ts_utc"] = (
            subset_base + pd.to_timedelta(raw["unit_number"] * 10000 + raw["step"], unit="min")
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        frame = raw[
            [
                "flight_id",
                "step",
                "altitude_m",
                "airspeed_kt",
                "bank_angle_deg",
                "fuel_remaining_pct",
                "ts_utc",
                "dataset_subset",
                "unit_number",
                "op_setting_1",
                "op_setting_2",
                "op_setting_3",
                "sensor_4",
                "sensor_20",
                "sensor_21",
            ]
        ].copy()
        frames.append(frame)
        subset_rows[subset_name] = int(len(frame))

    out = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    output_summary = summarize_csv_output(out_path)
    write_json(_row_summary_path(out_path), output_summary)
    manifest = build_provenance_manifest(
        domain="aerospace",
        dataset_key="cmapss",
        provider="NASA C-MAPSS",
        version="FD001-FD004 train corpora",
        raw_source=raw_source,
        processed_output=out_path,
        output_summary=output_summary,
        raw_inventory=summarize_files(raw_source.path),
        source_urls=["https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data"],
        license_notes="Follow NASA dataset terms; keep raw corpora untracked.",
        access_notes="Repo-local raw storage is canonical for this domain.",
        canonical_source=True,
        used_fallback=False,
        notes=[
            "ORIUS aerospace fields are derived proxies from the C-MAPSS engine simulator signals.",
            "fuel_remaining_pct is derived from per-engine cycle progression within each subset.",
            "C-MAPSS does not by itself close the defended aerospace runtime parity gate.",
        ],
        extras={
            "subset_rows": subset_rows,
            "runtime_contract_path": str(RUNTIME_CONTRACT_PATH),
            "derived_field_mapping": {
                "altitude_m": "sensor_4",
                "airspeed_kt": "sensor_20 * 10",
                "bank_angle_deg": "op_setting_2 * 50",
                "fuel_remaining_pct": "1 - ((cycle - 1) / max_cycle_per_engine)",
            },
        },
    )
    write_json(PROVENANCE_PATH, manifest)
    print(f"C-MAPSS aerospace -> {out_path} ({len(out)} rows)")
    return out_path


def generate_synthetic_flight(out_path: Path, n_steps: int = 5000) -> Path:
    """Generate synthetic flight envelope telemetry in ORIUS format."""
    random.seed(42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alt = 3000.0
    v = 150.0
    bank = 0.0
    fuel = 80.0
    rows = []
    for step in range(n_steps):
        alt = max(500, alt + random.gauss(0, 50))
        v = max(60, min(350, v + random.gauss(0, 2)))
        bank = max(-30, min(30, bank + random.gauss(0, 1)))
        fuel = max(5, fuel - random.uniform(0.05, 0.15))
        ts = f"2026-01-01T{step // 3600:02d}:{(step % 3600) // 60:02d}:{step % 60:02d}Z"
        rows.append(
            {
                "flight_id": "syn_001",
                "step": step,
                "altitude_m": f"{alt:.1f}",
                "airspeed_kt": f"{v:.1f}",
                "bank_angle_deg": f"{bank:.1f}",
                "fuel_remaining_pct": f"{fuel:.1f}",
                "ts_utc": ts,
            }
        )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "flight_id",
                "step",
                "altitude_m",
                "airspeed_kt",
                "bank_angle_deg",
                "fuel_remaining_pct",
                "ts_utc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    output_summary = summarize_csv_output(out_path)
    write_json(_row_summary_path(out_path), output_summary)
    write_json(
        RAW_DIR / "synthetic_provenance.json",
        {
            "generated_at_utc": output_summary["generated_at_utc"],
            "domain": "aerospace",
            "dataset_key": "synthetic",
            "provider": "generated",
            "version": "seed-42",
            "source_kind": "synthetic",
            "processed_output": str(out_path),
            "canonical_source": False,
            "used_fallback": True,
            "output_summary": output_summary,
            "notes": [
                "synthetic aerospace output is opt-in only",
                "synthetic aerospace output is not eligible for strict all-domain real-data closure",
            ],
        },
    )
    print(f"Synthetic aerospace -> {out_path} ({len(rows)} rows)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Aerospace datasets for ORIUS")
    parser.add_argument("--source", choices=["cmapss", "synthetic"], default="cmapss", help="Dataset source")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    parser.add_argument(
        "--steps", type=int, default=5000, help="Number of timesteps for synthetic generation"
    )
    parser.add_argument(
        "--external-root",
        type=Path,
        default=None,
        help="Optional external raw-data root used to resolve the aerospace runtime replay surface contract.",
    )
    args = parser.parse_args()
    ensure_dirs()
    write_runtime_replay_contract(external_root=args.external_root)

    try:
        if args.source == "synthetic":
            generate_synthetic_flight(args.out, n_steps=args.steps)
        else:
            convert_cmapss_to_orius(args.out)
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
