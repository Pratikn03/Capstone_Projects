#!/usr/bin/env python3
"""Normalize AV datasets into the canonical ORIUS longitudinal contract.

Canonical real-data source:
- Waymo Open Motion Dataset from repo-local raw storage

Secondary sources:
- Argoverse 2 Motion for compatibility checks
- Argoverse 2 Sensor for sensor-fault manifest generation

Legacy/testing sources:
- HEE raw CSV already in-repo
- explicit synthetic generation, opt-in only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.data_pipeline.external_raw import (
    EXTERNAL_DATA_ROOT_ENV,
)
from orius.data_pipeline.real_data_contract import (
    ResolvedRawSource,
    build_provenance_manifest,
    resolve_repo_or_external_raw_dir,
    summarize_csv_output,
    summarize_files,
    utc_now_iso,
    write_json,
)


DATA_AV = REPO_ROOT / "data" / "av"
RAW_DIR = DATA_AV / "raw"
PROCESSED_DIR = DATA_AV / "processed"

REAL_SOURCE_CONFIG: dict[str, dict[str, object]] = {
    "waymo_motion": {
        "dataset_dir": "waymo_open_motion",
        "external_dataset_key": "waymo_open_motion",
        "provider": "Waymo Open Dataset",
        "version": "motion",
        "source_urls": [
            "https://waymo.com/open/data/motion/",
            "https://waymo.com/open/faq/",
        ],
        "license_notes": "Waymo Open Dataset license applies; raw payloads must not be redistributed in git.",
        "access_notes": "Registration and license acceptance are typically required.",
        "canonical_source": True,
    },
    "argoverse2_motion": {
        "dataset_dir": "argoverse2_motion",
        "external_dataset_key": "argoverse2_motion",
        "provider": "Argoverse 2",
        "version": "motion",
        "source_urls": ["https://www.argoverse.org/av2.html"],
        "license_notes": "Argoverse 2 license applies; raw payloads remain untracked.",
        "access_notes": "Use as a companion motion corpus; not the canonical AV source.",
        "canonical_source": False,
    },
    "argoverse2_sensor": {
        "dataset_dir": "argoverse2_sensor",
        "external_dataset_key": "argoverse2_sensor",
        "provider": "Argoverse 2",
        "version": "sensor",
        "source_urls": ["https://www.argoverse.org/av2.html"],
        "license_notes": "Argoverse 2 license applies; raw payloads remain untracked.",
        "access_notes": "Companion sensor corpus used for inventory and fault metadata.",
        "canonical_source": False,
    },
}

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "vehicle_id": ("vehicle_id", "track_id", "object_id", "Vehicle_ID"),
    "step": ("step", "frame_index", "timestep", "Frame_ID"),
    "position_m": ("position_m", "position_x", "center_x", "x", "Local_X"),
    "speed_mps": ("speed_mps", "speed", "speed_ms", "v_Velocity"),
    "speed_limit_mps": ("speed_limit_mps", "speed_limit", "posted_speed_mps"),
    "lead_position_m": ("lead_position_m", "lead_x", "preceding_x"),
    "lead_distance_m": ("lead_distance_m", "Spacing", "headway_distance_m"),
    "timestamp": ("ts_utc", "timestamp", "timestamp_utc", "Global_Time"),
    "source_split": ("source_split", "split", "dataset_split"),
    "vx": ("vx", "velocity_x", "vel_x", "speed_x"),
    "vy": ("vy", "velocity_y", "vel_y", "speed_y"),
}


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _row_summary_path(out_path: Path) -> Path:
    return out_path.parent / f"{out_path.stem}_row_summary.json"


def _resolve_real_source(source_key: str, *, external_root: Path | None = None) -> tuple[dict[str, object], ResolvedRawSource]:
    cfg = REAL_SOURCE_CONFIG[source_key]
    raw_source = resolve_repo_or_external_raw_dir(
        RAW_DIR / str(cfg["dataset_dir"]),
        external_dataset_key=str(cfg["external_dataset_key"]),
        explicit_root=external_root,
        required=True,
    )
    assert raw_source is not None
    return cfg, raw_source


def _write_build_artifacts(
    *,
    source_key: str,
    source_cfg: dict[str, object],
    raw_source: ResolvedRawSource,
    out_path: Path,
    extra_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    output_summary = summarize_csv_output(out_path)
    write_json(_row_summary_path(out_path), output_summary)
    manifest = build_provenance_manifest(
        domain="av",
        dataset_key=str(source_cfg["dataset_dir"]),
        provider=str(source_cfg["provider"]),
        version=str(source_cfg["version"]),
        raw_source=raw_source,
        processed_output=out_path,
        output_summary=output_summary,
        raw_inventory=summarize_files(raw_source.path),
        source_urls=[str(url) for url in source_cfg["source_urls"]],
        license_notes=str(source_cfg["license_notes"]),
        access_notes=str(source_cfg["access_notes"]),
        canonical_source=bool(source_cfg["canonical_source"]),
        used_fallback=False,
        notes=[
            "repo-local raw layout is preferred; external raw storage is a fallback only",
            "synthetic data is opt-in and not eligible for strict real-data closure",
        ],
        extras=extra_payload,
    )
    write_json(RAW_DIR / f"{source_cfg['dataset_dir']}_provenance.json", manifest)
    return output_summary


def _first_matching(df: pd.DataFrame, *aliases: str) -> str | None:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _candidate_tabular_files(dataset_dir: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("**/*.parquet", "**/*.csv"):
        files.extend(sorted(path for path in dataset_dir.glob(pattern) if path.is_file()))
    return [path for path in files if path.suffix in {".csv", ".parquet"}]


def _load_tabular(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalize_timestamp(raw: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        if numeric.dropna().empty:
            return pd.Series([""] * len(raw))
        max_value = float(numeric.dropna().max())
        if max_value > 1e15:
            ts = pd.to_datetime(numeric, unit="us", errors="coerce", utc=True)
        elif max_value > 1e12:
            ts = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
        elif max_value > 1e9:
            ts = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
        else:
            base = pd.Timestamp("2026-01-01T00:00:00Z")
            ts = base + pd.to_timedelta(numeric, unit="s")
    else:
        ts = pd.to_datetime(raw, errors="coerce", utc=True)
    return ts.astype(str).str.replace("+00:00", "Z", regex=False)


def _normalize_longitudinal_frame(df: pd.DataFrame, *, default_split: str) -> pd.DataFrame:
    vehicle_col = _first_matching(df, *COLUMN_ALIASES["vehicle_id"])
    position_col = _first_matching(df, *COLUMN_ALIASES["position_m"])
    step_col = _first_matching(df, *COLUMN_ALIASES["step"])
    speed_col = _first_matching(df, *COLUMN_ALIASES["speed_mps"])
    speed_limit_col = _first_matching(df, *COLUMN_ALIASES["speed_limit_mps"])
    lead_position_col = _first_matching(df, *COLUMN_ALIASES["lead_position_m"])
    lead_distance_col = _first_matching(df, *COLUMN_ALIASES["lead_distance_m"])
    timestamp_col = _first_matching(df, *COLUMN_ALIASES["timestamp"])
    split_col = _first_matching(df, *COLUMN_ALIASES["source_split"])
    vx_col = _first_matching(df, *COLUMN_ALIASES["vx"])
    vy_col = _first_matching(df, *COLUMN_ALIASES["vy"])

    if vehicle_col is None or position_col is None:
        raise ValueError(
            "Could not map AV source schema to ORIUS contract. "
            "Need a vehicle identifier and longitudinal position column."
        )

    frame = pd.DataFrame()
    frame["vehicle_id"] = df[vehicle_col].astype(str)
    frame["position_m"] = pd.to_numeric(df[position_col], errors="coerce")

    if speed_col is not None:
        frame["speed_mps"] = pd.to_numeric(df[speed_col], errors="coerce")
    elif vx_col is not None:
        vx = pd.to_numeric(df[vx_col], errors="coerce")
        if vy_col is not None:
            vy = pd.to_numeric(df[vy_col], errors="coerce")
            frame["speed_mps"] = np.sqrt(vx.pow(2) + vy.pow(2))
        else:
            frame["speed_mps"] = vx.abs()
    else:
        raise ValueError("Could not map AV source schema to ORIUS speed_mps.")

    if step_col is not None:
        frame["step"] = pd.to_numeric(df[step_col], errors="coerce")
    else:
        frame["step"] = df.groupby(vehicle_col).cumcount()

    if speed_limit_col is not None:
        frame["speed_limit_mps"] = pd.to_numeric(df[speed_limit_col], errors="coerce").fillna(30.0)
    else:
        frame["speed_limit_mps"] = 30.0

    if lead_position_col is not None:
        frame["lead_position_m"] = pd.to_numeric(df[lead_position_col], errors="coerce")
    elif lead_distance_col is not None:
        lead_distance = pd.to_numeric(df[lead_distance_col], errors="coerce")
        frame["lead_position_m"] = frame["position_m"] + lead_distance
    else:
        frame["lead_position_m"] = pd.NA

    if timestamp_col is not None:
        frame["ts_utc"] = _normalize_timestamp(df[timestamp_col])
    else:
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        frame["ts_utc"] = (
            base + pd.to_timedelta(frame["step"].fillna(0), unit="s")
        ).astype(str).str.replace("+00:00", "Z", regex=False)

    frame["source_split"] = df[split_col].astype(str) if split_col is not None else default_split
    frame = frame.dropna(subset=["position_m", "speed_mps", "step"]).copy()
    frame["step"] = frame["step"].astype(int)
    return frame[
        [
            "vehicle_id",
            "step",
            "position_m",
            "speed_mps",
            "speed_limit_mps",
            "lead_position_m",
            "ts_utc",
            "source_split",
        ]
    ].sort_values(["vehicle_id", "step"]).reset_index(drop=True)


def _infer_split(path: Path, dataset_dir: Path) -> str:
    rel = path.relative_to(dataset_dir).as_posix().lower()
    stem = path.stem.lower()
    for token in ("train", "training", "validation", "val", "test", "calibration"):
        if token in rel or token in stem:
            return "val" if token == "validation" else token
    return "unknown"


def build_real_av_dataset(source_key: str, out_path: Path, *, external_root: Path | None = None) -> Path:
    source_cfg, raw_source = _resolve_real_source(source_key, external_root=external_root)
    dataset_dir = raw_source.path
    candidate_files = _candidate_tabular_files(dataset_dir)
    if not candidate_files:
        raise FileNotFoundError(
            f"No CSV or Parquet exports found under {dataset_dir}. "
            "Place prepared tabular exports derived from the official raw source there."
        )

    frames: list[pd.DataFrame] = []
    sources: dict[str, int] = {}
    for path in candidate_files:
        raw = _load_tabular(path)
        normalized = _normalize_longitudinal_frame(raw, default_split=_infer_split(path, dataset_dir))
        frames.append(normalized)
        sources[str(path.relative_to(dataset_dir))] = int(len(normalized))

    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["vehicle_id", "step", "position_m", "speed_mps", "ts_utc"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    output_summary = _write_build_artifacts(
        source_key=source_key,
        source_cfg=source_cfg,
        raw_source=raw_source,
        out_path=out_path,
        extra_payload={
            "source_key": source_key,
            "normalized_files": sources,
            "contract_columns": [
                "vehicle_id",
                "step",
                "position_m",
                "speed_mps",
                "speed_limit_mps",
                "lead_position_m",
                "ts_utc",
                "source_split",
            ],
        },
    )
    write_json(
        RAW_DIR / f"{source_key}_build_summary.json",
        {
            "source": source_key,
            "raw_source_kind": raw_source.source_kind,
            "raw_root": str(dataset_dir),
            "checked_locations": list(raw_source.checked_locations),
            "output_csv": str(out_path),
            "rows": int(output_summary["rows"]),
            "files": sources,
            "external_env_var": EXTERNAL_DATA_ROOT_ENV,
        },
    )
    return out_path


def build_sensor_summary(out_path: Path, *, external_root: Path | None = None) -> Path:
    source_cfg, raw_source = _resolve_real_source("argoverse2_sensor", external_root=external_root)
    dataset_dir = raw_source.path
    summary = {
        "source": "argoverse2_sensor",
        "raw_source_kind": raw_source.source_kind,
        "raw_root": str(dataset_dir),
        "checked_locations": list(raw_source.checked_locations),
        "files_by_suffix": {},
    }
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        summary["files_by_suffix"].setdefault(path.suffix or "<none>", 0)
        summary["files_by_suffix"][path.suffix or "<none>"] += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = build_provenance_manifest(
        domain="av",
        dataset_key=str(source_cfg["dataset_dir"]),
        provider=str(source_cfg["provider"]),
        version=str(source_cfg["version"]),
        raw_source=raw_source,
        processed_output=out_path,
        output_summary={
            "processed_output": str(out_path),
            "generated_at_utc": utc_now_iso(),
            "rows": 0,
            "columns": [],
        },
        raw_inventory=summarize_files(raw_source.path),
        source_urls=[str(url) for url in source_cfg["source_urls"]],
        license_notes=str(source_cfg["license_notes"]),
        access_notes=str(source_cfg["access_notes"]),
        canonical_source=bool(source_cfg["canonical_source"]),
        used_fallback=False,
        notes=["sensor summary only; no trainable ORIUS surface is produced from this output"],
        extras={"files_by_suffix": summary["files_by_suffix"]},
    )
    write_json(RAW_DIR / "argoverse2_sensor_provenance.json", manifest)
    return out_path


def generate_synthetic_trajectories(out_path: Path, n_vehicles: int = 50, steps_per_vehicle: int = 200) -> Path:
    """Generate synthetic longitudinal trajectories in ORIUS format."""
    import random

    random.seed(42)
    speed_limit = 30.0
    dt = 0.1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for vid in range(n_vehicles):
        x, v = float(vid * 50), 5.0 + random.uniform(0, 10)
        for step in range(steps_per_vehicle):
            a = random.gauss(0, 0.5)
            v = max(0.0, min(speed_limit, v + a * dt))
            x = x + v * dt
            lead = x + 20 + random.uniform(5, 30) if random.random() < 0.7 else None
            rows.append(
                {
                    "vehicle_id": vid,
                    "step": step,
                    "position_m": f"{x:.2f}",
                    "speed_mps": f"{v:.2f}",
                    "speed_limit_mps": str(speed_limit),
                    "lead_position_m": f"{lead:.2f}" if lead is not None else "",
                    "ts_utc": f"2026-01-01T{step // 3600:02d}:{(step % 3600) // 60:02d}:{(step % 60):02d}Z",
                    "source_split": "synthetic",
                }
            )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "vehicle_id",
                "step",
                "position_m",
                "speed_mps",
                "speed_limit_mps",
                "lead_position_m",
                "ts_utc",
                "source_split",
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
            "domain": "av",
            "dataset_key": "synthetic",
            "provider": "generated",
            "version": "seed-42",
            "source_kind": "synthetic",
            "processed_output": str(out_path),
            "canonical_source": False,
            "used_fallback": True,
            "output_summary": output_summary,
            "notes": [
                "synthetic output is opt-in only",
                "synthetic output is not eligible for strict all-domain real-data closure",
            ],
        },
    )
    return out_path


def convert_ngsim_to_orius(csv_path: Path, out_path: Path) -> Path:
    """Convert a legacy NGSIM-like CSV to the ORIUS AV contract."""
    raw = pd.read_csv(csv_path)
    out = _normalize_longitudinal_frame(raw, default_split="legacy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    write_json(_row_summary_path(out_path), summarize_csv_output(out_path))
    return out_path


def convert_hee_to_orius(out_path: Path) -> Path:
    """Convert the in-repo HEE raw CSV into the ORIUS AV contract."""
    hee_path = RAW_DIR / "hee_dataset" / "objectposition.csv"
    if not hee_path.exists():
        raise FileNotFoundError(f"Missing HEE CSV at {hee_path}")
    raw = pd.read_csv(hee_path)
    raw = raw.rename(
        columns={
            _first_matching(raw, "id", "track_id", "object_id") or "id": "vehicle_id",
            _first_matching(raw, "timestamp", "time", "frame") or "timestamp": "timestamp",
            _first_matching(raw, "xCenter", "x", "position_x") or "x": "position_m",
            _first_matching(raw, "xVelocity", "vx", "velocity_x") or "vx": "vx",
        }
    )
    out = _normalize_longitudinal_frame(raw, default_split="hee")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    output_summary = summarize_csv_output(out_path)
    write_json(_row_summary_path(out_path), output_summary)
    raw_source = ResolvedRawSource(
        path=RAW_DIR / "hee_dataset",
        source_kind="repo_local",
        checked_locations=(str(RAW_DIR / "hee_dataset"),),
    )
    manifest = build_provenance_manifest(
        domain="av",
        dataset_key="hee_dataset",
        provider="Bosch Research HEE",
        version="objectposition.csv",
        raw_source=raw_source,
        processed_output=out_path,
        output_summary=output_summary,
        raw_inventory=summarize_files(raw_source.path),
        source_urls=["https://github.com/boschresearch/hee_dataset"],
        license_notes="See data/av/raw/hee_dataset/LICENSE.txt for the dataset terms.",
        access_notes="Legacy compatibility source only; not the canonical AV corpus.",
        canonical_source=False,
        used_fallback=False,
        notes=["HEE remains available for bounded compatibility only"],
    )
    write_json(RAW_DIR / "hee_dataset_provenance.json", manifest)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AV datasets for ORIUS")
    parser.add_argument(
        "--source",
        choices=[
            "waymo_motion",
            "argoverse2_motion",
            "argoverse2_sensor",
            "hee",
            "ngsim",
            "synthetic",
        ],
        default="waymo_motion",
        help="Dataset source to normalize",
    )
    parser.add_argument("--external-root", type=Path, default=None, help="Override ORIUS_EXTERNAL_DATA_ROOT")
    parser.add_argument("--convert", type=Path, help="Convert an existing legacy AV CSV to ORIUS format")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "av_trajectories_orius.csv", help="Output path")
    parser.add_argument(
        "--sensor-summary-out",
        type=Path,
        default=RAW_DIR / "argoverse2_sensor_summary.json",
        help="Output for the Argoverse 2 sensor inventory summary",
    )
    args = parser.parse_args()
    ensure_dirs()

    try:
        if args.convert:
            convert_ngsim_to_orius(args.convert, args.out)
            print(f"Converted -> {args.out}")
            return 0

        if args.source == "synthetic":
            generate_synthetic_trajectories(args.out)
            print(f"Synthetic trajectories -> {args.out}")
            return 0

        if args.source == "ngsim":
            raise FileNotFoundError(
                "Canonical AV ingestion no longer downloads NGSIM automatically. "
                "Use --convert with an existing CSV or switch to --source waymo_motion."
            )

        if args.source == "hee":
            convert_hee_to_orius(args.out)
            print(f"HEE trajectories -> {args.out}")
            return 0

        if args.source == "argoverse2_sensor":
            out = build_sensor_summary(args.sensor_summary_out, external_root=args.external_root)
            print(f"Sensor summary -> {out}")
            return 0

        out = build_real_av_dataset(args.source, args.out, external_root=args.external_root)
        print(f"AV trajectories -> {out}")
        return 0
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
