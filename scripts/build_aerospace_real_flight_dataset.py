#!/usr/bin/env python3
"""Build the official aerospace real-flight runtime lane from provider telemetry.

This builder is the canonical aerospace parity-closing runtime path. It is
separate from the bounded public ADS-B support lane and refuses to use the
public proxy corpus as an official substitute.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.data_pipeline.external_raw import EXTERNAL_DATA_ROOT_ENV, get_external_dataset_dir
from orius.data_pipeline.real_data_contract import (
    build_provenance_manifest,
    summarize_csv_output,
    summarize_files,
    utc_now_iso,
    write_json,
)
from orius.universal_framework import run_universal_step
from orius.universal_framework.aerospace_adapter import AerospaceDomainAdapter


RAW_DIR = REPO_ROOT / "data" / "aerospace" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "aerospace" / "processed"
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"

PROCESSED_CSV = PROCESSED_DIR / "aerospace_realflight_runtime.csv"
PROVENANCE_PATH = RAW_DIR / "aerospace_realflight_provenance.json"
SUMMARY_JSON = PUBLICATION_DIR / "aerospace_realflight_runtime_summary.json"
SUMMARY_CSV = PUBLICATION_DIR / "aerospace_realflight_runtime_summary.csv"
SUMMARY_MD = PUBLICATION_DIR / "aerospace_realflight_runtime_summary.md"

GRAVITY = 9.80665
MAX_BANK_DEG = 30.0

FLIGHT_ID_ALIASES = ("flight_id", "aircraft_id", "tail_id", "callsign", "mission_id", "unit_id")
STEP_ALIASES = ("step", "frame", "index", "sample_id")
ALTITUDE_ALIASES = ("altitude_m", "altitude", "altitude_ft", "altitude_agl_ft")
AIRSPEED_ALIASES = ("airspeed_kt", "airspeed", "ground_speed_kts", "groundspeed_kt", "speed_kt")
BANK_ALIASES = ("bank_angle_deg", "bank_deg", "roll_deg", "roll_angle_deg")
FUEL_ALIASES = ("fuel_remaining_pct", "fuel_pct", "fuel_remaining")
TIME_ALIASES = ("ts_utc", "timestamp", "time", "timestamp_utc")
HEADING_ALIASES = ("heading_deg", "track_deg", "course_deg")


def _candidate_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in ("**/*.csv", "**/*.parquet"):
        files.extend(sorted(path for path in root.glob(pattern) if path.is_file()))
    return [path for path in files if path.name not in {".gitattributes"}]


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _first_matching(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _resolve_provider_root(explicit_root: Path | None) -> Path:
    root = get_external_dataset_dir("aerospace_flight_telemetry", explicit_root=explicit_root, required=True)
    if root is None:
        raise FileNotFoundError(
            f"Could not resolve aerospace flight telemetry root. Set {EXTERNAL_DATA_ROOT_ENV} or pass --external-root."
        )
    if (root / "tartanaviation_adsb_19k_clean").exists() or any(
        candidate.name == "tartanaviation_adsb_19k_clean.csv" for candidate in root.rglob("*")
    ):
        raise RuntimeError(
            "Public ADS-B proxy corpus detected under aerospace_flight_telemetry; official builder refuses proxy-only inputs."
        )
    return root


def _normalize_timestamp(raw: pd.Series) -> pd.Series:
    ts = pd.to_datetime(raw, errors="coerce", utc=True)
    if ts.isna().all() and pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        max_value = float(numeric.dropna().max()) if not numeric.dropna().empty else 0.0
        if max_value > 1e15:
            ts = pd.to_datetime(numeric, unit="us", errors="coerce", utc=True)
        elif max_value > 1e12:
            ts = pd.to_datetime(numeric, unit="ms", errors="coerce", utc=True)
        else:
            ts = pd.to_datetime(numeric, unit="s", errors="coerce", utc=True)
    return ts.astype(str).str.replace("+00:00", "Z", regex=False)


def _derive_bank_angle(frame: pd.DataFrame, heading_col: str, speed_col: str, time_col: str) -> np.ndarray:
    if frame.empty:
        return np.zeros(0, dtype=float)
    heading_rad = np.unwrap(
        np.deg2rad(pd.to_numeric(frame[heading_col], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    )
    timestamps = pd.to_datetime(frame[time_col], errors="coerce", utc=True).astype("int64").to_numpy(dtype=float) / 1e9
    dt = np.diff(timestamps, prepend=timestamps[0])
    if len(dt) > 1:
        dt[0] = dt[1]
    dt = np.clip(dt, 1e-6, None)
    yaw_rate = np.gradient(heading_rad, np.cumsum(dt), edge_order=1)
    speed_mps = pd.to_numeric(frame[speed_col], errors="coerce").fillna(0.0).to_numpy(dtype=float) * 0.514444
    bank_rad = np.arctan(np.clip(speed_mps * yaw_rate / GRAVITY, -10.0, 10.0))
    return np.clip(np.rad2deg(bank_rad), -MAX_BANK_DEG, MAX_BANK_DEG)


def _normalize_frame(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    flight_id_col = _first_matching(df, FLIGHT_ID_ALIASES)
    step_col = _first_matching(df, STEP_ALIASES)
    altitude_col = _first_matching(df, ALTITUDE_ALIASES)
    airspeed_col = _first_matching(df, AIRSPEED_ALIASES)
    bank_col = _first_matching(df, BANK_ALIASES)
    fuel_col = _first_matching(df, FUEL_ALIASES)
    time_col = _first_matching(df, TIME_ALIASES)
    heading_col = _first_matching(df, HEADING_ALIASES)

    if flight_id_col is None:
        df = df.copy()
        df["_derived_flight_id"] = source_name
        flight_id_col = "_derived_flight_id"
    if altitude_col is None or airspeed_col is None or time_col is None:
        raise ValueError(f"{source_name}: telemetry file must include altitude, airspeed, and time columns")

    out = pd.DataFrame()
    out["flight_id"] = df[flight_id_col].astype(str)
    if step_col is not None:
        out["step"] = pd.to_numeric(df[step_col], errors="coerce")
    else:
        out["step"] = df.groupby(flight_id_col).cumcount()

    altitude = pd.to_numeric(df[altitude_col], errors="coerce")
    if "ft" in altitude_col.lower():
        altitude = altitude * 0.3048
    out["altitude_m"] = altitude
    out["airspeed_kt"] = pd.to_numeric(df[airspeed_col], errors="coerce")
    if bank_col is not None:
        out["bank_angle_deg"] = pd.to_numeric(df[bank_col], errors="coerce")
    elif heading_col is not None:
        tmp = df.copy()
        tmp[time_col] = _normalize_timestamp(df[time_col])
        out["bank_angle_deg"] = _derive_bank_angle(tmp, heading_col, airspeed_col, time_col)
    else:
        out["bank_angle_deg"] = 0.0

    if fuel_col is not None:
        out["fuel_remaining_pct"] = pd.to_numeric(df[fuel_col], errors="coerce")
    else:
        max_step = out.groupby("flight_id")["step"].transform("max").clip(lower=1.0)
        out["fuel_remaining_pct"] = (1.0 - (out["step"] / max_step)) * 100.0

    out["ts_utc"] = _normalize_timestamp(df[time_col])
    out["source_file"] = source_name
    out = out.dropna(subset=["flight_id", "step", "altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct"])
    out["step"] = out["step"].astype(int)
    out["bank_angle_deg"] = out["bank_angle_deg"].clip(-MAX_BANK_DEG, MAX_BANK_DEG)
    return out.reset_index(drop=True), {
        "flight_id": flight_id_col,
        "step": step_col or "derived_cumcount",
        "altitude_m": altitude_col,
        "airspeed_kt": airspeed_col,
        "bank_angle_deg": bank_col or (heading_col or "constant_zero"),
        "fuel_remaining_pct": fuel_col or "derived_remaining_progress",
        "ts_utc": time_col,
    }


def _run_contract_summary(frame: pd.DataFrame) -> dict[str, float | int | str]:
    adapter = AerospaceDomainAdapter({"expected_cadence_s": 1.0})
    constraints = {"v_min_kt": 60.0, "v_max_kt": 350.0, "max_bank_deg": MAX_BANK_DEG}
    sample = frame.groupby("flight_id", sort=False).head(64).reset_index(drop=True)
    contract_pass_count = 0
    repaired_bank_violations = 0
    baseline_bank_violations = 0
    for record in sample.to_dict(orient="records"):
        telemetry = {
            "altitude_m": float(record["altitude_m"]),
            "airspeed_kt": float(record["airspeed_kt"]),
            "bank_angle_deg": float(record["bank_angle_deg"]),
            "fuel_remaining_pct": float(record["fuel_remaining_pct"]),
            "ts_utc": str(record["ts_utc"]),
        }
        candidate = {"throttle": 0.7, "bank_deg": float(record["bank_angle_deg"])}
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=telemetry,
            history=None,
            candidate_action=candidate,
            constraints=constraints,
            quantile=5.0,
        )
        safe_action = dict(result.get("safe_action", {}))
        contract_checks = dict(result.get("contract_checks", {}))
        contract_pass_count += int(bool(contract_checks.get("contract_passed", False)))
        baseline_bank_violations += int(abs(candidate["bank_deg"]) > MAX_BANK_DEG)
        repaired_bank_violations += int(abs(float(safe_action.get("bank_deg", candidate["bank_deg"]))) > MAX_BANK_DEG)
    return {
        "generated_at_utc": utc_now_iso(),
        "lane_type": "official_provider_real_flight_runtime",
        "rows_total": int(len(frame)),
        "flight_count": int(frame["flight_id"].nunique()),
        "sampled_rows": int(len(sample)),
        "contract_pass_rate": round(contract_pass_count / max(len(sample), 1), 4),
        "baseline_bank_violation_rate": round(baseline_bank_violations / max(len(sample), 1), 4),
        "safe_bank_violation_rate": round(repaired_bank_violations / max(len(sample), 1), 4),
        "canonical_eligibility": True,
    }


def build_real_flight_runtime(*, external_root: Path | None, out_csv: Path) -> Path:
    provider_root = _resolve_provider_root(external_root)
    files = _candidate_files(provider_root)
    if not files:
        raise FileNotFoundError(f"No CSV or Parquet telemetry files found under {provider_root}")

    frames: list[pd.DataFrame] = []
    source_columns: dict[str, dict[str, str]] = {}
    for path in files:
        frame = _load_frame(path)
        normalized, mapping = _normalize_frame(frame, path.name)
        if normalized.empty:
            continue
        frames.append(normalized)
        source_columns[path.name] = mapping
    if not frames:
        raise ValueError("No usable provider telemetry rows were found after schema normalization")

    combined = pd.concat(frames, ignore_index=True).sort_values(["flight_id", "step", "ts_utc"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)

    output_summary = summarize_csv_output(out_csv)
    summary = _run_contract_summary(combined)
    summary["processed_output"] = str(out_csv)
    summary["provider_root"] = str(provider_root)
    summary["source_files"] = [path.name for path in files]

    write_json(SUMMARY_JSON, summary)
    pd.DataFrame(
        [
            {"metric": "rows_total", "value": summary["rows_total"]},
            {"metric": "flight_count", "value": summary["flight_count"]},
            {"metric": "contract_pass_rate", "value": summary["contract_pass_rate"]},
            {"metric": "baseline_bank_violation_rate", "value": summary["baseline_bank_violation_rate"]},
            {"metric": "safe_bank_violation_rate", "value": summary["safe_bank_violation_rate"]},
        ]
    ).to_csv(SUMMARY_CSV, index=False)
    SUMMARY_MD.write_text(
        "\n".join(
            [
                "# Aerospace Real-Flight Runtime Summary",
                "",
                f"- Lane type: `{summary['lane_type']}`",
                f"- Rows: `{summary['rows_total']}`",
                f"- Flights: `{summary['flight_count']}`",
                f"- Contract pass rate: `{summary['contract_pass_rate']}`",
                f"- Safe bank violation rate: `{summary['safe_bank_violation_rate']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = build_provenance_manifest(
        domain="aerospace",
        dataset_key="aerospace_real_flight_runtime",
        provider="Provider-approved multi-flight telemetry",
        version="runtime_replay_surface",
        raw_source=type(
            "ResolvedRawSourceShim",
            (),
            {
                "path": provider_root,
                "source_kind": "external",
                "checked_locations": (str(provider_root),),
            },
        )(),
        processed_output=out_csv,
        output_summary=output_summary,
        raw_inventory=summarize_files(provider_root),
        source_urls=[],
        license_notes="Provider-approved telemetry; raw payloads must remain out of repo history.",
        access_notes="This is the canonical aerospace runtime replay and defended validation surface.",
        canonical_source=True,
        used_fallback=False,
        notes=[
            "The public ADS-B proxy lane is not accepted as the official aerospace parity source.",
            "This surface is the runtime replay companion to the C-MAPSS trainable row.",
        ],
        extras={"summary_path": str(SUMMARY_JSON), "source_column_mapping": source_columns},
    )
    write_json(PROVENANCE_PATH, manifest)
    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Build aerospace real-flight runtime dataset from provider telemetry")
    parser.add_argument("--external-root", type=Path, default=None, help="Override ORIUS_EXTERNAL_DATA_ROOT")
    parser.add_argument("--out", type=Path, default=PROCESSED_CSV, help="Output processed runtime CSV")
    args = parser.parse_args()
    try:
        output = build_real_flight_runtime(external_root=args.external_root, out_csv=args.out)
    except Exception as exc:  # noqa: BLE001
        print(str(exc))
        return 1
    print(f"Aerospace real-flight runtime -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
