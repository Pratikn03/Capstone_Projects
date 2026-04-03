#!/usr/bin/env python3
"""Build a bounded public-flight aerospace runtime lane from Hugging Face ADS-B data.

This script is intentionally separate from the official aerospace parity-closing
lane. It converts a public ADS-B trajectory corpus into the ORIUS aerospace
runtime contract, records explicit proxy-field provenance, and emits bounded
runtime/governance artifacts that can deepen the aerospace chapter without
changing the official equal-domain gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.certos.runtime import CertOSRuntime
from orius.data_pipeline.external_raw import EXTERNAL_DATA_ROOT_ENV, get_external_dataset_dir
from orius.data_pipeline.real_data_contract import (
    ResolvedRawSource,
    build_provenance_manifest,
    summarize_csv_output,
    summarize_files,
    utc_now_iso,
    write_json,
)
from orius.universal_framework import run_universal_step
from orius.universal_framework.aerospace_adapter import AerospaceDomainAdapter


HF_REPO_ID = "Pathange/tartanaviation-adsb-19k-clean"
HF_LOCAL_DIRNAME = "tartanaviation_adsb_19k_clean"
RAW_DIR = REPO_ROOT / "data" / "aerospace" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "aerospace" / "processed"
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"

PROCESSED_CSV = PROCESSED_DIR / "aerospace_public_adsb_runtime.csv"
PROVENANCE_PATH = RAW_DIR / "public_adsb_proxy_provenance.json"
SUMMARY_JSON = PUBLICATION_DIR / "aerospace_public_flight_runtime_summary.json"
SUMMARY_CSV = PUBLICATION_DIR / "aerospace_public_flight_runtime_summary.csv"
SUMMARY_MD = PUBLICATION_DIR / "aerospace_public_flight_runtime_summary.md"
GOVERNANCE_CSV = PUBLICATION_DIR / "aerospace_public_flight_governance_matrix.csv"
CALIBRATION_CSV = PUBLICATION_DIR / "aerospace_public_flight_calibration_diagnostics.csv"
CANDIDATE_PARITY_CSV = PUBLICATION_DIR / "aerospace_public_flight_candidate_parity.csv"

MAX_BANK_DEG = 30.0
GRAVITY = 9.80665


def _external_dataset_dir(explicit_root: Path | None) -> Path:
    root = get_external_dataset_dir("aerospace_flight_telemetry", explicit_root=explicit_root, required=True)
    if root is None:
        raise FileNotFoundError(
            "External dataset root could not be resolved. "
            f"Set {EXTERNAL_DATA_ROOT_ENV} or pass --external-root."
        )
    return root / HF_LOCAL_DIRNAME


def download_dataset(local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=["*.csv", "*.jsonl", "README.md", ".gitattributes"],
    )
    return local_dir


def _load_adsb_csv(local_dir: Path) -> pd.DataFrame:
    csv_path = local_dir / "tartanaviation_adsb_19k_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing ADS-B CSV at {csv_path}")
    df = pd.read_csv(csv_path)
    timestamp = pd.to_datetime(
        dict(
            year=df["year"],
            month=df["month"],
            day=df["day"],
            hour=df["hour"],
            minute=df["minute"],
            second=np.floor(df["second"]).astype(int),
        ),
        utc=True,
        errors="coerce",
    )
    frac_seconds = (df["second"] - np.floor(df["second"])).fillna(0.0)
    timestamp = timestamp + pd.to_timedelta(frac_seconds, unit="s")
    df["timestamp"] = timestamp
    df = df.dropna(subset=["timestamp", "aircraft_id", "altitude_ft", "ground_speed_kts", "heading_deg"]).copy()
    df = df.sort_values(["aircraft_id", "timestamp"]).reset_index(drop=True)
    return df


def _derive_bank_angle(group: pd.DataFrame) -> np.ndarray:
    if len(group) <= 1:
        return np.zeros(len(group))
    heading_rad = np.unwrap(np.deg2rad(group["heading_deg"].to_numpy(dtype=float)))
    ts_sec = group["timestamp"].astype("int64").to_numpy(dtype=float) / 1e9
    dt = np.diff(ts_sec)
    dt = np.where(np.abs(dt) < 1e-6, 1.0, dt)
    yaw_delta = np.diff(heading_rad)
    yaw_rate = np.divide(yaw_delta, dt, out=np.zeros_like(yaw_delta), where=np.abs(dt) > 1e-6)
    if yaw_rate.size == 0:
        yaw_rate = np.zeros(len(group), dtype=float)
    else:
        yaw_rate = np.concatenate([yaw_rate[:1], yaw_rate])
    speed_mps = group["ground_speed_kts"].to_numpy(dtype=float) * 0.514444
    bank_rad = np.arctan(np.clip(speed_mps * yaw_rate / GRAVITY, -10.0, 10.0))
    return np.clip(np.rad2deg(bank_rad), -MAX_BANK_DEG, MAX_BANK_DEG)


def convert_adsb_to_runtime_frame(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["aircraft_id"].value_counts()
    valid_ids = counts[counts >= 8].index
    runtime = df[df["aircraft_id"].isin(valid_ids)].copy()
    if runtime.empty:
        runtime = df.copy()

    runtime["flight_id"] = runtime["aircraft_id"].astype(str)
    runtime["step"] = runtime.groupby("flight_id").cumcount()
    runtime["altitude_m"] = pd.to_numeric(runtime["altitude_ft"], errors="coerce") * 0.3048
    runtime["airspeed_kt"] = pd.to_numeric(runtime["ground_speed_kts"], errors="coerce")

    bank_segments: list[pd.Series] = []
    for flight_id, group in runtime.groupby("flight_id", sort=False):
        bank = pd.Series(_derive_bank_angle(group), index=group.index, name=flight_id, dtype=float)
        bank_segments.append(bank)
    if bank_segments:
        runtime["bank_angle_deg"] = pd.concat(bank_segments).sort_index().to_numpy()
    else:
        runtime["bank_angle_deg"] = np.zeros(len(runtime), dtype=float)

    counts_per_flight = runtime.groupby("flight_id")["step"].transform("max").clip(lower=1)
    runtime["fuel_remaining_pct"] = (1.0 - (runtime["step"] / counts_per_flight)) * 100.0
    runtime["ts_utc"] = runtime["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    runtime["speed_source"] = "groundspeed_proxy"
    runtime["bank_proxy_kind"] = "heading_rate_turn_proxy"
    runtime["fuel_proxy_kind"] = "normalized_remaining_progress"
    runtime["source_dataset"] = HF_REPO_ID
    runtime["data_age_sec"] = pd.to_numeric(runtime.get("data_age_sec"), errors="coerce")

    cols = [
        "flight_id",
        "step",
        "altitude_m",
        "airspeed_kt",
        "bank_angle_deg",
        "fuel_remaining_pct",
        "ts_utc",
        "latitude",
        "longitude",
        "heading_deg",
        "data_age_sec",
        "speed_source",
        "bank_proxy_kind",
        "fuel_proxy_kind",
        "source_dataset",
    ]
    runtime = runtime[cols].dropna(subset=["altitude_m", "airspeed_kt", "bank_angle_deg"]).reset_index(drop=True)
    return runtime


def _summary_rows(frame: pd.DataFrame) -> tuple[dict[str, object], list[dict[str, object]]]:
    adapter = AerospaceDomainAdapter({"expected_cadence_s": 1.0})
    constraints = {"v_min_kt": 60.0, "v_max_kt": 350.0, "max_bank_deg": MAX_BANK_DEG}
    sample = frame.groupby("flight_id", sort=False).head(48).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    intervention_count = 0
    contract_pass_count = 0
    fallback_count = 0
    baseline_bank_violations = 0
    safe_bank_violations = 0

    for record in sample.to_dict(orient="records"):
        telemetry = {
            "altitude_m": float(record["altitude_m"]),
            "airspeed_kt": float(record["airspeed_kt"]),
            "bank_angle_deg": float(record["bank_angle_deg"]),
            "fuel_remaining_pct": float(record["fuel_remaining_pct"]),
            "ts_utc": record["ts_utc"],
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
        contract_passed = bool(contract_checks.get("contract_passed", False))
        intervention_reason = safe_action.get("intervention_reason", "")
        candidate_bank = float(candidate["bank_deg"])
        safe_bank = float(safe_action.get("bank_deg", candidate_bank) or candidate_bank)
        baseline_violation = abs(candidate_bank) > MAX_BANK_DEG
        safe_violation = abs(safe_bank) > MAX_BANK_DEG

        baseline_bank_violations += int(baseline_violation)
        safe_bank_violations += int(safe_violation)
        contract_pass_count += int(contract_passed)
        intervention_count += int(abs(candidate_bank - safe_bank) > 1e-6)
        fallback_count += int(str(intervention_reason).startswith("fallback"))

        rows.append(
            {
                "flight_id": record["flight_id"],
                "step": int(record["step"]),
                "candidate_bank_deg": round(candidate_bank, 3),
                "safe_bank_deg": round(safe_bank, 3),
                "contract_passed": contract_passed,
                "intervention_reason": intervention_reason,
                "baseline_bank_violation": baseline_violation,
                "safe_bank_violation": safe_violation,
            }
        )

    summary = {
        "generated_at_utc": utc_now_iso(),
        "repo_id": HF_REPO_ID,
        "rows_total": int(len(frame)),
        "sampled_rows": int(len(sample)),
        "flight_count": int(frame["flight_id"].nunique()),
        "sampled_flight_count": int(sample["flight_id"].nunique()),
        "contract_pass_rate": round(contract_pass_count / max(len(sample), 1), 4),
        "intervention_rate": round(intervention_count / max(len(sample), 1), 4),
        "fallback_rate": round(fallback_count / max(len(sample), 1), 4),
        "baseline_bank_violation_rate": round(baseline_bank_violations / max(len(sample), 1), 4),
        "safe_bank_violation_rate": round(safe_bank_violations / max(len(sample), 1), 4),
        "mean_airspeed_kt": round(float(frame["airspeed_kt"].mean()), 3),
        "mean_altitude_m": round(float(frame["altitude_m"].mean()), 3),
        "mean_abs_bank_deg": round(float(frame["bank_angle_deg"].abs().mean()), 3),
        "proxy_fields": ["bank_angle_deg", "fuel_remaining_pct"],
        "lane_type": "bounded_public_adsb_runtime",
        "canonical_eligibility": False,
    }
    return summary, rows


def _write_runtime_artifacts(summary: dict[str, object], runtime_rows: list[dict[str, object]], frame: pd.DataFrame) -> None:
    PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)
    write_json(SUMMARY_JSON, summary)

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    _write_text = (
        "# Aerospace Public-Flight Runtime Summary\n\n"
        f"- Source: `{HF_REPO_ID}`\n"
        f"- Rows: `{summary['rows_total']}` across `{summary['flight_count']}` flights\n"
        f"- Sampled rows for runtime checks: `{summary['sampled_rows']}`\n"
        f"- Contract pass rate: `{summary['contract_pass_rate']}`\n"
        f"- Intervention rate: `{summary['intervention_rate']}`\n"
        f"- Baseline bank violation rate: `{summary['baseline_bank_violation_rate']}`\n"
        f"- Safe bank violation rate: `{summary['safe_bank_violation_rate']}`\n\n"
        "This artifact is a bounded public-flight support lane only. It does not promote the official aerospace parity gate.\n"
    )
    SUMMARY_MD.write_text(_write_text, encoding="utf-8")

    governance = CertOSRuntime()
    last_state = None
    for row in runtime_rows[:3]:
        proposed = {"bank_deg": row["candidate_bank_deg"], "throttle": 0.7}
        safe = {"bank_deg": row["safe_bank_deg"], "throttle": 0.7}
        governance.validate_and_step(100.0, proposed, safe, 8)
        governance.validate_and_step(100.0, proposed, safe, 2)
        last_state = governance.validate_and_step(100.0, proposed, safe, 0)

    ops = [entry["op"] for entry in governance.raw_audit_log]
    governance_rows = [
        {
            "lane": "aerospace_public_flight",
            "issue_seen": "ISSUE" in ops,
            "validate_seen": "VALIDATE" in ops,
            "expire_seen": "EXPIRE" in ops,
            "fallback_seen": "FALLBACK" in ops,
            "hash_chain_ok": bool(last_state.hash_chain_ok if last_state is not None else False),
            "audit_events": len(governance.raw_audit_log),
            "status": "evaluated",
        }
    ]
    with GOVERNANCE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(governance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(governance_rows)

    diag = frame.copy()
    diag["speed_bucket"] = pd.cut(diag["airspeed_kt"], bins=[0, 120, 240, 360, 600], include_lowest=True)
    diag["altitude_bucket"] = pd.cut(diag["altitude_m"], bins=[0, 1500, 6000, 12000, 20000], include_lowest=True)
    diag_rows = (
        diag.groupby(["speed_bucket", "altitude_bucket"], observed=False)
        .agg(
            rows=("flight_id", "size"),
            flights=("flight_id", "nunique"),
            mean_abs_bank_deg=("bank_angle_deg", lambda s: round(float(s.abs().mean()), 3)),
            mean_data_age_sec=("data_age_sec", lambda s: round(float(s.fillna(0.0).mean()), 3)),
        )
        .reset_index()
    )
    diag_rows["calibration_boundary"] = "runtime_only_proxy_lane"
    diag_rows["formal_calibration_source"] = "C-MAPSS_trainable_surface"
    diag_rows.to_csv(CALIBRATION_CSV, index=False)

    candidate_rows = [
        {
            "target_tier": "public_flight_93_candidate",
            "domain": "aerospace_public_flight",
            "status": "bounded_support_only",
            "rows": summary["rows_total"],
            "contract_pass_rate": summary["contract_pass_rate"],
            "intervention_rate": summary["intervention_rate"],
            "canonical_eligibility": "no",
            "exact_limit": "Public ADS-B runtime evidence deepens the aerospace chapter but cannot promote the official equal-domain aerospace row.",
        }
    ]
    with CANDIDATE_PARITY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(candidate_rows)


def build_public_runtime(
    *,
    external_root: Path | None,
    out_csv: Path,
    download: bool,
) -> Path:
    local_dir = _external_dataset_dir(external_root)
    if download or not (local_dir / "tartanaviation_adsb_19k_clean.csv").exists():
        download_dataset(local_dir)

    df = _load_adsb_csv(local_dir)
    runtime = convert_adsb_to_runtime_frame(df)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    runtime.to_csv(out_csv, index=False)

    raw_source = ResolvedRawSource(
        path=local_dir,
        source_kind="external",
        checked_locations=(str(local_dir),),
    )
    output_summary = summarize_csv_output(out_csv)
    manifest = build_provenance_manifest(
        domain="aerospace",
        dataset_key="public_adsb_proxy",
        provider="Hugging Face public ADS-B mirror",
        version=HF_REPO_ID,
        raw_source=raw_source,
        processed_output=out_csv,
        output_summary=output_summary,
        raw_inventory=summarize_files(local_dir),
        source_urls=[f"https://huggingface.co/datasets/{HF_REPO_ID}"],
        license_notes="MIT license on the mirrored ADS-B dataset; bounded proxy-runtime use only.",
        access_notes=(
            "This is a bounded public-flight runtime support lane. It is not the official provider-approved "
            "multi-flight telemetry source required for equal-domain aerospace promotion."
        ),
        canonical_source=False,
        used_fallback=True,
        notes=[
            "airspeed_kt is populated from ADS-B groundspeed and should be interpreted as a proxy rather than direct measured airspeed",
            "bank_angle_deg is derived from heading-rate and speed rather than measured directly",
            "fuel_remaining_pct is a normalized remaining-flight-progress proxy",
            "public ADS-B support deepens runtime evidence only and cannot promote the official aerospace parity gate",
        ],
        extras={
            "proxy_fields": {
                "airspeed_kt": "populated_from_ground_speed_kts",
                "bank_angle_deg": "derived_from_heading_rate_and_ground_speed",
                "fuel_remaining_pct": "derived_from_remaining_flight_progress",
            },
            "lane_type": "bounded_public_adsb_runtime",
            "source_env": EXTERNAL_DATA_ROOT_ENV,
        },
    )
    write_json(PROVENANCE_PATH, manifest)

    summary, runtime_rows = _summary_rows(runtime)
    _write_runtime_artifacts(summary, runtime_rows, runtime)
    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bounded public-flight aerospace runtime artifacts")
    parser.add_argument("--external-root", type=Path, default=None, help="Override ORIUS_EXTERNAL_DATA_ROOT")
    parser.add_argument("--out", type=Path, default=PROCESSED_CSV, help="Processed public-flight runtime CSV output")
    parser.add_argument("--download", action="store_true", help="Force a fresh Hugging Face snapshot download")
    args = parser.parse_args()

    try:
        output = build_public_runtime(
            external_root=args.external_root,
            out_csv=args.out,
            download=args.download,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 1

    print(f"Aerospace public ADS-B runtime -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
