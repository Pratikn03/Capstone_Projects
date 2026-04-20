#!/usr/bin/env python3
"""Download MIMIC-III matched waveform numerics and bridge to ORIUS healthcare format.

Downloads a configurable number of ICU patient numerics records from the
open-access MIMIC-III Waveform Database Matched Subset on PhysioNet,
extracts critical-event episodes (desaturation, bradycardia, tachycardia,
tachypnea), and produces an ORIUS-contract CSV with reliability scores.

Usage:
    python scripts/build_physionet_healthcare_bridge.py
    python scripts/build_physionet_healthcare_bridge.py --n-patients 50
    python scripts/build_physionet_healthcare_bridge.py --n-patients 100 --out-dir data/healthcare/mimic3

Prerequisites:
    pip install wfdb
    Internet access to PhysioNet (no credentials needed for this dataset).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError:
    sys.exit("wfdb not installed. Run: pip install wfdb")

REPO_ROOT = Path(__file__).resolve().parents[1]
DB = "mimic3wdb-matched/1.0"

# ORIUS healthcare safety bounds
SPO2_MIN = 90.0
HR_MIN = 40.0
HR_MAX = 150.0
RESP_MIN = 5.0
RESP_MAX = 35.0

# Critical-event thresholds (stricter than safe-set, for labelling)
DESAT_SPO2 = 92.0           # SpO2 < 92 % → desaturation event
BRADY_HR = 50.0             # HR < 50 → bradycardia
TACHY_HR = 120.0            # HR > 120 → tachycardia
TACHYPNEA_RESP = 25.0       # RR > 25 → tachypnea
APNEA_RESP = 5.0            # RR < 5 → apnea/bradypnea

# Signals we want from the numerics records
TARGET_SIGNALS = {"SpO2", "HR", "RESP", "PULSE", "ABPMean", "ABPSys", "ABPDias"}

# EMA smoothing for forecast
EMA_ALPHA = 0.3


def _find_numerics_record(patient_dir: str) -> str | None:
    """Find the numerics record (suffix 'n') for a patient directory."""
    try:
        sub_recs = wfdb.get_record_list(DB + "/" + patient_dir.rstrip("/"))
    except Exception:
        return None

    for rec in sub_recs:
        if rec.endswith("n"):
            return rec
    return None


MAX_SAMPLES = 5000  # Cap per patient to keep downloads fast


def _download_numerics(patient_dir: str, rec_name: str) -> pd.DataFrame | None:
    """Download a single numerics record and return as DataFrame.

    Reads the header first to check signals; caps at MAX_SAMPLES.
    """
    pn_dir = f"{DB}/{patient_dir}"
    try:
        hdr = wfdb.rdheader(rec_name, pn_dir=pn_dir)
    except Exception:
        return None

    # Must have SpO2
    if "SpO2" not in hdr.sig_name:
        return None

    # Cap download length
    sampto = min(hdr.sig_len, MAX_SAMPLES)

    try:
        record = wfdb.rdrecord(rec_name, pn_dir=pn_dir, sampto=sampto)
    except Exception:
        return None

    if record.p_signal is None or len(record.sig_name) == 0:
        return None

    df = pd.DataFrame(record.p_signal, columns=record.sig_name)

    # Generate timestamps from sample frequency
    n = len(df)
    fs = record.fs if record.fs and record.fs > 0 else 1.0 / 60.0
    dt_sec = 1.0 / fs
    df["time_sec"] = np.arange(n) * dt_sec

    return df


def _compute_reliability(spo2: np.ndarray, hr: np.ndarray, resp: np.ndarray) -> np.ndarray:
    """Compute per-step reliability score from signal quality heuristics.

    Reliability is degraded by:
    - NaN / missing values
    - Physiologically implausible ranges
    - HR-Pulse disagreement (if available)
    - Rapid signal changes (artifact-like)
    """
    n = len(spo2)
    w = np.ones(n)

    # Penalise NaN
    w[np.isnan(spo2)] *= 0.1
    w[np.isnan(hr)] *= 0.3
    w[np.isnan(resp)] *= 0.3

    # Penalise implausible ranges
    spo2_clean = np.nan_to_num(spo2, nan=95.0)
    hr_clean = np.nan_to_num(hr, nan=75.0)
    resp_clean = np.nan_to_num(resp, nan=15.0)

    w[(spo2_clean < 50) | (spo2_clean > 100)] *= 0.2
    w[(hr_clean < 20) | (hr_clean > 250)] *= 0.2
    w[(resp_clean < 0) | (resp_clean > 60)] *= 0.2

    # Penalise large jumps (artifacts)
    if n > 1:
        dspo2 = np.abs(np.diff(spo2_clean, prepend=spo2_clean[0]))
        dhr = np.abs(np.diff(hr_clean, prepend=hr_clean[0]))
        w[dspo2 > 10] *= 0.5
        w[dhr > 30] *= 0.5

    return np.clip(w, 0.05, 1.0)


def _ema_forecast(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """Exponential moving average forecast (one-step-ahead)."""
    n = len(values)
    forecast = np.empty(n)
    clean = np.nan_to_num(values, nan=np.nanmean(values) if np.any(~np.isnan(values)) else 0.0)
    forecast[0] = clean[0]
    for i in range(1, n):
        forecast[i] = alpha * clean[i - 1] + (1 - alpha) * forecast[i - 1]
    return forecast


def _label_critical_events(
    spo2: np.ndarray, hr: np.ndarray, resp: np.ndarray
) -> dict[str, Any]:
    """Label critical events in a patient's time series."""
    labels: dict[str, Any] = {
        "has_desaturation": False,
        "has_bradycardia": False,
        "has_tachycardia": False,
        "has_tachypnea": False,
        "has_apnea": False,
        "is_critical": False,
        "n_desat_steps": 0,
        "n_brady_steps": 0,
        "n_tachy_steps": 0,
        "n_tachypnea_steps": 0,
        "n_apnea_steps": 0,
        "min_spo2": float("nan"),
        "max_hr": float("nan"),
        "min_hr": float("nan"),
        "max_resp": float("nan"),
    }

    spo2_c = np.nan_to_num(spo2, nan=95.0)
    hr_c = np.nan_to_num(hr, nan=75.0)
    resp_c = np.nan_to_num(resp, nan=15.0)

    # Desaturation
    desat_mask = spo2_c < DESAT_SPO2
    labels["n_desat_steps"] = int(desat_mask.sum())
    labels["has_desaturation"] = labels["n_desat_steps"] > 0
    labels["min_spo2"] = float(np.nanmin(spo2)) if np.any(~np.isnan(spo2)) else float("nan")

    # Bradycardia
    brady_mask = hr_c < BRADY_HR
    labels["n_brady_steps"] = int(brady_mask.sum())
    labels["has_bradycardia"] = labels["n_brady_steps"] > 0
    labels["min_hr"] = float(np.nanmin(hr)) if np.any(~np.isnan(hr)) else float("nan")

    # Tachycardia
    tachy_mask = hr_c > TACHY_HR
    labels["n_tachy_steps"] = int(tachy_mask.sum())
    labels["has_tachycardia"] = labels["n_tachy_steps"] > 0
    labels["max_hr"] = float(np.nanmax(hr)) if np.any(~np.isnan(hr)) else float("nan")

    # Tachypnea
    tpnea_mask = resp_c > TACHYPNEA_RESP
    labels["n_tachypnea_steps"] = int(tpnea_mask.sum())
    labels["has_tachypnea"] = labels["n_tachypnea_steps"] > 0
    labels["max_resp"] = float(np.nanmax(resp)) if np.any(~np.isnan(resp)) else float("nan")

    # Apnea / bradypnea
    apnea_mask = resp_c < APNEA_RESP
    labels["n_apnea_steps"] = int(apnea_mask.sum())
    labels["has_apnea"] = labels["n_apnea_steps"] > 0

    labels["is_critical"] = any([
        labels["has_desaturation"],
        labels["has_bradycardia"],
        labels["has_tachycardia"],
        labels["has_tachypnea"],
        labels["has_apnea"],
    ])

    return labels


def build_mimic3_bridge(
    n_patients: int = 30,
    out_dir: Path | None = None,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Download MIMIC-III numerics and bridge to ORIUS.

    Returns (orius_csv_path, manifest_path).
    """
    if out_dir is None:
        out_dir = REPO_ROOT / "data" / "healthcare" / "mimic3" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    print(f"Fetching MIMIC-III record list from PhysioNet ...")
    all_patients = wfdb.get_record_list(DB)
    print(f"  {len(all_patients)} patient directories available")

    # Shuffle and try up to 3x patients to get n_patients with good data
    patient_indices = rng.permutation(len(all_patients))

    all_rows: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    patients_processed = 0
    patients_tried = 0

    for idx in patient_indices:
        if patients_processed >= n_patients:
            break
        if patients_tried >= n_patients * 3:
            print(f"  Tried {patients_tried} patients, stopping with {patients_processed}")
            break

        patient_dir = all_patients[idx]
        patients_tried += 1

        # Find numerics record
        rec_name = _find_numerics_record(patient_dir)
        if rec_name is None:
            continue

        # Download
        try:
            df = _download_numerics(patient_dir, rec_name)
        except Exception:
            continue
        if df is None or len(df) < 30:
            continue

        # Extract signals
        sig_map = {"SpO2": None, "HR": None, "RESP": None, "PULSE": None}
        for sig in sig_map:
            if sig in df.columns:
                sig_map[sig] = df[sig].values

        spo2 = sig_map["SpO2"]
        hr = sig_map["HR"]
        resp = sig_map["RESP"]
        pulse = sig_map["PULSE"]

        # Must have SpO2 and at least one of HR/PULSE
        if spo2 is None:
            continue
        if hr is None and pulse is None:
            continue
        if hr is None:
            hr = pulse  # Fall back to pulse rate
        if resp is None:
            resp = np.full(len(spo2), np.nan)

        n = len(spo2)
        patients_processed += 1
        patient_id = patient_dir.rstrip("/").split("/")[-1]
        print(f"  [{patients_processed}/{n_patients}] {patient_id}: "
              f"{n} samples, SpO2 range [{np.nanmin(spo2):.0f}, {np.nanmax(spo2):.0f}]%")

        # Compute reliability
        reliability = _compute_reliability(spo2, hr, resp)

        # Compute EMA forecast
        spo2_forecast = _ema_forecast(spo2)

        # Label critical events
        labels = _label_critical_events(spo2, hr, resp)

        tag = "CRITICAL" if labels["is_critical"] else "normal"
        events = []
        if labels["has_desaturation"]: events.append(f"desat({labels['n_desat_steps']})")
        if labels["has_bradycardia"]: events.append(f"brady({labels['n_brady_steps']})")
        if labels["has_tachycardia"]: events.append(f"tachy({labels['n_tachy_steps']})")
        if labels["has_tachypnea"]: events.append(f"tachypnea({labels['n_tachypnea_steps']})")
        if labels["has_apnea"]: events.append(f"apnea({labels['n_apnea_steps']})")
        print(f"    {tag}  {', '.join(events) if events else 'no events'}")

        manifest_entries.append({
            "patient_id": patient_id,
            "patient_dir": patient_dir,
            "record": rec_name,
            "n_samples": n,
            "duration_hours": float(df["time_sec"].iloc[-1] / 3600) if "time_sec" in df.columns else None,
            **labels,
        })

        # Build ORIUS rows
        spo2_clean = np.nan_to_num(spo2, nan=95.0)
        hr_clean = np.nan_to_num(hr, nan=75.0)
        resp_clean = np.nan_to_num(resp, nan=15.0)

        for step in range(n):
            all_rows.append({
                "timestamp": f"{patient_id}_t{step}",
                "target": float(spo2_clean[step]),
                "forecast": float(spo2_forecast[step]),
                "reliability": float(reliability[step]),
                "hr": float(hr_clean[step]),
                "pulse": float(pulse[step]) if pulse is not None and step < len(pulse) else float(hr_clean[step]),
                "resp": float(resp_clean[step]),
                "patient_id": patient_id,
                "domain_label": "healthcare",
                "is_critical": labels["is_critical"],
            })

    # Write outputs
    if not all_rows:
        print("WARNING: No patient data collected!")
        csv_path = out_dir / "mimic3_healthcare_orius.csv"
        manifest_path = out_dir / "mimic3_manifest.json"
        return csv_path, manifest_path

    orius_df = pd.DataFrame(all_rows)
    csv_path = out_dir / "mimic3_healthcare_orius.csv"
    orius_df.to_csv(csv_path, index=False)

    n_critical = sum(1 for m in manifest_entries if m["is_critical"])
    manifest_data = {
        "database": DB,
        "n_patients_downloaded": patients_processed,
        "n_patients_tried": patients_tried,
        "total_rows": len(orius_df),
        "critical_patients": n_critical,
        "normal_patients": patients_processed - n_critical,
        "thresholds": {
            "desaturation_spo2": DESAT_SPO2,
            "bradycardia_hr": BRADY_HR,
            "tachycardia_hr": TACHY_HR,
            "tachypnea_resp": TACHYPNEA_RESP,
            "apnea_resp": APNEA_RESP,
        },
        "patients": manifest_entries,
    }
    manifest_path = out_dir / "mimic3_manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2, default=str) + "\n")

    print(f"\n{'=' * 70}")
    print(f"MIMIC-III → ORIUS Healthcare Bridge Complete")
    print(f"{'=' * 70}")
    print(f"  Patients:       {patients_processed} ({n_critical} critical, "
          f"{patients_processed - n_critical} normal)")
    print(f"  Total rows:     {len(orius_df):,}")
    print(f"  Output CSV:     {csv_path}")
    print(f"  Manifest:       {manifest_path}")
    print(f"{'=' * 70}")

    return csv_path, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge MIMIC-III numerics to ORIUS healthcare")
    parser.add_argument("--n-patients", type=int, default=30,
                        help="Number of patients to download (default: 30)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path, manifest_path = build_mimic3_bridge(
        n_patients=args.n_patients,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
