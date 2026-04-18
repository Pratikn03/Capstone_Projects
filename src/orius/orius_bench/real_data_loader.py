"""Real-dataset loader for ORIUS-Bench tracks.

Provides two loading paths per domain:
  1. Repo-local canonical raw/runtime surfaces staged under ``data/<domain>/``.
  2. Calibrated synthetic fallback used only when those staged surfaces are absent.

Canonical repo-local surfaces
-----------------------------
Industrial raw (UCI CCPP):
    data/industrial/raw/CCPP.csv

Healthcare raw (PhysioNet BIDMC numerics):
    data/healthcare/raw/bidmc_csv/*_Numerics.csv

Vehicle runtime:
    data/av/processed/av_trajectories_orius.csv

Industrial runtime:
    data/industrial/processed/industrial_orius.csv

Healthcare runtime:
    data/healthcare/processed/healthcare_orius.csv

Navigation runtime:
    data/navigation/processed/navigation_orius.csv

Aerospace support/runtime lane:
    data/aerospace/processed/aerospace_public_adsb_runtime.csv

Aerospace defended runtime lane:
    data/aerospace/processed/aerospace_realflight_runtime.csv
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _dataset_registry import DATASET_REGISTRY, repo_path


def _repo_registry_path(dataset_key: str, attr: str) -> Path:
    relative = getattr(DATASET_REGISTRY[dataset_key], attr)
    path = repo_path(relative)
    if path is None:
        raise ValueError(f"Dataset registry entry {dataset_key}.{attr} is not configured.")
    return path


CCPP_PATH = _repo_registry_path("INDUSTRIAL", "canonical_raw_source_path")
BIDMC_PATH = _repo_registry_path("HEALTHCARE", "canonical_raw_source_path")
BIDMC_SYNTHETIC_PATH = BIDMC_PATH / "_synthetic_bidmc_vitals.csv"
AV_PATH = _repo_registry_path("AV", "canonical_runtime_path")
INDUSTRIAL_RUNTIME_PATH = _repo_registry_path("INDUSTRIAL", "canonical_runtime_path")
HEALTHCARE_RUNTIME_PATH = _repo_registry_path("HEALTHCARE", "canonical_runtime_path")
NAVIGATION_PATH = _repo_registry_path("NAVIGATION", "canonical_runtime_path")
AEROSPACE_RUNTIME_PATH = _repo_registry_path("AEROSPACE", "support_runtime_path")
AEROSPACE_REALFLIGHT_PATH = _repo_registry_path("AEROSPACE", "canonical_runtime_path")


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck synthetic generator
# ---------------------------------------------------------------------------

def _ou_series(
    mu: float,
    sigma: float,
    theta: float,
    n: int,
    seed: int,
    clip_lo: float | None = None,
    clip_hi: float | None = None,
    dt: float = 1.0,
) -> list[float]:
    """Generate mean-reverting OU series  dX = θ(μ-X)dt + σ dW."""
    rng = np.random.default_rng(seed)
    x = mu
    out: list[float] = []
    for _ in range(n):
        dw = rng.normal(0.0, math.sqrt(dt))
        x = x + theta * (mu - x) * dt + sigma * dw
        if clip_lo is not None:
            x = max(clip_lo, x)
        if clip_hi is not None:
            x = min(clip_hi, x)
        out.append(float(x))
    return out


# ---------------------------------------------------------------------------
# CCPP loader / generator
# ---------------------------------------------------------------------------

def load_ccpp_rows(path: Path | None = None) -> list[dict[str, float]]:
    """Load UCI CCPP rows from CSV.

    Expected columns: AT, V, AP, RH, PE
    Returns list of dicts with those keys as floats.
    Skips rows that cannot be parsed.
    """
    p = Path(path) if path else CCPP_PATH
    rows: list[dict[str, float]] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append({
                    "AT": float(row["AT"]),
                    "V":  float(row["V"]),
                    "AP": float(row["AP"]),
                    "RH": float(row["RH"]),
                    "PE": float(row["PE"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def generate_ccpp_synthetic(n: int = 9568, seed: int = 42) -> list[dict[str, float]]:
    """Generate calibrated synthetic CCPP data from published OU parameters.

    Correlates AT and PE with a negative relationship (higher temp → lower power)
    matching the published Pearson r ≈ -0.95.
    """
    at_vals = _ou_series(19.65, 4.83, 0.15, n, seed=seed, clip_lo=1.0, clip_hi=38.0)
    ap_vals = _ou_series(1013.26, 5.94, 0.10, n, seed=seed + 1)
    rh_vals = _ou_series(73.31, 14.55, 0.12, n, seed=seed + 2, clip_lo=25.0, clip_hi=100.0)
    v_vals  = _ou_series(54.30, 12.71, 0.20, n, seed=seed + 3, clip_lo=25.0, clip_hi=81.6)
    # PE anticorrelated with AT: pe = mu_pe - 0.85*(at - mu_at) + noise
    rng = np.random.default_rng(seed + 4)
    pe_vals: list[float] = []
    for at in at_vals:
        pe = 454.37 - 0.85 * (at - 19.65) + float(rng.normal(0, 5.0))
        pe = max(420.0, min(496.0, pe))
        pe_vals.append(pe)
    return [
        {"AT": at, "V": v, "AP": ap, "RH": rh, "PE": pe}
        for at, v, ap, rh, pe in zip(at_vals, v_vals, ap_vals, rh_vals, pe_vals)
    ]


def get_ccpp_rows(seed: int = 42) -> list[dict[str, float]]:
    """Return CCPP rows: real data if available, else calibrated synthetic."""
    if CCPP_PATH.exists():
        try:
            rows = load_ccpp_rows()
            if rows:
                return rows
        except Exception:
            pass
    return generate_ccpp_synthetic(seed=seed)


# ---------------------------------------------------------------------------
# BIDMC loader / generator
# ---------------------------------------------------------------------------

def load_bidmc_rows(path: Path | None = None) -> list[dict[str, float]]:
    """Load PhysioNet BIDMC vitals from CSV.

    Accepts either a consolidated CSV file or a directory of ``*_Numerics.csv``
    files from the repo-local BIDMC layout.

    Expected columns: HR, SpO2, RR/RESP  (case-insensitive).
    Rows with NaN/missing values are skipped.
    """
    p = Path(path) if path else BIDMC_PATH
    rows: list[dict[str, float]] = []
    if p.is_dir():
        csv_paths = sorted(p.glob("*_Numerics.csv"))
        if not csv_paths and BIDMC_SYNTHETIC_PATH.exists():
            csv_paths = [BIDMC_SYNTHETIC_PATH]
    else:
        csv_paths = [p]

    for csv_path in csv_paths:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                norm = {k.strip().upper(): v for k, v in row.items() if k}
                try:
                    hr = float(norm["HR"])
                    spo2 = float(norm["SPO2"])
                    rr_raw = norm.get("RR", norm.get("RESP"))
                    if rr_raw is None:
                        continue
                    rr = float(rr_raw)
                    if any(math.isnan(v) for v in (hr, spo2, rr)):
                        continue
                    rows.append({"HR": hr, "SpO2": spo2, "RR": rr})
                except (KeyError, ValueError):
                    continue
    return rows


def generate_bidmc_synthetic(n: int = 4000, seed: int = 42) -> list[dict[str, float]]:
    """Generate calibrated synthetic BIDMC vitals from published OU parameters."""
    hr_vals   = _ou_series(82.0, 18.0, 0.20, n, seed=seed,     clip_lo=30.0, clip_hi=200.0)
    spo2_vals = _ou_series(97.5,  1.2, 0.35, n, seed=seed + 1, clip_lo=70.0, clip_hi=100.0)
    rr_vals   = _ou_series(18.0,  4.0, 0.25, n, seed=seed + 2, clip_lo=4.0,  clip_hi=60.0)
    return [
        {"HR": hr, "SpO2": spo2, "RR": rr}
        for hr, spo2, rr in zip(hr_vals, spo2_vals, rr_vals)
    ]


def get_bidmc_rows(seed: int = 42) -> list[dict[str, float]]:
    """Return BIDMC vitals: real data if available, else calibrated synthetic."""
    if BIDMC_PATH.exists():
        try:
            rows = load_bidmc_rows(BIDMC_PATH)
            if rows:
                return rows
        except Exception:
            pass
    return generate_bidmc_synthetic(seed=seed)


# ---------------------------------------------------------------------------
# Navigation / aerospace runtime loaders
# ---------------------------------------------------------------------------

def load_navigation_rows(path: Path | None = None) -> list[dict[str, float]]:
    """Load processed ORIUS navigation rows."""
    p = Path(path) if path else NAVIGATION_PATH
    rows: list[dict[str, Any]] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append(
                    {
                        "robot_id": row.get("robot_id", row.get("source_sequence", "robot-0")),
                        "step": int(float(row.get("step", 0))),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "vx": float(row["vx"]),
                        "vy": float(row["vy"]),
                        "ts_utc": row.get("ts_utc", ""),
                        "source_sequence": row.get("source_sequence", ""),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def load_aerospace_runtime_rows(path: Path | None = None) -> list[dict[str, float]]:
    """Load processed ORIUS aerospace runtime rows."""
    p = Path(path) if path else AEROSPACE_RUNTIME_PATH
    rows: list[dict[str, Any]] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append(
                    {
                        "flight_id": row.get("flight_id", "flight-0"),
                        "step": int(float(row.get("step", 0))),
                        "altitude_m": float(row["altitude_m"]),
                        "airspeed_kt": float(row["airspeed_kt"]),
                        "bank_angle_deg": float(row["bank_angle_deg"]),
                        "fuel_remaining_pct": float(row["fuel_remaining_pct"]),
                        "ts_utc": row.get("ts_utc", ""),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def load_vehicle_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Load processed AV trajectory rows.

    Supports both Path A (speed-limit-only) and Path B (RSS-augmented)
    schemas.  RSS columns are optional; missing fields default to None.
    """
    p = Path(path) if path else AV_PATH
    rows: list[dict[str, Any]] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                d: dict[str, Any] = {
                    "vehicle_id": row.get("vehicle_id", "veh-0"),
                    "step": int(float(row.get("step", 0))),
                    "position_m": float(row["position_m"]),
                    "speed_mps": float(row["speed_mps"]),
                    "speed_limit_mps": float(row["speed_limit_mps"]),
                    "lead_position_m": float(row["lead_position_m"]),
                    "ts_utc": row.get("ts_utc", ""),
                }
                # Path B RSS columns (optional)
                if "lead_present" in row:
                    d["lead_present"] = row["lead_present"] in ("True", "true", "1")
                if "lead_rel_x_m" in row and row["lead_rel_x_m"]:
                    d["lead_rel_x_m"] = float(row["lead_rel_x_m"])
                if "lead_speed_mps" in row and row["lead_speed_mps"]:
                    d["lead_speed_mps"] = float(row["lead_speed_mps"])
                if "rss_safe_gap_m" in row and row["rss_safe_gap_m"]:
                    d["rss_safe_gap_m"] = float(row["rss_safe_gap_m"])
                if "rss_violation_true" in row:
                    d["rss_violation_true"] = row["rss_violation_true"] in ("True", "true", "1")
                rows.append(d)
            except (KeyError, ValueError):
                continue
    return rows


def load_industrial_runtime_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Load processed industrial rows."""
    p = Path(path) if path else INDUSTRIAL_RUNTIME_PATH
    rows: list[dict[str, Any]] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append(
                    {
                        "sensor_id": row.get("sensor_id", "sensor-0"),
                        "step": int(float(row.get("step", 0))),
                        "temp_c": float(row["temp_c"]),
                        "vacuum_cmhg": float(row.get("vacuum_cmhg", 0.0)),
                        "pressure_mbar": float(row["pressure_mbar"]),
                        "humidity_pct": float(row.get("humidity_pct", 0.0)),
                        "power_mw": float(row["power_mw"]),
                        "ts_utc": row.get("ts_utc", ""),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def load_healthcare_runtime_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Load processed healthcare rows."""
    p = Path(path) if path else HEALTHCARE_RUNTIME_PATH
    rows: list[dict[str, Any]] = []
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rows.append(
                    {
                        "patient_id": row.get("patient_id", "patient-0"),
                        "step": int(float(row.get("step", 0))),
                        "hr_bpm": float(row["hr_bpm"]),
                        "spo2_pct": float(row["spo2_pct"]),
                        "respiratory_rate": float(row["respiratory_rate"]),
                        "ts_utc": row.get("ts_utc", ""),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


# ---------------------------------------------------------------------------
# Dataset-info helper (for prepare_datasets.py)
# ---------------------------------------------------------------------------

def dataset_status() -> dict[str, Any]:
    """Return availability status of each dataset."""
    ccpp_real = False
    ccpp_rows = 0
    if CCPP_PATH.exists():
        try:
            ccpp_rows = len(load_ccpp_rows())
            ccpp_real = ccpp_rows > 0
        except Exception:
            pass

    bidmc_real = False
    bidmc_rows = 0
    if BIDMC_PATH.exists():
        try:
            bidmc_rows = len(load_bidmc_rows(BIDMC_PATH))
            bidmc_real = bidmc_rows > 0
        except Exception:
            pass

    return {
        "ccpp": {
            "path": str(CCPP_PATH),
            "real_data": ccpp_real,
            "rows": ccpp_rows,
            "fallback_rows": 9568,
        },
        "bidmc": {
            "path": str(BIDMC_PATH),
            "real_data": bidmc_real,
            "rows": bidmc_rows,
            "fallback_rows": 4000,
        },
        "navigation": {
            "path": str(NAVIGATION_PATH),
            "real_data": NAVIGATION_PATH.exists(),
        },
        "aerospace_runtime": {
            "path": str(AEROSPACE_REALFLIGHT_PATH),
            "real_data": AEROSPACE_REALFLIGHT_PATH.exists(),
        },
        "aerospace_support_runtime": {
            "path": str(AEROSPACE_RUNTIME_PATH),
            "real_data": AEROSPACE_RUNTIME_PATH.exists(),
        },
    }
