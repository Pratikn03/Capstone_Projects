#!/usr/bin/env python3
"""Build BIDMC/MIMIC/eICU retrospective source-holdout healthcare splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIDMC = REPO_ROOT / "data" / "healthcare" / "processed" / "healthcare_bidmc_orius.csv"
DEFAULT_MIMIC = REPO_ROOT / "data" / "healthcare" / "mimic3" / "processed" / "mimic3_healthcare_orius.csv"
DEFAULT_EICU_ROOT = REPO_ROOT / "data" / "healthcare" / "eicu" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "healthcare" / "heldout_95"
REQUIRED_COLUMNS = ["patient_id", "timestamp", "hr_bpm", "spo2_pct", "respiratory_rate", "source_dataset"]
CLAIM_BOUNDARY = (
    "Retrospective source/site/time-forward healthcare replay only; not live "
    "clinical deployment, prospective trial evidence, clinical decision support "
    "approval, or regulated clinical use."
)


SOURCE_BASE_TIMES = {
    "bidmc": "2026-01-01T00:00:00Z",
    "mimic3": "2026-02-01T00:00:00Z",
    "eicu": "2026-04-01T00:00:00Z",
}


def _path_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _coerce_timestamp(series: pd.Series, frame: pd.DataFrame, source: str) -> pd.Series:
    raw = series.astype(str)
    step = raw.str.extract(r"_t(\d+)$", expand=False)
    if step.notna().sum() == 0:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        if parsed.notna().any():
            return parsed

    step_numeric = pd.to_numeric(step, errors="coerce")
    if step_numeric.isna().all():
        step_numeric = frame.groupby(
            frame.get("patient_id", pd.Series(range(len(frame)))).astype(str)
        ).cumcount()
    else:
        step_numeric = step_numeric.fillna(
            frame.groupby(frame.get("patient_id", pd.Series(range(len(frame)))).astype(str)).cumcount()
        )
    base = pd.Timestamp(SOURCE_BASE_TIMES.get(source, "2026-01-01T00:00:00Z"))
    return base + pd.to_timedelta(step_numeric.astype(int), unit="min")


def _first_existing(frame: pd.DataFrame, names: tuple[str, ...], default: float) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index)


def _read_orius_healthcare(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        for candidate in ("ts_utc", "charttime", "time"):
            if candidate in frame.columns:
                frame["timestamp"] = frame[candidate]
                break
    if "timestamp" not in frame.columns:
        frame["timestamp"] = pd.date_range("2026-01-01", periods=len(frame), freq="s", tz="UTC")
    if "patient_id" not in frame.columns:
        for candidate in ("subject_id", "stay_id", "patientunitstayid", "uniquepid"):
            if candidate in frame.columns:
                frame["patient_id"] = frame[candidate]
                break
    frame["patient_id"] = frame.get("patient_id", pd.Series(range(len(frame)))).astype(str)
    frame["timestamp"] = _coerce_timestamp(frame["timestamp"], frame, source)
    frame["hr_bpm"] = _first_existing(frame, ("hr_bpm", "hr", "heart_rate", "heartrate", "pulse"), 80.0)
    frame["spo2_pct"] = _first_existing(
        frame, ("spo2_pct", "spo2", "sao2", "target", "oxygen_saturation"), 96.0
    )
    frame["respiratory_rate"] = _first_existing(
        frame, ("respiratory_rate", "resp", "rr", "respiration"), 16.0
    )
    frame["source_dataset"] = source
    frame = (
        frame.dropna(subset=["timestamp"])
        .sort_values(["timestamp", "patient_id"], kind="stable")
        .reset_index(drop=True)
    )
    return frame[REQUIRED_COLUMNS]


def _load_eicu_root(eicu_root: Path) -> tuple[pd.DataFrame, str]:
    if not eicu_root.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS), "not_staged"
    candidates = sorted(eicu_root.rglob("*.csv"))
    if not candidates:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), "not_staged"
    frames: list[pd.DataFrame] = []
    for path in candidates[:20]:
        try:
            raw = pd.read_csv(path, nrows=50_000)
        except Exception:
            continue
        columns = {str(col).lower(): col for col in raw.columns}
        patient_col = (
            columns.get("patientunitstayid") or columns.get("patient_id") or columns.get("uniquepid")
        )
        if patient_col is None:
            continue
        hr_col = columns.get("heartrate") or columns.get("hr") or columns.get("hr_bpm")
        spo2_col = columns.get("sao2") or columns.get("spo2") or columns.get("spo2_pct")
        rr_col = (
            columns.get("respiration") or columns.get("respiratoryrate") or columns.get("respiratory_rate")
        )
        if hr_col is None and spo2_col is None and rr_col is None:
            continue
        frame = pd.DataFrame()
        frame["patient_id"] = raw[patient_col].astype(str)
        frame["hr_bpm"] = pd.to_numeric(raw[hr_col], errors="coerce") if hr_col else 80.0
        frame["spo2_pct"] = pd.to_numeric(raw[spo2_col], errors="coerce") if spo2_col else 96.0
        frame["respiratory_rate"] = pd.to_numeric(raw[rr_col], errors="coerce") if rr_col else 16.0
        frame["timestamp"] = pd.date_range("2026-04-01", periods=len(frame), freq="min", tz="UTC")
        frame["source_dataset"] = "eicu"
        frames.append(frame[REQUIRED_COLUMNS])
    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), "not_staged"
    return pd.concat(frames, ignore_index=True), "staged"


def _patient_block_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        empty = frame.copy()
        return empty, empty, empty
    patient_order = (
        frame.groupby("patient_id", dropna=False)["timestamp"]
        .min()
        .sort_values(kind="stable")
        .index.astype(str)
        .tolist()
    )
    n = len(patient_order)
    first = max(1, int(n * 0.70))
    second = max(first + 1, int(n * 0.85)) if n > 2 else first
    train_ids = set(patient_order[:first])
    cal_ids = set(patient_order[first:second])
    val_ids = set(patient_order[second:])
    if not val_ids and cal_ids:
        val_ids.add(cal_ids.pop())
    return (
        frame[frame["patient_id"].astype(str).isin(train_ids)].copy(),
        frame[frame["patient_id"].astype(str).isin(cal_ids)].copy(),
        frame[frame["patient_id"].astype(str).isin(val_ids)].copy(),
    )


def _shift_after(frame: pd.DataFrame, after: pd.Timestamp | None) -> pd.DataFrame:
    if frame.empty or after is None:
        return frame
    frame = frame.copy()
    min_ts = frame["timestamp"].min()
    if pd.notna(min_ts) and min_ts <= after:
        delta = after + pd.Timedelta(days=1) - min_ts
        frame["timestamp"] = frame["timestamp"] + delta
    return frame


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _split_detail(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "split": name,
        "rows": int(len(frame)),
        "patients": int(frame["patient_id"].nunique()) if "patient_id" in frame else 0,
        "sources": "|".join(sorted(frame["source_dataset"].astype(str).unique())) if not frame.empty else "",
        "start": str(frame["timestamp"].min()) if not frame.empty else "",
        "end": str(frame["timestamp"].max()) if not frame.empty else "",
    }


def build_healthcare_source_holdout_surface(
    *,
    bidmc: Path = DEFAULT_BIDMC,
    mimic: Path = DEFAULT_MIMIC,
    eicu_root: Path = DEFAULT_EICU_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    dev_sources: tuple[str, ...] = ("bidmc",),
    holdout_sources: tuple[str, ...] = ("mimic3", "eicu"),
    time_forward: bool = True,
    patient_disjoint: bool = True,
) -> dict[str, Any]:
    sources = {
        "bidmc": _read_orius_healthcare(bidmc, "bidmc"),
        "mimic3": _read_orius_healthcare(mimic, "mimic3"),
    }
    eicu_frame, eicu_status = _load_eicu_root(eicu_root)
    sources["eicu"] = eicu_frame

    dev_frames = [sources[name] for name in dev_sources if name in sources and not sources[name].empty]
    holdout_frames = [
        sources[name] for name in holdout_sources if name in sources and not sources[name].empty
    ]
    if not dev_frames:
        raise FileNotFoundError("No development healthcare source rows were available.")
    if not holdout_frames:
        raise FileNotFoundError("No holdout healthcare source rows were available.")

    dev = pd.concat(dev_frames, ignore_index=True).sort_values(["timestamp", "patient_id"], kind="stable")
    holdout = pd.concat(holdout_frames, ignore_index=True).sort_values(
        ["timestamp", "patient_id"], kind="stable"
    )
    train, calibration, val = _patient_block_split(dev)
    if time_forward:
        val_end = val["timestamp"].max() if not val.empty else dev["timestamp"].max()
        holdout = _shift_after(holdout, val_end)
    test = holdout.reset_index(drop=True)

    split_frames = {
        "train": train.reset_index(drop=True),
        "calibration": calibration.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in split_frames.items():
        _write_frame(frame, out_dir / f"{name}.parquet")

    details = [_split_detail(name, frame) for name, frame in split_frames.items()]
    details_path = out_dir / "split_details.csv"
    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)

    train_patients = (
        set(train["patient_id"].astype(str))
        | set(calibration["patient_id"].astype(str))
        | set(val["patient_id"].astype(str))
    )
    test_patients = set(test["patient_id"].astype(str))
    actual_patient_disjoint = bool(train_patients.isdisjoint(test_patients))
    actual_time_forward = (
        bool(test["timestamp"].min() > val["timestamp"].max()) if not test.empty and not val.empty else False
    )
    manifest = {
        "status": "completed_source_holdout_surface",
        "split_strategy": "development_patient_blocks_plus_later_source_holdout",
        "source_datasets": sorted(name for name, frame in sources.items() if not frame.empty),
        "development_sources": list(dev_sources),
        "holdout_sources_requested": list(holdout_sources),
        "holdout_sources_available": sorted(test["source_dataset"].astype(str).unique().tolist()),
        "eicu_status": eicu_status,
        "patient_disjoint": actual_patient_disjoint,
        "time_forward": actual_time_forward,
        "patient_disjoint_required": bool(patient_disjoint),
        "time_forward_required": bool(time_forward),
        "row_counts": {name: int(len(frame)) for name, frame in split_frames.items()},
        "split_files": {name: _path_ref(out_dir / f"{name}.parquet") for name in split_frames},
        "details": _path_ref(details_path),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bidmc", type=Path, default=DEFAULT_BIDMC)
    parser.add_argument("--mimic", type=Path, default=DEFAULT_MIMIC)
    parser.add_argument("--eicu-root", type=Path, default=DEFAULT_EICU_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dev-sources", type=str, default="bidmc")
    parser.add_argument("--holdout-sources", type=str, default="mimic3,eicu")
    parser.add_argument("--time-forward", action="store_true")
    parser.add_argument("--patient-disjoint", action="store_true")
    args = parser.parse_args()
    manifest = build_healthcare_source_holdout_surface(
        bidmc=args.bidmc,
        mimic=args.mimic,
        eicu_root=args.eicu_root,
        out_dir=args.out_dir,
        dev_sources=tuple(item.strip() for item in args.dev_sources.split(",") if item.strip()),
        holdout_sources=tuple(item.strip() for item in args.holdout_sources.split(",") if item.strip()),
        time_forward=args.time_forward,
        patient_disjoint=args.patient_disjoint,
    )
    print(json.dumps({"status": manifest["status"], "eicu_status": manifest["eicu_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
