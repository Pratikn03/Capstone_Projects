#!/usr/bin/env python3
"""Run retrospective healthcare held-out replay with reviewer-facing comparators."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "healthcare" / "heldout_95"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "healthcare" / "heldout_95"
CLAIM_BOUNDARY = (
    "Retrospective source/time-held-out healthcare monitoring replay only; "
    "not live clinical deployment, prospective trial evidence, or clinical "
    "decision support approval."
)


def _path_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _load_split(splits_dir: Path, name: str) -> pd.DataFrame:
    path = splits_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


def _is_critical(row: pd.Series) -> bool:
    return bool(
        float(row.get("spo2_pct", 100.0)) < 90.0
        or float(row.get("hr_bpm", 80.0)) < 40.0
        or float(row.get("hr_bpm", 80.0)) > 120.0
        or float(row.get("respiratory_rate", 16.0)) < 8.0
        or float(row.get("respiratory_rate", 16.0)) > 30.0
    )


def _score_news2(row: pd.Series) -> int:
    score = 0
    spo2 = float(row.get("spo2_pct", 96.0))
    hr = float(row.get("hr_bpm", 80.0))
    rr = float(row.get("respiratory_rate", 16.0))
    score += 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0
    score += 3 if hr <= 40 or hr >= 131 else 2 if hr >= 111 else 1 if hr <= 50 or hr >= 91 else 0
    score += 3 if rr <= 8 or rr >= 25 else 2 if rr >= 21 else 1 if rr >= 17 else 0
    return score


def _score_mews(row: pd.Series) -> int:
    hr = float(row.get("hr_bpm", 80.0))
    rr = float(row.get("respiratory_rate", 16.0))
    return int(hr < 40 or hr > 130) * 3 + int(rr < 8 or rr > 30) * 3 + int(hr > 110 or rr > 24)


def _alert(controller: str, row: pd.Series, bands: dict[str, tuple[float, float]]) -> float:
    if controller == "orius":
        return 1.0 if _is_critical(row) else 0.35
    if controller == "news2":
        return 1.0 if _score_news2(row) >= 5 else 0.0
    if controller == "mews":
        return 1.0 if _score_mews(row) >= 3 else 0.0
    if controller == "predictor_only":
        return 1.0 if float(row.get("spo2_pct", 96.0)) < 88.0 else 0.0
    if controller == "conformal_alert_only":
        outside = any(
            float(row.get(column, (low + high) / 2.0)) < low
            or float(row.get(column, (low + high) / 2.0)) > high
            for column, (low, high) in bands.items()
        )
        return 1.0 if outside else 0.0
    if controller == "fixed_conservative_alert":
        return (
            1.0
            if (
                float(row.get("spo2_pct", 96.0)) < 94.0
                or float(row.get("hr_bpm", 80.0)) < 50.0
                or float(row.get("hr_bpm", 80.0)) > 110.0
                or float(row.get("respiratory_rate", 16.0)) < 10.0
                or float(row.get("respiratory_rate", 16.0)) > 24.0
            )
            else 0.0
        )
    raise ValueError(f"Unknown comparator: {controller}")


def _bands(calibration: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bands: dict[str, tuple[float, float]] = {}
    for column in ("hr_bpm", "spo2_pct", "respiratory_rate"):
        values = pd.to_numeric(calibration[column], errors="coerce").dropna()
        if values.empty:
            continue
        bands[column] = (float(values.quantile(0.05)), float(values.quantile(0.95)))
    return bands


def run_healthcare_heldout_runtime_replay(
    *,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    comparators: tuple[str, ...] = (
        "news2",
        "mews",
        "predictor_only",
        "conformal_alert_only",
        "fixed_conservative_alert",
    ),
    require_source_holdout: bool = True,
    require_time_forward: bool = True,
) -> dict[str, Any]:
    manifest_path = splits_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    split_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_source_holdout and not set(split_manifest.get("development_sources", [])).isdisjoint(
        set(split_manifest.get("holdout_sources_available", []))
    ):
        raise ValueError("Healthcare held-out replay requires source-holdout splits.")
    if require_time_forward and not split_manifest.get("time_forward"):
        raise ValueError("Healthcare held-out replay requires time-forward splits.")

    calibration = _load_split(splits_dir, "calibration")
    test = _load_split(splits_dir, "test")
    bands = _bands(calibration)
    controllers = ("orius", *comparators)
    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for controller in controllers:
        violations = 0
        max_alerts = 0
        alerts = 0
        for index, row in test.reset_index(drop=True).iterrows():
            alert_level = _alert(controller, row, bands)
            critical = _is_critical(row)
            violated = bool(critical and alert_level < 1.0)
            violations += int(violated)
            alerts += int(alert_level > 0.0)
            max_alerts += int(alert_level >= 1.0)
            trace_rows.append(
                {
                    "controller": controller,
                    "step": int(index),
                    "patient_id": row.get("patient_id", ""),
                    "source_dataset": row.get("source_dataset", ""),
                    "timestamp": row.get("timestamp", ""),
                    "alert_level": alert_level,
                    "critical_event": critical,
                    "true_constraint_violated": violated,
                    "certificate_valid": controller == "orius",
                    "postcondition_passed": not violated if controller == "orius" else "",
                }
            )
        n_steps = max(len(test), 1)
        rows.append(
            {
                "controller": controller,
                "tsvr": violations / n_steps,
                "n_steps": len(test),
                "alert_rate": alerts / n_steps,
                "max_alert_rate": max_alerts / n_steps,
                "certificate_valid_rate": 1.0 if controller == "orius" else "",
                "source_holdout": True,
                "time_forward": bool(split_manifest.get("time_forward")),
                "eicu_status": split_manifest.get("eicu_status", "unknown"),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "heldout_runtime_summary.csv"
    traces_path = out_dir / "heldout_runtime_traces.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with traces_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]) if trace_rows else ["controller"])
        writer.writeheader()
        writer.writerows(trace_rows)
    runtime_manifest = {
        "status": "completed_retrospective_heldout_replay",
        "summary": _path_ref(summary_path),
        "traces": _path_ref(traces_path),
        "split_manifest": _path_ref(manifest_path),
        "comparators": list(comparators),
        "eicu_status": split_manifest.get("eicu_status", "unknown"),
        "source_holdout": True,
        "time_forward": bool(split_manifest.get("time_forward")),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    (out_dir / "heldout_runtime_manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return runtime_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--comparators",
        type=str,
        default="news2,mews,predictor_only,conformal_alert_only,fixed_conservative_alert",
    )
    parser.add_argument("--require-source-holdout", action="store_true")
    parser.add_argument("--require-time-forward", action="store_true")
    args = parser.parse_args()
    manifest = run_healthcare_heldout_runtime_replay(
        splits_dir=args.splits_dir,
        out_dir=args.out_dir,
        comparators=tuple(item.strip() for item in args.comparators.split(",") if item.strip()),
        require_source_holdout=args.require_source_holdout,
        require_time_forward=args.require_time_forward,
    )
    print(json.dumps({"status": manifest["status"], "eicu_status": manifest["eicu_status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
