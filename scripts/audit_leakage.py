"""Temporal and feature leakage audit for publish gating."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.release.artifact_loader import load_pickle_artifact

DATASETS = {
    "DE": {
        "splits_dir": Path("data/processed/splits"),
        "features_path": Path("data/processed/features.parquet"),
        "models_dir": Path("artifacts/models"),
    },
    "US": {
        "splits_dir": Path("data/processed/us_eia930/splits"),
        "features_path": Path("data/processed/us_eia930/features.parquet"),
        "models_dir": Path("artifacts/models_eia930"),
    },
    "HEALTHCARE": {
        "splits_dir": Path("data/healthcare/processed/splits"),
        "features_path": Path("data/healthcare/processed/features.parquet"),
        "models_dir": Path("artifacts/models_healthcare"),
    },
}


def _load_publish_cfg(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("publish_audit", {}) if isinstance(payload, dict) else {}


def _to_ts_series(df: pd.DataFrame) -> pd.Series:
    if "timestamp" not in df.columns:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dropna()


def _read_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _pair_overlap(a: pd.Series, b: pd.Series) -> int:
    if a.empty or b.empty:
        return 0
    return int(len(set(a.astype(str)) & set(b.astype(str))))


def _id_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="object")
    return df[column].dropna().astype(str)


def _boundaries_ok(
    train_ts: pd.Series,
    calibration_ts: pd.Series,
    val_ts: pd.Series,
    test_ts: pd.Series,
) -> dict[str, Any]:
    def _max(ts: pd.Series) -> str | None:
        return ts.max().isoformat() if not ts.empty else None

    def _min(ts: pd.Series) -> str | None:
        return ts.min().isoformat() if not ts.empty else None

    train_max = _max(train_ts)
    cal_min = _min(calibration_ts)
    cal_max = _max(calibration_ts)
    val_min = _min(val_ts)
    val_max = _max(val_ts)
    test_min = _min(test_ts)

    train_cal_ok = (train_max is None) or (cal_min is None) or (train_max < cal_min)
    cal_val_ok = (cal_max is None) or (val_min is None) or (cal_max < val_min)
    train_val_ok = (train_max is None) or (val_min is None) or (train_max < val_min)
    train_test_ok = (train_max is None) or (test_min is None) or (train_max < test_min)
    cal_test_ok = (cal_max is None) or (test_min is None) or (cal_max < test_min)
    val_test_ok = (val_max is None) or (test_min is None) or (val_max < test_min)

    return {
        "train_max": train_max,
        "calibration_min": cal_min,
        "calibration_max": cal_max,
        "val_min": val_min,
        "val_max": val_max,
        "test_min": test_min,
        "train_calibration_ok": bool(train_cal_ok),
        "calibration_val_ok": bool(cal_val_ok),
        "train_val_ok": bool(train_val_ok),
        "train_test_ok": bool(train_test_ok),
        "calibration_test_ok": bool(cal_test_ok),
        "val_test_ok": bool(val_test_ok),
    }


def _scan_feature_names(
    features_path: Path, forbidden_exact: set[str], forbidden_patterns: list[re.Pattern[str]]
) -> dict[str, Any]:
    out = {"path": str(features_path), "forbidden_exact_hits": [], "forbidden_pattern_hits": []}
    if not features_path.exists():
        out["missing"] = True
        return out
    df = pd.read_parquet(features_path)
    columns = [str(c) for c in df.columns]
    out["forbidden_exact_hits"] = sorted([c for c in columns if c in forbidden_exact])
    pattern_hits: list[str] = []
    for col in columns:
        for pat in forbidden_patterns:
            if pat.search(col):
                pattern_hits.append(col)
                break
    out["forbidden_pattern_hits"] = sorted(set(pattern_hits))
    return out


def _scan_model_artifact_features(
    models_dir: Path, forbidden_exact: set[str], forbidden_patterns: list[re.Pattern[str]]
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not models_dir.exists():
        return findings
    for artifact in sorted(models_dir.glob("*.pkl")):
        try:
            bundle = load_pickle_artifact(artifact)
        except Exception:
            continue
        if not isinstance(bundle, dict):
            continue
        feature_cols = bundle.get("feature_cols", [])
        if not isinstance(feature_cols, list):
            continue
        cols = [str(c) for c in feature_cols]
        exact_hits = sorted([c for c in cols if c in forbidden_exact])
        pattern_hits = sorted({c for c in cols for pat in forbidden_patterns if pat.search(c)})
        if exact_hits or pattern_hits:
            findings.append(
                {
                    "artifact": str(artifact),
                    "forbidden_exact_hits": exact_hits,
                    "forbidden_pattern_hits": pattern_hits,
                }
            )
    return findings


def run_leakage_audit(*, config_path: Path) -> dict[str, Any]:
    cfg = _load_publish_cfg(config_path)
    gates = cfg.get("leakage_gates", {}) if isinstance(cfg.get("leakage_gates"), dict) else {}

    max_overlap_train_val = int(gates.get("max_overlap_train_val", 0))
    max_overlap_train_calibration = int(gates.get("max_overlap_train_calibration", 0))
    max_overlap_calibration_val = int(gates.get("max_overlap_calibration_val", 0))
    max_overlap_calibration_test = int(gates.get("max_overlap_calibration_test", 0))
    max_overlap_train_test = int(gates.get("max_overlap_train_test", 0))
    max_overlap_val_test = int(gates.get("max_overlap_val_test", 0))
    forbidden_exact = {str(x) for x in gates.get("forbidden_feature_exact", [])}
    forbidden_patterns = [re.compile(str(p)) for p in gates.get("forbidden_feature_patterns", [])]

    datasets_out: dict[str, Any] = {}
    violations: list[dict[str, Any]] = []

    for name, dcfg in DATASETS.items():
        train_df = _read_split(dcfg["splits_dir"] / "train.parquet")
        calibration_df = _read_split(dcfg["splits_dir"] / "calibration.parquet")
        val_df = _read_split(dcfg["splits_dir"] / "val.parquet")
        test_df = _read_split(dcfg["splits_dir"] / "test.parquet")

        train_ts = _to_ts_series(train_df)
        calibration_ts = _to_ts_series(calibration_df)
        val_ts = _to_ts_series(val_df)
        test_ts = _to_ts_series(test_df)
        train_patients = _id_series(train_df, "patient_id")
        calibration_patients = _id_series(calibration_df, "patient_id")
        val_patients = _id_series(val_df, "patient_id")
        test_patients = _id_series(test_df, "patient_id")

        overlap_train_calibration = _pair_overlap(train_ts, calibration_ts)
        overlap_calibration_val = _pair_overlap(calibration_ts, val_ts)
        overlap_calibration_test = _pair_overlap(calibration_ts, test_ts)
        overlap_train_val = _pair_overlap(train_ts, val_ts)
        overlap_train_test = _pair_overlap(train_ts, test_ts)
        overlap_val_test = _pair_overlap(val_ts, test_ts)
        patient_overlap_train_calibration = _pair_overlap(train_patients, calibration_patients)
        patient_overlap_calibration_val = _pair_overlap(calibration_patients, val_patients)
        patient_overlap_calibration_test = _pair_overlap(calibration_patients, test_patients)
        patient_overlap_train_val = _pair_overlap(train_patients, val_patients)
        patient_overlap_train_test = _pair_overlap(train_patients, test_patients)
        patient_overlap_val_test = _pair_overlap(val_patients, test_patients)

        boundaries = _boundaries_ok(train_ts, calibration_ts, val_ts, test_ts)
        feature_scan = _scan_feature_names(dcfg["features_path"], forbidden_exact, forbidden_patterns)
        model_feature_hits = _scan_model_artifact_features(
            dcfg["models_dir"], forbidden_exact, forbidden_patterns
        )

        dataset_fail = False
        if overlap_train_calibration > max_overlap_train_calibration:
            dataset_fail = True
            violations.append(
                {"dataset": name, "type": "overlap_train_calibration", "value": overlap_train_calibration}
            )
        if patient_overlap_train_calibration > max_overlap_train_calibration:
            dataset_fail = True
            violations.append(
                {
                    "dataset": name,
                    "type": "patient_overlap_train_calibration",
                    "value": patient_overlap_train_calibration,
                }
            )
        if overlap_calibration_val > max_overlap_calibration_val:
            dataset_fail = True
            violations.append(
                {"dataset": name, "type": "overlap_calibration_val", "value": overlap_calibration_val}
            )
        if patient_overlap_calibration_val > max_overlap_calibration_val:
            dataset_fail = True
            violations.append(
                {
                    "dataset": name,
                    "type": "patient_overlap_calibration_val",
                    "value": patient_overlap_calibration_val,
                }
            )
        if overlap_calibration_test > max_overlap_calibration_test:
            dataset_fail = True
            violations.append(
                {"dataset": name, "type": "overlap_calibration_test", "value": overlap_calibration_test}
            )
        if patient_overlap_calibration_test > max_overlap_calibration_test:
            dataset_fail = True
            violations.append(
                {
                    "dataset": name,
                    "type": "patient_overlap_calibration_test",
                    "value": patient_overlap_calibration_test,
                }
            )
        if overlap_train_val > max_overlap_train_val:
            dataset_fail = True
            violations.append({"dataset": name, "type": "overlap_train_val", "value": overlap_train_val})
        if patient_overlap_train_val > max_overlap_train_val:
            dataset_fail = True
            violations.append(
                {"dataset": name, "type": "patient_overlap_train_val", "value": patient_overlap_train_val}
            )
        if overlap_train_test > max_overlap_train_test:
            dataset_fail = True
            violations.append({"dataset": name, "type": "overlap_train_test", "value": overlap_train_test})
        if patient_overlap_train_test > max_overlap_train_test:
            dataset_fail = True
            violations.append(
                {"dataset": name, "type": "patient_overlap_train_test", "value": patient_overlap_train_test}
            )
        if overlap_val_test > max_overlap_val_test:
            dataset_fail = True
            violations.append({"dataset": name, "type": "overlap_val_test", "value": overlap_val_test})
        if patient_overlap_val_test > max_overlap_val_test:
            dataset_fail = True
            violations.append(
                {"dataset": name, "type": "patient_overlap_val_test", "value": patient_overlap_val_test}
            )
        if (
            not boundaries["train_calibration_ok"]
            or not boundaries["calibration_val_ok"]
            or not boundaries["train_val_ok"]
            or not boundaries["train_test_ok"]
            or not boundaries["calibration_test_ok"]
            or not boundaries["val_test_ok"]
        ):
            dataset_fail = True
            violations.append({"dataset": name, "type": "boundary_order", "value": boundaries})
        if feature_scan.get("forbidden_exact_hits") or feature_scan.get("forbidden_pattern_hits"):
            dataset_fail = True
            violations.append(
                {
                    "dataset": name,
                    "type": "features_forbidden",
                    "value": {
                        "exact": feature_scan.get("forbidden_exact_hits", []),
                        "pattern": feature_scan.get("forbidden_pattern_hits", []),
                    },
                }
            )
        if model_feature_hits:
            dataset_fail = True
            violations.append(
                {"dataset": name, "type": "model_features_forbidden", "value": model_feature_hits}
            )

        datasets_out[name] = {
            "rows": {
                "train": int(len(train_df)),
                "calibration": int(len(calibration_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
            },
            "overlap": {
                "train_calibration": int(overlap_train_calibration),
                "calibration_val": int(overlap_calibration_val),
                "calibration_test": int(overlap_calibration_test),
                "train_val": int(overlap_train_val),
                "train_test": int(overlap_train_test),
                "val_test": int(overlap_val_test),
            },
            "patient_overlap": {
                "train_calibration": int(patient_overlap_train_calibration),
                "calibration_val": int(patient_overlap_calibration_val),
                "calibration_test": int(patient_overlap_calibration_test),
                "train_val": int(patient_overlap_train_val),
                "train_test": int(patient_overlap_train_test),
                "val_test": int(patient_overlap_val_test),
            },
            "boundaries": boundaries,
            "feature_scan": feature_scan,
            "model_feature_hits": model_feature_hits,
            "fail": dataset_fail,
        }

    fail = any(d.get("fail", False) for d in datasets_out.values())
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "gates": {
            "max_overlap_train_calibration": max_overlap_train_calibration,
            "max_overlap_calibration_val": max_overlap_calibration_val,
            "max_overlap_calibration_test": max_overlap_calibration_test,
            "max_overlap_train_val": max_overlap_train_val,
            "max_overlap_train_test": max_overlap_train_test,
            "max_overlap_val_test": max_overlap_val_test,
            "forbidden_feature_exact": sorted(forbidden_exact),
            "forbidden_feature_patterns": [p.pattern for p in forbidden_patterns],
        },
        "datasets": datasets_out,
        "violations": violations,
        "fail": fail,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage audit for splits/features/model artifacts")
    parser.add_argument("--config", default="configs/publish_audit.yaml")
    parser.add_argument("--out-json", default="reports/publish/leakage_audit.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_leakage_audit(config_path=Path(args.config))
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload.get("fail"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
