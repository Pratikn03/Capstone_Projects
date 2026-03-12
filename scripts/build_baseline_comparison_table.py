#!/usr/bin/env python3
"""Build the promoted six-model forecasting comparison tables.

The publication-facing contract is intentionally conservative:
- point metrics must be present for all thesis headline rows
- uncertainty metrics remain GBM-only unless a model-specific artifact exists
- exported CSV/TeX surfaces render missing values as `---`, never `NaN`

This script can run in two modes:
1. explicit directories (`--de-dir`, `--us-dir`, ...)
2. release-family mode (`--release-id`), which reads accepted candidate
   manifests for DE and US_MISO and rebuilds the table from that single
   release family.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reports" / "publication"

REGION_DEFAULTS = {
    "DE": {
        "metrics_json": REPO_ROOT / "reports" / "week2_metrics.json",
        "uncertainty_dir": REPO_ROOT / "artifacts" / "uncertainty",
    },
    "US": {
        "metrics_json": REPO_ROOT / "reports" / "eia930" / "week2_metrics.json",
        "uncertainty_dir": REPO_ROOT / "artifacts" / "uncertainty/eia930",
    },
}

TARGETS = ["load_mw", "wind_mw", "solar_mw"]
TARGET_LABELS = {"load_mw": "Load", "wind_mw": "Wind", "solar_mw": "Solar"}

MODEL_ORDER = ["gbm", "lstm", "tcn", "nbeats", "tft", "patchtst"]
MODEL_LABELS = {
    "gbm": "GBM",
    "lstm": "LSTM",
    "tcn": "TCN",
    "nbeats": "N-BEATS",
    "tft": "TFT",
    "patchtst": "PatchTST",
}

COLUMNS = [
    "Region",
    "Target",
    "Model",
    "RMSE",
    "MAE",
    "sMAPE (%)",
    "R2",
    "PICP@90 (%)",
    "Interval Width (MW)",
]
POINT_COLUMNS = ["RMSE", "MAE", "sMAPE (%)", "R2"]
UQ_COLUMNS = ["PICP@90 (%)", "Interval Width (MW)"]
PRECISION = {
    "RMSE": 2,
    "MAE": 2,
    "sMAPE (%)": 2,
    "R2": 4,
    "PICP@90 (%)": 1,
    "Interval Width (MW)": 1,
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def _resolve_repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return value != value
    return False


def format_value(value: Any, column: str) -> str:
    if _is_missing(value):
        return "---"
    if isinstance(value, float):
        return f"{value:.{PRECISION[column]}f}"
    return str(value)


def load_conformal(uncertainty_dir: Path, target: str) -> dict[str, float | None]:
    """Return fallback GBM conformal metrics from <target>_conformal.json."""
    path = uncertainty_dir / f"{target}_conformal.json"
    data = load_json(path)
    meta = data.get("meta", {}) if isinstance(data.get("meta"), dict) else {}
    return {
        "picp": meta.get("global_coverage"),
        "width": meta.get("global_mean_width"),
    }


def load_model_uncertainty(
    *,
    target_data: dict[str, Any],
    model_key: str,
    uncertainty_dir: Path,
    target: str,
) -> dict[str, float | None]:
    model_payload = target_data.get(model_key, {})
    if isinstance(model_payload, dict):
        uncertainty = model_payload.get("uncertainty", {})
        if isinstance(uncertainty, dict) and uncertainty:
            return {
                "picp": uncertainty.get("picp_90", uncertainty.get("global_coverage")),
                "width": uncertainty.get("mean_interval_width", uncertainty.get("global_mean_width")),
            }
    if model_key == "gbm":
        return load_conformal(uncertainty_dir, target)
    return {"picp": None, "width": None}


def extract_rows(region: str, metrics_json: Path, uncertainty_dir: Path) -> list[dict[str, Any]]:
    metrics = load_json(metrics_json)
    targets_data = metrics.get("targets", {}) if isinstance(metrics.get("targets"), dict) else {}
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_data = targets_data.get(target, {}) if isinstance(targets_data.get(target), dict) else {}
        for model_key in MODEL_ORDER:
            metrics_payload = target_data.get(model_key)
            row: dict[str, Any] = {
                "Region": region,
                "Target": TARGET_LABELS.get(target, target),
                "Model": MODEL_LABELS.get(model_key, model_key),
                "RMSE": None,
                "MAE": None,
                "sMAPE (%)": None,
                "R2": None,
                "PICP@90 (%)": None,
                "Interval Width (MW)": None,
            }
            if isinstance(metrics_payload, dict):
                uncertainty = load_model_uncertainty(
                    target_data=target_data,
                    model_key=model_key,
                    uncertainty_dir=uncertainty_dir,
                    target=target,
                )
                smape_raw = metrics_payload.get("smape")
                picp_raw = uncertainty.get("picp")
                width_raw = uncertainty.get("width")
                row.update(
                    {
                        "RMSE": float(metrics_payload["rmse"]) if metrics_payload.get("rmse") is not None else None,
                        "MAE": float(metrics_payload["mae"]) if metrics_payload.get("mae") is not None else None,
                        "sMAPE (%)": float(smape_raw) * 100 if smape_raw is not None else None,
                        "R2": float(metrics_payload["r2"]) if metrics_payload.get("r2") is not None else None,
                        "PICP@90 (%)": float(picp_raw) * 100 if picp_raw is not None else None,
                        "Interval Width (MW)": float(width_raw) if width_raw is not None else None,
                    }
                )
            rows.append(row)
    return rows


def _format_rows_for_export(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    formatted: list[dict[str, str]] = []
    for row in rows:
        formatted.append(
            {
                "Region": str(row["Region"]),
                "Target": str(row["Target"]),
                "Model": str(row["Model"]),
                "RMSE": format_value(row["RMSE"], "RMSE"),
                "MAE": format_value(row["MAE"], "MAE"),
                "sMAPE (%)": format_value(row["sMAPE (%)"], "sMAPE (%)"),
                "R2": format_value(row["R2"], "R2"),
                "PICP@90 (%)": format_value(row["PICP@90 (%)"], "PICP@90 (%)"),
                "Interval Width (MW)": format_value(row["Interval Width (MW)"], "Interval Width (MW)"),
            }
        )
    return formatted


def _region_status(region: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_rows = len(TARGETS) * len(MODEL_ORDER)
    point_complete = 0
    gbm_uq_complete = 0
    missing_point_rows: list[str] = []
    for row in rows:
        point_ok = all(not _is_missing(row[column]) for column in POINT_COLUMNS)
        if point_ok:
            point_complete += 1
        else:
            missing_point_rows.append(f"{row['Target']}:{row['Model']}")
        if row["Model"] == "GBM":
            uq_ok = all(not _is_missing(row[column]) for column in UQ_COLUMNS)
            if uq_ok:
                gbm_uq_complete += 1
    return {
        "region": region,
        "expected_rows": expected_rows,
        "point_complete_rows": point_complete,
        "point_metrics_complete": point_complete == expected_rows,
        "gbm_uq_complete_rows": gbm_uq_complete,
        "gbm_uq_complete": gbm_uq_complete == len(TARGETS),
        "missing_point_rows": missing_point_rows,
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    export_rows = _format_rows_for_export(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(export_rows)
    print(f"  CSV   -> {path}")


def _write_table_header(lines: list[str], caption: str, label: str, column_spec: str) -> None:
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.95}")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{column_spec}}}")
    lines.append(r"\toprule")


def _write_table_footer(lines: list[str]) -> None:
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")


def write_latex(rows: list[dict[str, Any]], path: Path, region: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    _write_table_header(
        lines,
        (
            f"{region} forecast baseline comparison across the six promoted model families "
            "under identical splits, horizon, and lookback settings."
        ),
        f"tab:baseline_comparison_{region.lower()}",
        "llrrrrrr",
    )
    lines.append(r"Target & Model & RMSE & MAE & sMAPE (\%) & $R^2$ & P90 & Width \\")
    lines.append(r"\midrule")
    for target in TARGETS:
        target_label = TARGET_LABELS[target]
        target_rows = [row for row in rows if row["Target"] == target_label]
        for index, row in enumerate(target_rows):
            prefix = rf"\multirow{{{len(target_rows)}}}{{*}}{{{target_label}}}" if index == 0 else ""
            model = row["Model"]
            model_str = rf"\textbf{{{model}}}" if model == "GBM" else model
            lines.append(
                f"{prefix} & {model_str} & "
                f"{format_value(row['RMSE'], 'RMSE')} & "
                f"{format_value(row['MAE'], 'MAE')} & "
                f"{format_value(row['sMAPE (%)'], 'sMAPE (%)')} & "
                f"{format_value(row['R2'], 'R2')} & "
                f"{format_value(row['PICP@90 (%)'], 'PICP@90 (%)')} & "
                f"{format_value(row['Interval Width (MW)'], 'Interval Width (MW)')} \\\\"
            )
        if target != TARGETS[-1]:
            lines.append(r"\midrule")
    _write_table_footer(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  LaTeX -> {path}")


def write_latex_combined(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    _write_table_header(
        lines,
        (
            "Promoted six-model forecasting comparison across DE and canonical US under a shared "
            "split, horizon, lookback, and metric contract. P90 and width remain blank unless "
            "the pipeline produced model-specific uncertainty artifacts."
        ),
        "tab:baseline_comparison_all",
        "lllrrrrrr",
    )
    lines.append(r"Region & Target & Model & RMSE & MAE & sMAPE (\%) & $R^2$ & P90 & Width \\")
    lines.append(r"\midrule")
    for row in rows:
        model = row["Model"]
        model_str = rf"\textbf{{{model}}}" if model == "GBM" else model
        lines.append(
            f"{row['Region']} & {row['Target']} & {model_str} & "
            f"{format_value(row['RMSE'], 'RMSE')} & "
            f"{format_value(row['MAE'], 'MAE')} & "
            f"{format_value(row['sMAPE (%)'], 'sMAPE (%)')} & "
            f"{format_value(row['R2'], 'R2')} & "
            f"{format_value(row['PICP@90 (%)'], 'PICP@90 (%)')} & "
            f"{format_value(row['Interval Width (MW)'], 'Interval Width (MW)')} \\\\"
        )
    _write_table_footer(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  LaTeX -> {path}")


def _load_release_inputs(release_id: str) -> dict[str, dict[str, Path]]:
    dataset_map = {"DE": "de", "US": "us_miso"}
    resolved: dict[str, dict[str, Path]] = {}
    for region, dataset_lower in dataset_map.items():
        manifest_path = (
            REPO_ROOT
            / "artifacts"
            / "runs"
            / dataset_lower
            / release_id
            / "registry"
            / "run_manifest.json"
        )
        payload = load_json(manifest_path)
        if not payload:
            raise FileNotFoundError(f"Missing or invalid run manifest for {region}: {manifest_path}")
        if str(payload.get("release_id", "")) != release_id:
            raise ValueError(
                f"Run manifest {manifest_path} belongs to release_id={payload.get('release_id')!r}, "
                f"expected {release_id!r}"
            )
        if payload.get("accepted") is not True:
            raise ValueError(f"Run manifest {manifest_path} is not accepted.")
        artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), dict) else {}
        metrics_dir = _resolve_repo_path(artifacts.get("reports_dir"))
        uncertainty_dir = _resolve_repo_path(artifacts.get("uncertainty_dir"))
        if metrics_dir is None or uncertainty_dir is None:
            raise ValueError(f"Run manifest {manifest_path} is missing reports_dir or uncertainty_dir")
        resolved[region] = {
            "metrics_json": metrics_dir / "week2_metrics.json",
            "uncertainty_dir": uncertainty_dir,
            "manifest_path": manifest_path,
        }
    return resolved


def build_tables(
    *,
    out_dir: Path,
    release_id: str | None = None,
    de_metrics_json: Path,
    us_metrics_json: Path,
    de_uncertainty_dir: Path,
    us_uncertainty_dir: Path,
) -> dict[str, Any]:
    for metrics_path in (de_metrics_json, us_metrics_json):
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    region_rows = {
        "DE": extract_rows("DE", de_metrics_json, de_uncertainty_dir),
        "US": extract_rows("US", us_metrics_json, us_uncertainty_dir),
    }
    all_rows = region_rows["DE"] + region_rows["US"]

    write_csv(region_rows["DE"], out_dir / "baseline_comparison_de.csv")
    write_latex(region_rows["DE"], out_dir / "baseline_comparison_de.tex", "DE")
    write_csv(region_rows["US"], out_dir / "baseline_comparison_us.csv")
    write_latex(region_rows["US"], out_dir / "baseline_comparison_us.tex", "US")
    write_csv(all_rows, out_dir / "baseline_comparison_all.csv")
    write_latex_combined(all_rows, out_dir / "baseline_comparison_all.tex")

    status = {
        "release_id": release_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "regions": {
            "DE": _region_status("DE", region_rows["DE"]),
            "US": _region_status("US", region_rows["US"]),
        },
    }
    status["thesis_headline_point_metrics_complete"] = bool(
        status["regions"]["DE"]["point_metrics_complete"] and status["regions"]["US"]["point_metrics_complete"]
    )
    status["gbm_uq_complete"] = bool(
        status["regions"]["DE"]["gbm_uq_complete"] and status["regions"]["US"]["gbm_uq_complete"]
    )
    status_path = out_dir / "baseline_comparison_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"  JSON  -> {status_path}")
    return status


def build_release_baseline_comparison(*, release_id: str, out_dir: Path = OUT_DIR) -> dict[str, Any]:
    inputs = _load_release_inputs(release_id)
    return build_tables(
        out_dir=out_dir,
        release_id=release_id,
        de_metrics_json=inputs["DE"]["metrics_json"],
        us_metrics_json=inputs["US"]["metrics_json"],
        de_uncertainty_dir=inputs["DE"]["uncertainty_dir"],
        us_uncertainty_dir=inputs["US"]["uncertainty_dir"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build six-model baseline comparison tables.")
    parser.add_argument("--release-id", default=None, help="Read DE and US_MISO inputs from this accepted release family")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--de-dir", type=Path, default=None, help="Directory containing DE week2_metrics.json")
    parser.add_argument("--us-dir", type=Path, default=None, help="Directory containing US week2_metrics.json")
    parser.add_argument("--de-uncertainty-dir", type=Path, default=None, help="DE uncertainty artifact directory")
    parser.add_argument("--us-uncertainty-dir", type=Path, default=None, help="US uncertainty artifact directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.release_id:
        status = build_release_baseline_comparison(release_id=args.release_id, out_dir=out_dir)
    else:
        status = build_tables(
            out_dir=out_dir,
            de_metrics_json=(args.de_dir / "week2_metrics.json") if args.de_dir else REGION_DEFAULTS["DE"]["metrics_json"],
            us_metrics_json=(args.us_dir / "week2_metrics.json") if args.us_dir else REGION_DEFAULTS["US"]["metrics_json"],
            de_uncertainty_dir=args.de_uncertainty_dir or REGION_DEFAULTS["DE"]["uncertainty_dir"],
            us_uncertainty_dir=args.us_uncertainty_dir or REGION_DEFAULTS["US"]["uncertainty_dir"],
        )

    print(json.dumps(status, indent=2))
    if not status.get("thesis_headline_point_metrics_complete", False):
        print("WARNING: thesis headline rows are still incomplete.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
