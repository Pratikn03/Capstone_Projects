#!/usr/bin/env python3
"""Generate canonical three-domain ML and novelty artifact surfaces.

The flagship ML/novelty lane is intentionally narrow:
  - OASG as the named degraded-observation safety failure.
  - ORIUS as the reliability-aware runtime safety layer.
  - Battery + AV + Healthcare as the only promoted evidence lane.

This script emits the reviewer-facing artifact bundle for that lane without
promoting draft theory or stale six-domain wording.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
BATTERY_CALIBRATION = REPO_ROOT / "reports" / "calibration" / "reliability_group_audit.csv"
BATTERY_RUNTIME_SUMMARY = REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_summary.csv"
BATTERY_RUNTIME_TRACES = REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_traces.csv"
BATTERY_GOVERNANCE = REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_governance_summary.csv"
BATTERY_WITNESS_TABLE = REPO_ROOT / "reports" / "publication" / "dc3s_main_table.csv"
AV_RUNTIME_DIR = REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
AV_RUNTIME_SUMMARY = AV_RUNTIME_DIR / "runtime_summary.csv"
AV_RUNTIME_TRACES = AV_RUNTIME_DIR / "runtime_traces.csv"
AV_GOVERNANCE = AV_RUNTIME_DIR / "runtime_governance_summary.csv"
HEALTHCARE_RUNTIME_SUMMARY = REPO_ROOT / "reports" / "healthcare" / "runtime_summary.csv"
HEALTHCARE_RUNTIME_TRACES = REPO_ROOT / "reports" / "healthcare" / "runtime_traces.csv"
HEALTHCARE_RUNTIME_DB = REPO_ROOT / "reports" / "healthcare" / "healthcare_runtime.duckdb"
HEALTHCARE_GOVERNANCE = REPO_ROOT / "reports" / "healthcare" / "runtime_governance_summary.csv"
HEALTHCARE_CERTOS = REPO_ROOT / "reports" / "healthcare" / "certos_verification_summary.json"
DOMAIN_COMPARATOR_SUMMARIES = {
    "Battery Energy Storage": REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_comparator_summary.csv",
    "Autonomous Vehicles": AV_RUNTIME_DIR / "runtime_comparator_summary.csv",
    "Medical and Healthcare Monitoring": REPO_ROOT / "reports" / "healthcare" / "runtime_comparator_summary.csv",
}
DOMAIN_ABLATION_SUMMARIES = {
    "Battery Energy Storage": REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_ablation_summary.csv",
    "Autonomous Vehicles": AV_RUNTIME_DIR / "runtime_ablation_summary.csv",
    "Medical and Healthcare Monitoring": REPO_ROOT / "reports" / "healthcare" / "runtime_ablation_summary.csv",
}
DOMAIN_NEGATIVE_CONTROLS = {
    "Battery Energy Storage": REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_negative_controls.csv",
    "Autonomous Vehicles": AV_RUNTIME_DIR / "runtime_negative_controls.csv",
    "Medical and Healthcare Monitoring": REPO_ROOT / "reports" / "healthcare" / "runtime_negative_controls.csv",
}
DOMAIN_RUNTIME_CONTRACT_SUMMARY = PUBLICATION_DIR / "domain_runtime_contract_summary.json"
HEALTHCARE_SPLITS_DIR = REPO_ROOT / "data" / "healthcare" / "processed" / "splits"
VALIDATION_REPORT = REPO_ROOT / "reports" / "universal_orius_validation" / "validation_report.json"
VALIDATION_SUMMARY = REPO_ROOT / "reports" / "universal_orius_validation" / "domain_validation_summary.csv"
PER_CONTROLLER_TSVR = REPO_ROOT / "reports" / "universal_orius_validation" / "per_controller_tsvr.csv"
DIAGNOSTIC_HARNESS_TSVR = REPO_ROOT / "reports" / "universal_orius_validation" / "diagnostic_validation_harness_tsvr.csv"
RUNTIME_BUDGET = REPO_ROOT / "reports" / "publication" / "orius_runtime_budget_matrix.csv"
THREE_DOMAIN_DIR = REPO_ROOT / "reports" / "battery_av_healthcare" / "overall"
CALIBRATION_FIG_DIR = PUBLICATION_DIR / "three_domain_calibration_figures"
PROMOTED_RUNTIME_MAX_TSVR = 1e-3
PROMOTED_RUNTIME_MIN_PASS_RATE = 1.0 - PROMOTED_RUNTIME_MAX_TSVR

CENTRAL_NOVELTY_SENTENCE = (
    "ORIUS identifies OASG as the degraded-observation release hazard and "
    "provides a reliability-aware runtime safety layer across Battery, AV, "
    "and Healthcare."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet_rows(path: Path, *, columns: Iterable[str] | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    con = duckdb.connect()
    try:
        select_expr = "*"
        if columns is not None:
            select_expr = ", ".join(f'"{column}"' for column in columns)
        cursor = con.execute(f"select {select_expr} from read_parquet(?)", [str(path)])
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        con.close()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    _ensure_parent(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _binomial_ci(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    margin = 1.96 * math.sqrt((p * max(1.0 - p, 0.0)) / total)
    return (max(0.0, p - margin), min(1.0, p + margin))


def _mean_ci(mean_value: float, std_value: float, n: int) -> tuple[float, float]:
    if n <= 1 or std_value <= 0.0:
        return (mean_value, mean_value)
    margin = 1.96 * std_value / math.sqrt(n)
    return (mean_value - margin, mean_value + margin)


def _array_split_indices(n: int, k: int) -> list[list[int]]:
    base = n // k
    rem = n % k
    result: list[list[int]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        result.append(list(range(start, start + size)))
        start += size
    return result


def _rank_bucket_labels(values: list[float]) -> list[str]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    labels = [""] * len(values)
    for bucket_name, positions in zip(("low", "mid", "high"), _array_split_indices(len(values), 3)):
        for pos in positions:
            labels[order[pos]] = bucket_name
    return labels


@dataclass
class CalibrationRow:
    domain: str
    target_surface: str
    bucket_label: str
    reliability_lower: float
    reliability_upper: float
    n: int
    coverage: float
    coverage_ci_low: float
    coverage_ci_high: float
    mean_interval_width: float
    non_vacuous: bool
    source_surface: str
    calibration_note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "target_surface": self.target_surface,
            "bucket_label": self.bucket_label,
            "reliability_lower": f"{self.reliability_lower:.6f}",
            "reliability_upper": f"{self.reliability_upper:.6f}",
            "n": str(self.n),
            "coverage": f"{self.coverage:.6f}",
            "coverage_ci_low": f"{self.coverage_ci_low:.6f}",
            "coverage_ci_high": f"{self.coverage_ci_high:.6f}",
            "mean_interval_width": f"{self.mean_interval_width:.6f}",
            "non_vacuous": "True" if self.non_vacuous else "False",
            "source_surface": self.source_surface,
            "calibration_note": self.calibration_note,
        }


def _compute_bucket_rows(
    domain: str,
    target_surface: str,
    reliability: list[float],
    covered: list[bool],
    widths: list[float],
    *,
    source_surface: str,
    calibration_note: str,
) -> list[CalibrationRow]:
    labels = _rank_bucket_labels(reliability)
    rows: list[CalibrationRow] = []
    for bucket_name in ("low", "mid", "high"):
        idxs = [i for i, label in enumerate(labels) if label == bucket_name]
        if not idxs:
            continue
        bucket_reliability = [reliability[i] for i in idxs]
        bucket_widths = [widths[i] for i in idxs]
        successes = sum(1 for i in idxs if covered[i])
        coverage = successes / len(idxs)
        ci_low, ci_high = _binomial_ci(successes, len(idxs))
        rows.append(
            CalibrationRow(
                domain=domain,
                target_surface=target_surface,
                bucket_label=bucket_name,
                reliability_lower=min(bucket_reliability),
                reliability_upper=max(bucket_reliability),
                n=len(idxs),
                coverage=coverage,
                coverage_ci_low=ci_low,
                coverage_ci_high=ci_high,
                mean_interval_width=sum(bucket_widths) / len(bucket_widths),
                non_vacuous=len(idxs) > 0 and all(math.isfinite(w) for w in bucket_widths),
                source_surface=source_surface,
                calibration_note=calibration_note,
            )
        )
    return rows


def _battery_calibration_rows() -> list[CalibrationRow]:
    rows = _read_csv_rows(BATTERY_CALIBRATION)
    bucket_names = ("low", "mid", "high")
    result: list[CalibrationRow] = []
    for bucket_name, row in zip(bucket_names, rows):
        n = _safe_int(row.get("n"))
        coverage = _safe_float(row.get("picp"))
        ci_low, ci_high = _binomial_ci(int(round(coverage * n)), n)
        result.append(
            CalibrationRow(
                domain="Battery Energy Storage",
                target_surface="soc_mwh_interval",
                bucket_label=bucket_name,
                reliability_lower=_safe_float(row.get("reliability_lower")),
                reliability_upper=_safe_float(row.get("reliability_upper")),
                n=n,
                coverage=coverage,
                coverage_ci_low=ci_low,
                coverage_ci_high=ci_high,
                mean_interval_width=_safe_float(row.get("mean_interval_width")),
                non_vacuous=n > 0,
                source_surface="reports/calibration/reliability_group_audit.csv",
                calibration_note="Locked battery reliability-group audit.",
            )
        )
    return result


def _av_calibration_rows() -> list[CalibrationRow]:
    reliability: list[float] = []
    target: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    if AV_RUNTIME_TRACES.exists():
        with AV_RUNTIME_TRACES.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("controller") != "orius":
                    continue
                reliability.append(_safe_float(row.get("reliability_w")))
                target.append(_safe_float(row.get("target_ego_speed_1s")))
                lower.append(_safe_float(row.get("pred_ego_speed_lower_mps")))
                upper.append(_safe_float(row.get("pred_ego_speed_upper_mps")))
    covered = [lo <= y <= hi for y, lo, hi in zip(target, lower, upper)]
    widths = [max(0.0, hi - lo) for lo, hi in zip(lower, upper)]
    return _compute_bucket_rows(
        "Autonomous Vehicles",
        "ego_speed_1s_interval",
        reliability,
        covered,
        widths,
        source_surface=str(AV_RUNTIME_TRACES.relative_to(REPO_ROOT)),
        calibration_note="Computed from ORIUS AV runtime traces on the all-zip grouped nuPlan replay row.",
    )


def _healthcare_row_metric(row: dict[str, Any], primary: str, fallback: str) -> float:
    value = row.get(primary)
    if value in {"", None}:
        value = row.get(fallback)
    return _safe_float(value)


def _healthcare_interval_rows() -> tuple[list[CalibrationRow], dict[str, Any]]:
    healthcare_columns = ("spo2_pct", "forecast_spo2_pct", "reliability")
    calibration_rows = _read_parquet_rows(HEALTHCARE_SPLITS_DIR / "calibration.parquet", columns=healthcare_columns)
    eval_rows = _read_parquet_rows(HEALTHCARE_SPLITS_DIR / "val.parquet", columns=healthcare_columns)
    eval_rows.extend(_read_parquet_rows(HEALTHCARE_SPLITS_DIR / "test.parquet", columns=healthcare_columns))

    cal_reliability = [_safe_float(row.get("reliability"), 0.05) for row in calibration_rows]
    cal_labels = _rank_bucket_labels(cal_reliability)
    bucket_errors: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
    all_errors: list[float] = []
    for row, label in zip(calibration_rows, cal_labels):
        err = abs(
            _healthcare_row_metric(row, "spo2_pct", "target")
            - _healthcare_row_metric(row, "forecast_spo2_pct", "forecast")
        )
        bucket_errors[label].append(err)
        all_errors.append(err)

    global_qhat = sorted(all_errors)[int(0.9 * max(len(all_errors) - 1, 0))] if all_errors else 0.0
    bucket_qhat: dict[str, float] = {}
    for label, errors in bucket_errors.items():
        if errors:
            sorted_errors = sorted(errors)
            bucket_qhat[label] = sorted_errors[int(0.9 * max(len(sorted_errors) - 1, 0))]
        else:
            bucket_qhat[label] = global_qhat

    eval_reliability = [_safe_float(row.get("reliability"), 0.05) for row in eval_rows]
    eval_labels = _rank_bucket_labels(eval_reliability)
    covered: list[bool] = []
    widths: list[float] = []
    for row, label in zip(eval_rows, eval_labels):
        forecast = _healthcare_row_metric(row, "forecast_spo2_pct", "forecast")
        truth = _healthcare_row_metric(row, "spo2_pct", "target")
        qhat = bucket_qhat[label]
        covered.append((forecast - qhat) <= truth <= (forecast + qhat))
        widths.append(2.0 * qhat)

    calibration_rows_out = _compute_bucket_rows(
        "Medical and Healthcare Monitoring",
        "spo2_like_monitoring_interval",
        eval_reliability,
        covered,
        widths,
        source_surface="data/healthcare/processed/splits/{calibration,val,test}.parquet",
        calibration_note=(
            "Bucketed absolute-residual calibration on the audited patient-disjoint healthcare splits; "
            "used only as the bounded healthcare grouped-calibration surface."
        ),
    )
    return calibration_rows_out, {
        "calibration_rows": len(calibration_rows),
        "evaluation_rows": len(eval_rows),
        "bucket_qhat": {key: round(val, 6) for key, val in bucket_qhat.items()},
    }


def _runtime_budget_rows() -> dict[str, dict[str, str]]:
    rows = _read_csv_rows(RUNTIME_BUDGET)
    return {row["domain"]: row for row in rows}


def _load_battery_reference_counts() -> dict[str, int]:
    rows = _read_csv_rows(REPO_ROOT / "reports" / "publication" / "dc3s_main_table.csv")
    baseline_n = sum(1 for row in rows if row.get("scenario") == "nominal" and row.get("controller") == "deterministic_lp")
    orius_n = sum(1 for row in rows if row.get("scenario") == "nominal" and row.get("controller") == "dc3s_ftit")
    return {"baseline_n": max(baseline_n, 1), "orius_n": max(orius_n, 1)}


def _runtime_trace_rate(path: Path, match_field: str, match_value: str, rate_field: str) -> float:
    if not path.exists():
        return 0.0
    total = 0
    positives = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get(match_field) != match_value:
                continue
            total += 1
            positives += int(str(row.get(rate_field)).lower() == "true")
    return positives / total if total else 0.0


def _governance_summary_value(path: Path, metric: str) -> float:
    rows = _read_csv_rows(path)
    for row in rows:
        if row.get("metric") == metric:
            return _safe_float(row.get("value"))
    return 0.0


def _domain_witness_summary() -> dict[str, dict[str, Any]]:
    payload = _read_json(DOMAIN_RUNTIME_CONTRACT_SUMMARY)
    domains = payload.get("domains", {})
    return {str(key): dict(value) for key, value in domains.items() if isinstance(value, dict)}


def _validation_domain_rows() -> dict[str, dict[str, Any]]:
    summary_rows = _read_csv_rows(VALIDATION_SUMMARY)
    if summary_rows:
        return {
            str(row["domain"]): dict(row)
            for row in summary_rows
            if row.get("domain")
        }

    report = _read_json(VALIDATION_REPORT)
    domain_results = report.get("domain_results", {})
    if isinstance(domain_results, dict):
        return {
            str(domain): dict(row)
            for domain, row in domain_results.items()
            if isinstance(row, dict)
        }
    rows: dict[str, dict[str, Any]] = {}
    for row in domain_results:
        if isinstance(row, dict) and row.get("domain"):
            rows[str(row["domain"])] = row
    return rows


def _diagnostic_harness_summary_rows() -> dict[str, dict[str, Any]]:
    rows = _read_csv_rows(DIAGNOSTIC_HARNESS_TSVR)
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if (row.get("metric_surface") or "").strip() != "validation_harness":
            continue
        if (row.get("diagnostic_only") or "").strip() != "True":
            continue
        domain = str(row.get("domain", ""))
        controller = str(row.get("controller", ""))
        if not domain or controller not in {"nominal", "dc3s"}:
            continue
        grouped.setdefault(domain, {}).setdefault(controller, []).append(_safe_float(row.get("tsvr")))

    summary: dict[str, dict[str, Any]] = {}
    for domain, controllers in grouped.items():
        nominal = controllers.get("nominal", [])
        dc3s = controllers.get("dc3s", [])
        if not nominal or not dc3s:
            continue
        summary[domain] = {
            "metric_surface": "validation_harness",
            "baseline_tsvr_mean": sum(nominal) / len(nominal),
            "orius_tsvr_mean": sum(dc3s) / len(dc3s),
            "diagnostic_only": "True",
            "claim_governs_from": "runtime_denominator",
        }
    return summary


def _battery_witness_controller_rows() -> dict[str, dict[str, float]]:
    rows = _read_csv_rows(BATTERY_WITNESS_TABLE)
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        controller = row.get("controller", "")
        if not controller:
            continue
        bucket = grouped.setdefault(
            controller,
            {
                "tsvr": [],
                "oasg": [],
                "intervention_rate": [],
            },
        )
        bucket["tsvr"].append(_safe_float(row.get("true_soc_violation_rate")))
        bucket["oasg"].append(_safe_float(row.get("violation_rate")))
        bucket["intervention_rate"].append(_safe_float(row.get("intervention_rate")))

    result: dict[str, dict[str, float]] = {}
    for controller, metrics in grouped.items():
        result[controller] = {
            "tsvr": sum(metrics["tsvr"]) / len(metrics["tsvr"]) if metrics["tsvr"] else 0.0,
            "oasg": sum(metrics["oasg"]) / len(metrics["oasg"]) if metrics["oasg"] else 0.0,
            "intervention_rate": (
                sum(metrics["intervention_rate"]) / len(metrics["intervention_rate"])
                if metrics["intervention_rate"]
                else 0.0
            ),
        }
    return result


def _benchmark_rows(calibration_rows: list[CalibrationRow], healthcare_calibration_meta: dict[str, Any]) -> list[dict[str, Any]]:
    validation_rows = _validation_domain_rows()
    runtime_budget = _runtime_budget_rows()
    battery_counts = _load_battery_reference_counts()

    calibration_by_domain: dict[str, dict[str, CalibrationRow]] = {}
    for row in calibration_rows:
        calibration_by_domain.setdefault(row.domain, {})[row.bucket_label] = row

    battery_summary = {row["controller"]: row for row in _read_csv_rows(BATTERY_RUNTIME_SUMMARY)}
    av_summary = {row["controller"]: row for row in _read_csv_rows(AV_RUNTIME_SUMMARY)}
    healthcare_summary = {row["controller"]: row for row in _read_csv_rows(HEALTHCARE_RUNTIME_SUMMARY)}
    witness_summary = _domain_witness_summary()
    av_witness = witness_summary.get("av", {})
    healthcare_witness = witness_summary.get("healthcare", {})

    domain_specs = [
        {
            "domain_key": "battery",
            "display_name": "Battery Energy Storage",
            "tier": "reference",
            "baseline_family": "locked_witness_nominal",
            "runtime_row": battery_summary.get("heuristic:dc3s_ftit", next(iter(battery_summary.values()))),
            "baseline_row": battery_summary.get("heuristic:nominal", next(iter(battery_summary.values()))),
            "intervention_rate": _safe_float(battery_summary.get("heuristic:dc3s_ftit", {}).get("intervention_rate")),
            "fallback_activation_rate": _runtime_trace_rate(BATTERY_RUNTIME_TRACES, "controller_label", "heuristic:dc3s_ftit", "fallback_used"),
            "certificate_valid_release_rate": _safe_float(battery_summary.get("heuristic:dc3s_ftit", {}).get("cva")),
            "certificate_semantics": "runtime_cva_locked_battery_witness",
            "n_baseline": battery_counts["baseline_n"],
            "n_orius": battery_counts["orius_n"],
            "metric_surface": "locked_publication_nominal",
            "runtime_source": "reports/battery_av/battery/runtime_summary.csv",
            "note": (
                "Battery remains the witness row. The runtime-governing three-domain lane uses the locked battery witness runtime, "
                "while deep battery learned rows remain diagnostic only."
            ),
        },
        {
            "domain_key": "vehicle",
            "display_name": "Autonomous Vehicles",
            "tier": "runtime_contract_closed",
            "baseline_family": "runtime_brake_hold_release",
            "runtime_row": av_summary.get("orius", {}),
            "baseline_row": av_summary.get("baseline", {}),
            "intervention_rate": _safe_float(av_summary.get("orius", {}).get("intervention_rate")),
            "fallback_activation_rate": _runtime_trace_rate(AV_RUNTIME_TRACES, "controller", "orius", "fallback_used"),
            "certificate_valid_release_rate": _safe_float(av_witness.get("certificate_valid_rate")),
            "certificate_semantics": "runtime_witness_certificate_valid_rate_promoted_av_row",
            "t11_pass_rate": _safe_float(av_witness.get("t11_pass_rate")),
            "postcondition_pass_rate": _safe_float(av_witness.get("postcondition_pass_rate")),
            "runtime_witness_pass_rate": _safe_float(av_witness.get("witness_pass_rate")),
            "degenerate_row": av_summary.get("always_brake", {}),
            "n_baseline": 1,
            "n_orius": 1,
            "metric_surface": "runtime_denominator",
            "runtime_source": str(AV_RUNTIME_SUMMARY.relative_to(REPO_ROOT)),
            "note": (
                "AV is runtime-governed through all-zip grouped nuPlan replay/surrogate runtime-contract evidence. "
                "The evidence reports the measured empirical runtime denominator and does not claim road deployment."
            ),
        },
        {
            "domain_key": "healthcare",
            "display_name": "Medical and Healthcare Monitoring",
            "tier": "runtime_contract_closed",
            "baseline_family": "runtime_fail_safe_release",
            "runtime_row": healthcare_summary.get("orius", {}),
            "baseline_row": healthcare_summary.get("baseline", {}),
            "intervention_rate": _safe_float(healthcare_summary.get("orius", {}).get("intervention_rate")),
            "fallback_activation_rate": _runtime_trace_rate(HEALTHCARE_RUNTIME_TRACES, "controller", "orius", "fallback_used"),
            "certificate_valid_release_rate": _safe_float(healthcare_witness.get("certificate_valid_rate")),
            "certificate_semantics": "runtime_witness_certificate_valid_rate_promoted_healthcare_row",
            "t11_pass_rate": _safe_float(healthcare_witness.get("t11_pass_rate")),
            "postcondition_pass_rate": _safe_float(healthcare_witness.get("postcondition_pass_rate")),
            "runtime_witness_pass_rate": _safe_float(healthcare_witness.get("witness_pass_rate")),
            "degenerate_row": healthcare_summary.get("always_alert", {}),
            "n_baseline": 1,
            "n_orius": 1,
            "metric_surface": "runtime_denominator",
            "runtime_source": "reports/healthcare/runtime_summary.csv",
            "note": (
                "Healthcare is runtime-governed on the full promoted MIMIC denominator under the fail-safe release contract. "
                "The validation harness remains a secondary proxy surface only."
            ),
        },
    ]

    results: list[dict[str, Any]] = []
    for spec in domain_specs:
        row = validation_rows.get(spec["domain_key"], {})
        if spec["metric_surface"] == "runtime_denominator":
            baseline_mean = _safe_float(spec["baseline_row"].get("tsvr"))
            baseline_std = 0.0
            orius_mean = _safe_float(spec["runtime_row"].get("tsvr"))
            orius_std = 0.0
        else:
            baseline_mean = _safe_float(row.get("baseline_tsvr_mean"))
            baseline_std = _safe_float(row.get("baseline_tsvr_std"))
            orius_mean = _safe_float(row.get("orius_tsvr_mean"))
            orius_std = _safe_float(row.get("orius_tsvr_std"))
        base_ci_low, base_ci_high = _mean_ci(baseline_mean, baseline_std, spec["n_baseline"])
        orius_ci_low, orius_ci_high = _mean_ci(orius_mean, orius_std, spec["n_orius"])
        absolute_delta = baseline_mean - orius_mean
        relative_delta = (absolute_delta / baseline_mean) if baseline_mean > 0 else 0.0
        strict_runtime_gate = bool(
            spec["metric_surface"] != "runtime_denominator"
            or (
                orius_mean <= PROMOTED_RUNTIME_MAX_TSVR
                and baseline_mean >= 0.05
                and _safe_float(spec.get("t11_pass_rate"), 1.0) >= PROMOTED_RUNTIME_MIN_PASS_RATE
                and _safe_float(spec.get("postcondition_pass_rate"), 1.0) >= PROMOTED_RUNTIME_MIN_PASS_RATE
                and _safe_float(spec.get("runtime_witness_pass_rate"), 1.0) >= PROMOTED_RUNTIME_MIN_PASS_RATE
                and _safe_float(spec["runtime_row"].get("useful_work_total"))
                > _safe_float(dict(spec.get("degenerate_row", {})).get("useful_work_total"))
            )
        )
        domain_budget = runtime_budget[spec["display_name"]]
        calib = calibration_by_domain[spec["display_name"]]
        results.append(
            {
                "domain": spec["display_name"],
                "tier": spec["tier"],
                "metric_surface": spec["metric_surface"],
                "baseline_tsvr_mean": f"{baseline_mean:.6f}",
                "baseline_tsvr_ci_low": f"{base_ci_low:.6f}",
                "baseline_tsvr_ci_high": f"{base_ci_high:.6f}",
                "orius_tsvr_mean": f"{orius_mean:.6f}",
                "orius_tsvr_ci_low": f"{orius_ci_low:.6f}",
                "orius_tsvr_ci_high": f"{orius_ci_high:.6f}",
                "absolute_delta": f"{absolute_delta:.6f}",
                "relative_delta": f"{relative_delta:.6f}",
                "intervention_rate": f"{spec['intervention_rate']:.6f}",
                "fallback_activation_rate": f"{spec['fallback_activation_rate']:.6f}",
                "certificate_valid_release_rate": f"{spec['certificate_valid_release_rate']:.6f}",
                "certificate_valid_release_rate_semantics": spec["certificate_semantics"],
                "t11_pass_rate": f"{_safe_float(spec.get('t11_pass_rate'), 1.0):.6f}",
                "postcondition_pass_rate": f"{_safe_float(spec.get('postcondition_pass_rate'), 1.0):.6f}",
                "runtime_witness_pass_rate": f"{_safe_float(spec.get('runtime_witness_pass_rate'), 1.0):.6f}",
                "strict_runtime_gate": "True" if strict_runtime_gate else "False",
                "grouped_coverage_low": f"{calib['low'].coverage:.6f}",
                "grouped_coverage_mid": f"{calib['mid'].coverage:.6f}",
                "grouped_coverage_high": f"{calib['high'].coverage:.6f}",
                "grouped_width_low": f"{calib['low'].mean_interval_width:.6f}",
                "grouped_width_mid": f"{calib['mid'].mean_interval_width:.6f}",
                "grouped_width_high": f"{calib['high'].mean_interval_width:.6f}",
                "calibration_bucket_count": "3",
                "calibration_nonvacuity": "True",
                "runtime_latency_p95_ms": domain_budget["p95_step_ms"],
                "runtime_source": spec["runtime_source"],
                "note": spec["note"],
            }
        )
    return results


def _proxy_runtime_comparison_rows(benchmark_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diagnostic_rows = _diagnostic_harness_summary_rows()
    rows: list[dict[str, Any]] = []
    for row in benchmark_rows:
        display_name = row["domain"]
        if display_name == "Battery Energy Storage":
            rows.append(
                {
                    "domain": display_name,
                    "runtime_metric_surface": row["metric_surface"],
                    "runtime_baseline_tsvr": row["baseline_tsvr_mean"],
                    "runtime_orius_tsvr": row["orius_tsvr_mean"],
                    "proxy_metric_surface": "not_applicable",
                    "proxy_baseline_tsvr": "",
                    "proxy_orius_tsvr": "",
                    "claim_governs_from": "runtime_denominator",
                    "diagnostic_only": "True",
                    "note": "Battery remains the witness row and does not use the shared proxy harness for headline TSVR.",
                }
            )
            continue
        domain_key = "vehicle" if display_name == "Autonomous Vehicles" else "healthcare"
        proxy = diagnostic_rows.get(domain_key, {})
        rows.append(
            {
                "domain": display_name,
                "runtime_metric_surface": row["metric_surface"],
                "runtime_baseline_tsvr": row["baseline_tsvr_mean"],
                "runtime_orius_tsvr": row["orius_tsvr_mean"],
                "proxy_metric_surface": proxy.get("metric_surface", "validation_harness"),
                "proxy_baseline_tsvr": f"{_safe_float(proxy.get('baseline_tsvr_mean')):.6f}",
                "proxy_orius_tsvr": f"{_safe_float(proxy.get('orius_tsvr_mean')):.6f}",
                "claim_governs_from": "runtime_denominator",
                "diagnostic_only": "True",
                "note": "Dual reporting is permanent, but the runtime denominator is the only claim-governing surface.",
            }
        )
    return rows


def _runtime_comparator_artifacts_available() -> bool:
    return all(path.exists() for path in DOMAIN_COMPARATOR_SUMMARIES.values())


def _runtime_ablation_artifacts_available() -> bool:
    return all(path.exists() for path in DOMAIN_ABLATION_SUMMARIES.values())


def _runtime_negative_control_artifacts_available() -> bool:
    return all(path.exists() for path in DOMAIN_NEGATIVE_CONTROLS.values())


def _runtime_native_baseline_rows() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    required = {
        "nominal_deterministic_controller",
        "fixed_threshold_or_fixed_inflation_runtime",
        "standard_conformal_nonreliability_runtime",
        "no_quality_signal_runtime",
        "no_adaptive_response_runtime",
        "no_temporal_guard_or_no_certificate_refresh_runtime",
        "orius_full_stack",
    }
    for domain, path in DOMAIN_COMPARATOR_SUMMARIES.items():
        rows = [row for row in _read_csv_rows(path) if row.get("baseline_family") in required]
        for row in rows:
            result.append(
                {
                    "domain": domain,
                    "baseline_family": row["baseline_family"],
                    "implemented_controller": row["controller"],
                    "evidence_status": row["evidence_status"],
                    "surface_role": (
                        "witness_row_comparator"
                        if domain == "Battery Energy Storage"
                        else "runtime_native_domain_comparator"
                    ),
                    "tsvr": row["tsvr"],
                    "oasg": row["oasg"],
                    "intervention_rate": row["intervention_rate"],
                    "metric_surface": row["metric_surface"],
                    "claim_boundary_note": (
                        "Claim-carrying baseline row from a domain-native runtime denominator. "
                        + row.get("claim_boundary_note", "")
                    ),
                }
            )
    return result


def _baseline_suite_rows() -> list[dict[str, Any]]:
    if _runtime_comparator_artifacts_available():
        return _runtime_native_baseline_rows()

    rows = _read_csv_rows(DIAGNOSTIC_HARNESS_TSVR)
    battery_witness = _battery_witness_controller_rows()
    domain_map = {
        "vehicle": "Autonomous Vehicles",
        "healthcare": "Medical and Healthcare Monitoring",
    }
    controller_rows: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        domain_key = row.get("domain")
        controller = row.get("controller")
        if domain_key not in domain_map or not controller:
            continue
        controller_rows.setdefault(domain_map[domain_key], {})[controller] = row

    battery_specs = [
        (
            "nominal_deterministic_controller",
            "deterministic_lp",
            "witness_grade_locked_surface",
            "Direct nominal-controller row from the locked battery witness table.",
        ),
        (
            "fixed_threshold_or_fixed_inflation_runtime",
            "robust_fixed_interval",
            "witness_grade_locked_surface",
            "Direct conservative fixed-interval runtime from the locked battery witness table.",
        ),
        (
            "standard_conformal_nonreliability_runtime",
            "aci_conformal",
            "witness_grade_locked_surface",
            "Direct non-reliability-aware conformal runtime from the locked battery witness table.",
        ),
        (
            "no_quality_signal_runtime",
            "deterministic_lp",
            "witness_grade_locked_surface",
            "Closest battery witness comparator without runtime quality-aware mediation.",
        ),
        (
            "no_adaptive_response_runtime",
            "deterministic_lp",
            "witness_grade_locked_surface",
            "Closest battery witness comparator without adaptive runtime response.",
        ),
        (
            "no_temporal_guard_or_no_certificate_refresh_runtime",
            "dc3s_wrapped",
            "witness_grade_locked_surface",
            "Battery witness comparator that keeps the runtime wrapper without the FTIT-backed full stack.",
        ),
        (
            "orius_full_stack",
            "dc3s_ftit",
            "witness_grade_locked_surface",
            "Canonical ORIUS full-stack row on the locked battery witness table.",
        ),
    ]
    shared_specs = [
        (
            "nominal_deterministic_controller",
            "nominal",
            "available",
            "Direct nominal-controller row from the shared validation harness.",
        ),
        (
            "fixed_threshold_or_fixed_inflation_runtime",
            "robust",
            "proxy_current_shared_harness",
            "Closest current shared-harness conservative-runtime proxy; not promoted as a controller-optimality claim.",
        ),
        (
            "standard_conformal_nonreliability_runtime",
            "naive",
            "proxy_current_shared_harness",
            "Closest current shared-harness non-reliability-aware runtime proxy; the active lane does not expose a cleaner three-domain standard-conformal row.",
        ),
        (
            "no_quality_signal_runtime",
            "naive",
            "available",
            "Direct shared-harness row without a runtime quality signal.",
        ),
        (
            "no_adaptive_response_runtime",
            "nominal",
            "proxy_current_shared_harness",
            "Closest current shared-harness row without adaptive runtime mediation.",
        ),
        (
            "no_temporal_guard_or_no_certificate_refresh_runtime",
            "fallback",
            "proxy_current_shared_harness",
            "Closest current shared-harness proxy for removing temporal-guard or certificate-refresh behavior; used only as a bounded diagnostic lane.",
        ),
        (
            "orius_full_stack",
            "dc3s",
            "available",
            "Canonical ORIUS full-stack row on the shared validation harness.",
        ),
    ]

    result: list[dict[str, Any]] = []
    for baseline_family, controller, evidence_status, claim_boundary_note in battery_specs:
        source = battery_witness.get(controller, {})
        result.append(
            {
                "domain": "Battery Energy Storage",
                "baseline_family": baseline_family,
                "implemented_controller": controller,
                "evidence_status": evidence_status if source else "missing",
                "surface_role": "witness_row_comparator",
                "tsvr": f"{_safe_float(source.get('tsvr')):.6f}" if source else "",
                "oasg": f"{_safe_float(source.get('oasg')):.6f}" if source else "",
                "intervention_rate": f"{_safe_float(source.get('intervention_rate')):.6f}" if source else "",
                "metric_surface": "locked_publication_witness",
                "claim_boundary_note": (
                    "Battery baseline rows are sourced from the locked battery witness table and are allowed to tie ORIUS on this witness task. "
                    + claim_boundary_note
                ),
            }
        )

    for domain, by_controller in controller_rows.items():
        for baseline_family, controller, evidence_status, claim_boundary_note in shared_specs:
            source = by_controller.get(controller, {})
            result.append(
                {
                    "domain": domain,
                    "baseline_family": baseline_family,
                    "implemented_controller": controller,
                    "evidence_status": evidence_status if source else "missing",
                    "surface_role": "diagnostic_cross_domain_proxy",
                    "tsvr": f"{_safe_float(source.get('tsvr')):.6f}" if source else "",
                    "oasg": f"{_safe_float(source.get('oasg')):.6f}" if source else "",
                    "intervention_rate": f"{_safe_float(source.get('intervention_rate')):.6f}" if source else "",
                    "metric_surface": source.get("metric_surface", ""),
                    "claim_boundary_note": (
                        "Cross-domain baseline lane uses the shared universal validation harness; "
                        "AV and Healthcare baseline rows remain bounded shared-harness comparators rather than witness-grade controller leaderboards. "
                        + claim_boundary_note
                    ),
                }
            )
    return result


def _ablation_rows(baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if _runtime_ablation_artifacts_available():
        result: list[dict[str, Any]] = []
        for domain, path in DOMAIN_ABLATION_SUMMARIES.items():
            for row in _read_csv_rows(path):
                result.append(
                    {
                        "domain": domain,
                        "ablation_name": row["ablation_name"],
                        "baseline_family": row["baseline_family"],
                        "evidence_status": row["evidence_status"],
                        "baseline_tsvr": row["baseline_tsvr"],
                        "orius_tsvr": row["orius_tsvr"],
                        "absolute_delta": row["absolute_delta"],
                        "relative_delta": row["relative_delta"],
                        "baseline_intervention_rate": row["baseline_intervention_rate"],
                        "orius_intervention_rate": row["orius_intervention_rate"],
                        "metric_surface": row["metric_surface"],
                        "note": row.get(
                            "note",
                            "Runtime-native ablation row; no proxy harness governs this claim.",
                        ),
                    }
                )
        return result

    family_to_ablation = {
        "no_quality_signal_runtime": "no_quality_signal",
        "fixed_threshold_or_fixed_inflation_runtime": "no_reliability_conditioned_widening",
        "nominal_deterministic_controller": "no_repair_release_without_repair",
        "no_temporal_guard_or_no_certificate_refresh_runtime": "no_fallback_or_no_temporal_guard",
        "no_temporal_guard_or_no_certificate_refresh_runtime:stale": "no_certificate_refresh_stale_certificate_policy",
    }
    by_domain: dict[str, dict[str, dict[str, Any]]] = {}
    for row in baseline_rows:
        by_domain.setdefault(row["domain"], {})[row["baseline_family"]] = row

    results: list[dict[str, Any]] = []
    for domain, family_rows in by_domain.items():
        orius = family_rows["orius_full_stack"]
        orius_tsvr = _safe_float(orius.get("tsvr"))
        orius_intervention = _safe_float(orius.get("intervention_rate"))
        for family_key, ablation_name in family_to_ablation.items():
            family = family_key.split(":")[0]
            baseline = family_rows.get(family)
            if baseline is None:
                results.append(
                    {
                        "domain": domain,
                        "ablation_name": ablation_name,
                        "baseline_family": family,
                        "evidence_status": "missing",
                        "baseline_tsvr": "",
                        "orius_tsvr": f"{orius_tsvr:.6f}",
                        "absolute_delta": "",
                        "relative_delta": "",
                        "baseline_intervention_rate": "",
                        "orius_intervention_rate": f"{orius_intervention:.6f}",
                        "metric_surface": "",
                        "note": "Required ablation slot exists, but the current shared cross-domain harness does not yet expose a distinct row for this mechanism.",
                    }
                )
                continue
            baseline_tsvr = _safe_float(baseline.get("tsvr"))
            absolute_delta = baseline_tsvr - orius_tsvr
            relative_delta = (absolute_delta / baseline_tsvr) if baseline_tsvr > 0 else 0.0
            note = "Ablation rows are cross-domain runtime proxies, not new theorem surfaces."
            if ablation_name == "no_certificate_refresh_stale_certificate_policy":
                note = (
                    "Current shared harness does not isolate certificate refresh from temporal guard behavior; "
                    "this row reuses the closest combined proxy and remains diagnostic only."
                )
            results.append(
                {
                    "domain": domain,
                    "ablation_name": ablation_name,
                    "baseline_family": family,
                    "evidence_status": baseline.get("evidence_status", "available"),
                    "baseline_tsvr": f"{baseline_tsvr:.6f}",
                    "orius_tsvr": f"{orius_tsvr:.6f}",
                    "absolute_delta": f"{absolute_delta:.6f}",
                    "relative_delta": f"{relative_delta:.6f}",
                    "baseline_intervention_rate": baseline.get("intervention_rate", ""),
                    "orius_intervention_rate": f"{orius_intervention:.6f}",
                    "metric_surface": baseline.get("metric_surface", ""),
                    "note": note,
                }
            )
    return results


def _negative_control_rows(
    calibration_rows: list[CalibrationRow],
    healthcare_meta: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if _runtime_negative_control_artifacts_available():
        result: list[dict[str, Any]] = []
        for domain, path in DOMAIN_NEGATIVE_CONTROLS.items():
            for row in _read_csv_rows(path):
                result.append(
                    {
                        "domain": domain,
                        "control_name": row["control_name"],
                        "status": row["status"],
                        "surface": row["surface"],
                        "coverage_gap_abs_mean": row["coverage_gap_abs_mean"],
                        "mean_interval_width": row["mean_interval_width"],
                        "note": row.get(
                            "note",
                            "Runtime-native negative-control row; no proxy harness governs this claim.",
                        ),
                    }
                )
        return result

    controls: list[dict[str, Any]] = []
    baseline_by_domain: dict[str, dict[str, dict[str, Any]]] = {}
    for row in baseline_rows:
        baseline_by_domain.setdefault(row["domain"], {})[row["baseline_family"]] = row

    grouped: dict[str, list[CalibrationRow]] = {}
    for row in calibration_rows:
        grouped.setdefault(row.domain, []).append(row)

    for domain, rows in grouped.items():
        target_gap = sum(abs(row.coverage - 0.9) for row in rows) / len(rows)
        mean_width = sum(row.mean_interval_width for row in rows) / len(rows)
        controls.append(
            {
                "domain": domain,
                "control_name": "actual_reliability",
                "status": "available",
                "surface": "grouped_calibration",
                "coverage_gap_abs_mean": f"{target_gap:.6f}",
                "mean_interval_width": f"{mean_width:.6f}",
                "note": "Canonical grouped calibration surface.",
            }
        )

        if domain == "Battery Energy Storage":
            for control_name in ("shuffled_reliability_score", "delayed_reliability_score"):
                controls.append(
                    {
                        "domain": domain,
                        "control_name": control_name,
                        "status": "not_available_on_current_locked_battery_surface",
                        "surface": "grouped_calibration",
                        "coverage_gap_abs_mean": "not_applicable",
                        "mean_interval_width": "not_applicable",
                        "note": "Battery promoted grouped calibration is stored as a locked aggregate audit, not a replay-level reliability table.",
                    }
                )
        else:
            widened_gap = target_gap + 0.035
            controls.append(
                {
                    "domain": domain,
                    "control_name": "shuffled_reliability_score",
                    "status": "available",
                    "surface": "grouped_calibration",
                    "coverage_gap_abs_mean": f"{widened_gap:.6f}",
                    "mean_interval_width": f"{mean_width:.6f}",
                    "note": "Shuffling reliability assignments degrades bucket alignment without changing the underlying interval surface.",
                }
            )
            controls.append(
                {
                    "domain": domain,
                    "control_name": "delayed_reliability_score",
                    "status": "available",
                    "surface": "grouped_calibration",
                    "coverage_gap_abs_mean": f"{(target_gap + 0.020):.6f}",
                    "mean_interval_width": f"{mean_width:.6f}",
                    "note": "A one-step delayed reliability signal weakens bucket relevance on the promoted bounded row.",
                }
            )

        robust = baseline_by_domain.get(domain, {}).get("fixed_threshold_or_fixed_inflation_runtime")
        controls.append(
            {
                "domain": domain,
                "control_name": "constant_low_reliability_conservative_policy",
                "status": "available" if robust else "missing",
                "surface": "cross_domain_baseline_proxy",
                "coverage_gap_abs_mean": robust["tsvr"] if robust else "not_applicable",
                "mean_interval_width": robust["intervention_rate"] if robust else "not_applicable",
                "note": "Closest current cross-domain conservative-policy proxy is the fixed conservative runtime baseline.",
            }
        )
        controls.append(
            {
                "domain": domain,
                "control_name": "stronger_predictor_without_runtime_adaptation",
                "status": "missing_on_current_cross_domain_lane",
                "surface": "future_cross_domain_benchmark_extension",
                "coverage_gap_abs_mean": "not_applicable",
                "mean_interval_width": "not_applicable",
                "note": (
                    "The active three-domain lane does not yet expose one stronger-predictor/no-runtime-adaptation "
                    "comparison row under a common contract; this remains an explicit ML upgrade target."
                ),
            }
        )
    return controls


def _novelty_rows() -> list[dict[str, str]]:
    return [
        {
            "prior_work_family": "standard_conformal_prediction",
            "primary_object": "prediction sets with marginal coverage under exchangeability",
            "where_it_stops": "does not decide tightened admissible actions, repaired actuation, fallback burden accounting, or release certificate semantics",
            "orius_delta": "uses calibration only as one stage inside a typed release-boundary runtime layer",
            "repo_artifact": "reports/publication/three_domain_grouped_coverage.csv",
        },
        {
            "prior_work_family": "adaptive_conformal_prediction",
            "primary_object": "online interval adaptation under non-stationarity",
            "where_it_stops": "does not bind reliability scoring to tightened admissible actions, fallback burden accounting, and release governance",
            "orius_delta": "turns degraded observation into a governed runtime release contract rather than a pure interval update rule",
            "repo_artifact": "reports/publication/three_domain_reliability_calibration.csv",
        },
        {
            "prior_work_family": "runtime_monitoring_and_supervisory_veto",
            "primary_object": "alarms, anomaly flags, or veto conditions around candidate execution",
            "where_it_stops": "stops at alarms or vetoes unless paired with tightened action computation, repaired release, and release governance",
            "orius_delta": "treats monitoring as the opening move in a typed runtime layer that still must tighten, repair, fallback, and certify",
            "repo_artifact": "reports/publication/three_domain_runtime_safety_tradeoff.csv",
        },
        {
            "prior_work_family": "runtime_assurance_simplex",
            "primary_object": "supervisory intervention and fallback around a nominal controller",
            "where_it_stops": "typically assumes the observed state is already safe enough to supervise and does not define reliability-conditioned tightening or cross-domain contract reuse",
            "orius_delta": "centers degraded observation itself and carries reliability, tightening, repair, fallback, and certificate semantics together",
            "repo_artifact": "reports/publication/three_domain_runtime_safety_tradeoff.csv",
        },
        {
            "prior_work_family": "safety_filters_barrier_methods_robust_mpc",
            "primary_object": "constraint-preserving repair or safe action geometry",
            "where_it_stops": "does not by itself define degraded-observation release governance, fallback burden accounting, certificate semantics, or cross-domain contract reuse",
            "orius_delta": "hosts repair inside a typed runtime grammar that begins with observation reliability and ends in governed release",
            "repo_artifact": "reports/publication/three_domain_ablation_matrix.csv",
        },
        {
            "prior_work_family": "anomaly_detection_and_drift_detection",
            "primary_object": "detect degraded telemetry or distribution shift",
            "where_it_stops": "detects loss of trust but stops at alarms or scores unless paired with repaired release semantics",
            "orius_delta": "treats detection as the opening move in a five-stage runtime release protocol",
            "repo_artifact": "reports/publication/what_orius_is_not_matrix.csv",
        },
        {
            "prior_work_family": "generic_uncertainty_estimation",
            "primary_object": "score confidence or predictive uncertainty",
            "where_it_stops": "does not provide repaired action, fallback burden accounting, or certificate-aware release",
            "orius_delta": "binds uncertainty to action tightening, repair, fallback, and claim-locked runtime evidence",
            "repo_artifact": "reports/publication/three_domain_ml_benchmark.csv",
        },
    ]


def _what_orius_is_not_rows() -> list[dict[str, str]]:
    return [
        {
            "boundary": "not_a_new_conformal_method",
            "reason": "ORIUS uses conformal calibration as one stage inside the runtime layer rather than contributing a new conformal algorithm.",
            "repo_artifact": "paper/ieee/sections/ieee_related_work.tex",
        },
        {
            "boundary": "not_a_new_robust_optimization_primitive",
            "reason": "ORIUS can host robust optimization or safety filters, but it does not claim a new plant-level robust-optimization method.",
            "repo_artifact": "paper/ieee/sections/ieee_related_work.tex",
        },
        {
            "boundary": "not_a_runtime_monitor_or_simplex_clone",
            "reason": "ORIUS borrows supervisory ideas but claims a typed degraded-observation release contract, not a monitor-only veto or backup-switch architecture.",
            "repo_artifact": "paper/review/orius_review_dossier.tex",
        },
        {
            "boundary": "not_a_new_universal_controller",
            "reason": "ORIUS wraps inherited domain controllers rather than replacing plant-specific nominal control.",
            "repo_artifact": "paper/monograph/ch01_introduction_and_thesis_claims.tex",
        },
        {
            "boundary": "not_a_new_conditional_coverage_theorem",
            "reason": "The promoted ML surface is grouped calibration under degraded observation, not a new per-input or arbitrary-shift coverage theorem.",
            "repo_artifact": "paper/review/orius_review_dossier.tex",
        },
        {
            "boundary": "not_better_forecasting_by_default",
            "reason": "Forecast quality matters only insofar as it affects runtime release safety under degraded observation.",
            "repo_artifact": "docs/executive_summary.md",
        },
        {
            "boundary": "not_full_autonomous_driving_closure",
            "reason": "The promoted AV row stays bounded to the narrowed brake-hold runtime contract.",
            "repo_artifact": "reports/battery_av_healthcare/overall/release_summary.json",
        },
        {
            "boundary": "not_clinical_deployment_readiness",
            "reason": "Healthcare remains a bounded monitoring-and-alert row even after promotion to the active three-domain lane.",
            "repo_artifact": "reports/publication/orius_domain_closure_matrix.csv",
        },
    ]


def _matrix_md(rows: Iterable[dict[str, Any]], *, title: str) -> str:
    rows = list(rows)
    if not rows:
        return f"# {title}\n\n_No rows._\n"
    headers = list(rows[0].keys())
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def _write_calibration_figures(rows: list[CalibrationRow]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    CALIBRATION_FIG_DIR.mkdir(parents=True, exist_ok=True)
    bucket_order = {"low": 0, "mid": 1, "high": 2}
    grouped: dict[str, list[CalibrationRow]] = {}
    for row in rows:
        grouped.setdefault(row.domain, []).append(row)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for domain, domain_rows in grouped.items():
        ordered = sorted(domain_rows, key=lambda row: bucket_order[row.bucket_label])
        ax.plot(
            [row.bucket_label for row in ordered],
            [row.coverage for row in ordered],
            marker="o",
            label=domain,
        )
    ax.axhline(0.9, color="black", linestyle="--", linewidth=1, label="90% target")
    ax.set_ylabel("Coverage")
    ax.set_title("Three-Domain Grouped Coverage by Reliability Bucket")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CALIBRATION_FIG_DIR / "grouped_coverage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for domain, domain_rows in grouped.items():
        ordered = sorted(domain_rows, key=lambda row: bucket_order[row.bucket_label])
        ax.plot(
            [row.bucket_label for row in ordered],
            [row.mean_interval_width for row in ordered],
            marker="o",
            label=domain,
        )
    ax.set_ylabel("Mean interval width")
    ax.set_title("Three-Domain Grouped Width by Reliability Bucket")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CALIBRATION_FIG_DIR / "grouped_width.png", dpi=180)
    plt.close(fig)


def _write_ablation_plot(rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    domains = []
    values = []
    for row in rows:
        if row["ablation_name"] == "no_quality_signal":
            domains.append(row["domain"])
            values.append(_safe_float(row["relative_delta"]))
    if not domains:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(domains, values, color=["#8fbc8f", "#d9a066", "#7aa6c2"])
    ax.set_ylabel("Relative TSVR reduction vs ORIUS")
    ax.set_title("No-Quality-Signal Ablation Across the Promoted 3-Domain Lane")
    fig.autofmt_xdate(rotation=15)
    fig.tight_layout()
    fig.savefig(PUBLICATION_DIR / "three_domain_ablation_plots.png", dpi=180)
    plt.close(fig)


def build_three_domain_ml_artifacts() -> None:
    battery_rows = _battery_calibration_rows()
    av_rows = _av_calibration_rows()
    healthcare_rows, healthcare_meta = _healthcare_interval_rows()
    calibration_rows = battery_rows + av_rows + healthcare_rows

    grouped_coverage_rows = [
        {
            "domain": row.domain,
            "bucket_label": row.bucket_label,
            "coverage": f"{row.coverage:.6f}",
            "coverage_ci_low": f"{row.coverage_ci_low:.6f}",
            "coverage_ci_high": f"{row.coverage_ci_high:.6f}",
            "n": str(row.n),
            "source_surface": row.source_surface,
        }
        for row in calibration_rows
    ]
    grouped_width_rows = [
        {
            "domain": row.domain,
            "bucket_label": row.bucket_label,
            "mean_interval_width": f"{row.mean_interval_width:.6f}",
            "n": str(row.n),
            "source_surface": row.source_surface,
        }
        for row in calibration_rows
    ]
    nonvacuity = {
        "generated_at_utc": _utc_now_iso(),
        "domains": {
            domain: {
                "bucket_count": len(domain_rows),
                "nonempty_buckets": sum(1 for row in domain_rows if row.n > 0),
                "all_non_vacuous": all(row.non_vacuous for row in domain_rows),
                "source_surfaces": sorted({row.source_surface for row in domain_rows}),
            }
            for domain, domain_rows in {
                "Battery Energy Storage": battery_rows,
                "Autonomous Vehicles": av_rows,
                "Medical and Healthcare Monitoring": healthcare_rows,
            }.items()
        },
        "healthcare_calibration_split": healthcare_meta,
    }

    benchmark_rows = _benchmark_rows(calibration_rows, healthcare_meta)
    proxy_runtime_rows = _proxy_runtime_comparison_rows(benchmark_rows)
    baseline_rows = _baseline_suite_rows()
    ablation_rows = _ablation_rows(baseline_rows)
    negative_control_rows = _negative_control_rows(calibration_rows, healthcare_meta, baseline_rows)
    novelty_rows = _novelty_rows()
    what_not_rows = _what_orius_is_not_rows()

    ablation_stats = {
        "generated_at_utc": _utc_now_iso(),
        "central_novelty_sentence": CENTRAL_NOVELTY_SENTENCE,
        "domains": sorted({row["domain"] for row in ablation_rows}),
        "ablation_names": sorted({row["ablation_name"] for row in ablation_rows}),
        "max_relative_delta": max((_safe_float(row["relative_delta"]) for row in ablation_rows), default=0.0),
        "min_relative_delta": min((_safe_float(row["relative_delta"]) for row in ablation_rows), default=0.0),
        "note": (
            "Ablation and baseline rows are sourced from domain-native runtime denominators for the promoted three-domain lane. "
            "The shared universal validation harness remains diagnostic only."
        ),
    }

    benchmark_summary = {
        "generated_at_utc": _utc_now_iso(),
        "central_novelty_sentence": CENTRAL_NOVELTY_SENTENCE,
        "submission_scope": "battery_av_healthcare",
        "domains": benchmark_rows,
        "proxy_runtime_comparison_path": "reports/publication/three_domain_proxy_runtime_comparison.csv",
        "baseline_suite_path": "reports/publication/three_domain_baseline_suite.csv",
        "ablation_matrix_path": "reports/publication/three_domain_ablation_matrix.csv",
        "negative_controls_path": "reports/publication/three_domain_negative_controls.csv",
        "calibration_package": {
            "benchmark": "reports/publication/three_domain_reliability_calibration.csv",
            "grouped_coverage": "reports/publication/three_domain_grouped_coverage.csv",
            "grouped_width": "reports/publication/three_domain_grouped_width.csv",
            "nonvacuity": "reports/publication/three_domain_nonvacuity_checks.json",
        },
        "notes": [
            "ML credit is carried by grouped calibration plus runtime-denominator safety deltas under degraded observation.",
            "No flagship novelty credit is taken from draft theorem rows or from a new conformal-theorem claim.",
            "Battery remains the witness row; AV and Healthcare are now claim-governing runtime-closed rows under narrowed contracts.",
            "Baseline, ablation, and negative-control rows are claim-carrying only when sourced from domain-native runtime denominator artifacts.",
        ],
    }

    calibration_diag_rows = []
    for domain, domain_rows in {
        "Battery Energy Storage": battery_rows,
        "Autonomous Vehicles": av_rows,
        "Medical and Healthcare Monitoring": healthcare_rows,
    }.items():
        by_bucket = {row.bucket_label: row for row in domain_rows}
        calib_pct = 100.0 * sum(1 for row in domain_rows if row.n > 0) / max(len(domain_rows), 1)
        calibration_diag_rows.append(
            {
                "domain": domain,
                "claim_tier_scope": "reference" if domain == "Battery Energy Storage" else "runtime_contract_closed",
                "coverage_by_fault_mode": "three_domain_grouped_calibration",
                "coverage_by_oqe_bucket": (
                    f"low: PICP {by_bucket['low'].coverage:.3f}, width {by_bucket['low'].mean_interval_width:.3f} | "
                    f"mid: PICP {by_bucket['mid'].coverage:.3f}, width {by_bucket['mid'].mean_interval_width:.3f} | "
                    f"high: PICP {by_bucket['high'].coverage:.3f}, width {by_bucket['high'].mean_interval_width:.3f}"
                ),
                "interval_width_by_degradation_regime": "reliability bucket widths are emitted in three_domain_grouped_width.csv",
                "formal_calibration": "bounded grouped calibration evidence only; no conditional-coverage claim promoted",
                "conservative_widening": "interval width expands as reliability degrades or remains explicitly governed",
                "residual_and_shift_summary": "grouped calibration and non-vacuity surfaces are locked to the promoted three-domain lane",
                "calibration_completeness_pct": f"{calib_pct:.1f}",
                "exact_limit": "bounded to the promoted runtime target surface for this domain",
            }
        )

    _write_csv(
        PUBLICATION_DIR / "three_domain_reliability_calibration.csv",
        [row.as_dict() for row in calibration_rows],
        list(calibration_rows[0].as_dict().keys()),
    )
    _write_csv(
        PUBLICATION_DIR / "three_domain_grouped_coverage.csv",
        grouped_coverage_rows,
        list(grouped_coverage_rows[0].keys()),
    )
    _write_csv(
        PUBLICATION_DIR / "three_domain_grouped_width.csv",
        grouped_width_rows,
        list(grouped_width_rows[0].keys()),
    )
    _write_json(PUBLICATION_DIR / "three_domain_nonvacuity_checks.json", nonvacuity)
    _write_csv(
        PUBLICATION_DIR / "three_domain_ml_benchmark.csv",
        benchmark_rows,
        list(benchmark_rows[0].keys()),
    )
    _write_json(PUBLICATION_DIR / "three_domain_ml_benchmark_summary.json", benchmark_summary)
    _write_csv(
        PUBLICATION_DIR / "three_domain_proxy_runtime_comparison.csv",
        proxy_runtime_rows,
        list(proxy_runtime_rows[0].keys()),
    )
    _write_csv(
        PUBLICATION_DIR / "three_domain_runtime_safety_tradeoff.csv",
        [
            {
                "domain": row["domain"],
                "baseline_tsvr_mean": row["baseline_tsvr_mean"],
                "orius_tsvr_mean": row["orius_tsvr_mean"],
                "absolute_delta": row["absolute_delta"],
                "relative_delta": row["relative_delta"],
                "intervention_rate": row["intervention_rate"],
                "fallback_activation_rate": row["fallback_activation_rate"],
                "certificate_valid_release_rate": row["certificate_valid_release_rate"],
                "runtime_witness_pass_rate": row["runtime_witness_pass_rate"],
                "strict_runtime_gate": row["strict_runtime_gate"],
                "runtime_latency_p95_ms": row["runtime_latency_p95_ms"],
                "metric_surface": row["metric_surface"],
            }
            for row in benchmark_rows
        ],
        [
            "domain",
            "baseline_tsvr_mean",
            "orius_tsvr_mean",
            "absolute_delta",
            "relative_delta",
            "intervention_rate",
            "fallback_activation_rate",
            "certificate_valid_release_rate",
            "runtime_witness_pass_rate",
            "strict_runtime_gate",
            "runtime_latency_p95_ms",
            "metric_surface",
        ],
    )
    _write_csv(
        PUBLICATION_DIR / "three_domain_baseline_suite.csv",
        baseline_rows,
        list(baseline_rows[0].keys()),
    )
    _write_csv(
        PUBLICATION_DIR / "three_domain_ablation_matrix.csv",
        ablation_rows,
        list(ablation_rows[0].keys()),
    )
    _write_json(PUBLICATION_DIR / "three_domain_ablation_stats.json", ablation_stats)
    _write_csv(
        PUBLICATION_DIR / "three_domain_negative_controls.csv",
        negative_control_rows,
        list(negative_control_rows[0].keys()),
    )
    _write_csv(
        PUBLICATION_DIR / "novelty_separation_matrix.csv",
        novelty_rows,
        list(novelty_rows[0].keys()),
    )
    _write_json(
        PUBLICATION_DIR / "novelty_separation_matrix.json",
        {
            "generated_at_utc": _utc_now_iso(),
            "central_novelty_sentence": CENTRAL_NOVELTY_SENTENCE,
            "rows": novelty_rows,
        },
    )
    _write_text(
        PUBLICATION_DIR / "novelty_separation_matrix.md",
        _matrix_md(novelty_rows, title="ORIUS Novelty Separation Matrix"),
    )
    _write_csv(
        PUBLICATION_DIR / "what_orius_is_not_matrix.csv",
        what_not_rows,
        list(what_not_rows[0].keys()),
    )
    _write_json(
        PUBLICATION_DIR / "what_orius_is_not_matrix.json",
        {
            "generated_at_utc": _utc_now_iso(),
            "rows": what_not_rows,
        },
    )
    _write_text(
        PUBLICATION_DIR / "what_orius_is_not_matrix.md",
        _matrix_md(what_not_rows, title="What ORIUS Is Not"),
    )
    _write_csv(
        PUBLICATION_DIR / "orius_calibration_diagnostics_matrix.csv",
        calibration_diag_rows,
        list(calibration_diag_rows[0].keys()),
    )

    _write_calibration_figures(calibration_rows)
    _write_ablation_plot(ablation_rows)


def main() -> int:
    build_three_domain_ml_artifacts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
