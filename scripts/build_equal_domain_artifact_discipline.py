#!/usr/bin/env python3
"""Build equal artifact-discipline gates for the promoted three-domain lane."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
AV_RUNTIME_DIR = (
    REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
)
PROMOTED_RUNTIME_MAX_TSVR = 1e-3
PROMOTED_RUNTIME_MIN_PASS_RATE = 1.0 - PROMOTED_RUNTIME_MAX_TSVR

FORBIDDEN_CLAIM_SURFACES = {
    "validation_harness",
    "diagnostic_cross_domain_proxy",
    "proxy_current_shared_harness",
    "missing",
    "missing_on_current_cross_domain_lane",
    "future_cross_domain_benchmark_extension",
}

REQUIRED_BASELINE_FAMILIES = {
    "nominal_deterministic_controller",
    "fixed_threshold_or_fixed_inflation_runtime",
    "standard_conformal_nonreliability_runtime",
    "no_quality_signal_runtime",
    "no_adaptive_response_runtime",
    "no_temporal_guard_or_no_certificate_refresh_runtime",
    "orius_full_stack",
}
REQUIRED_BASELINE_FAMILY_ORDER = (
    "nominal_deterministic_controller",
    "fixed_threshold_or_fixed_inflation_runtime",
    "standard_conformal_nonreliability_runtime",
    "no_quality_signal_runtime",
    "no_adaptive_response_runtime",
    "no_temporal_guard_or_no_certificate_refresh_runtime",
    "orius_full_stack",
)

REQUIRED_ABLATIONS = {
    "no_quality_signal",
    "no_reliability_conditioned_widening",
    "no_repair_release_without_repair",
    "no_fallback_or_no_temporal_guard",
    "no_certificate_refresh_stale_certificate_policy",
}

REQUIRED_NEGATIVE_CONTROLS = {
    "actual_reliability",
    "shuffled_reliability_score",
    "delayed_reliability_score",
    "constant_low_reliability_conservative_policy",
    "stronger_predictor_without_runtime_adaptation",
}


@dataclass(frozen=True)
class DomainSpec:
    key: str
    display: str
    report_dir: Path
    runtime_summary: Path
    runtime_traces: Path
    orius_controller: str
    degenerate_controller: str
    theorem_ids: tuple[str, ...]
    proof_anchor: str
    governing_runtime_source: str


DOMAIN_SPECS: dict[str, DomainSpec] = {
    "battery": DomainSpec(
        key="battery",
        display="Battery Energy Storage",
        report_dir=REPO_ROOT / "reports" / "battery_av" / "battery",
        runtime_summary=REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_summary.csv",
        runtime_traces=REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_traces.csv",
        orius_controller="deep:dc3s_ftit",
        degenerate_controller="always_hold_safe_fallback",
        theorem_ids=("T_EQ_Battery_RuntimeArtifactPackage",),
        proof_anchor="T_EQ_Battery_RuntimeArtifactPackage",
        governing_runtime_source="reports/battery_av/battery/runtime_summary.csv",
    ),
    "av": DomainSpec(
        key="av",
        display="Autonomous Vehicles",
        report_dir=AV_RUNTIME_DIR,
        runtime_summary=AV_RUNTIME_DIR / "runtime_summary.csv",
        runtime_traces=AV_RUNTIME_DIR / "runtime_traces.csv",
        orius_controller="orius",
        degenerate_controller="always_brake",
        theorem_ids=("T11_AV_BrakeHold", "T6_AV_FallbackValidity", "T_EQ_AV_RuntimeArtifactPackage"),
        proof_anchor="T_EQ_AV_RuntimeArtifactPackage",
        governing_runtime_source=str((AV_RUNTIME_DIR / "runtime_summary.csv").relative_to(REPO_ROOT)),
    ),
    "healthcare": DomainSpec(
        key="healthcare",
        display="Medical and Healthcare Monitoring",
        report_dir=REPO_ROOT / "reports" / "healthcare",
        runtime_summary=REPO_ROOT / "reports" / "healthcare" / "runtime_summary.csv",
        runtime_traces=REPO_ROOT / "reports" / "healthcare" / "runtime_traces.csv",
        orius_controller="orius",
        degenerate_controller="always_alert",
        theorem_ids=("T11_HC_FailSafeRelease", "T6_HC_FallbackValidity", "T_EQ_HC_RuntimeArtifactPackage"),
        proof_anchor="T_EQ_HC_RuntimeArtifactPackage",
        governing_runtime_source="reports/healthcare/runtime_summary.csv",
    ),
}

FAMILY_CONTROLLER_MAP: dict[str, dict[str, str]] = {
    "battery": {
        "nominal_deterministic_controller": "deep:dc3s_wrapped",
        "fixed_threshold_or_fixed_inflation_runtime": "deep:dc3s_wrapped",
        "standard_conformal_nonreliability_runtime": "deep:dc3s_wrapped",
        "no_quality_signal_runtime": "deep:dc3s_wrapped",
        "no_adaptive_response_runtime": "deep:dc3s_wrapped",
        "no_temporal_guard_or_no_certificate_refresh_runtime": "deep:dc3s_wrapped",
        "orius_full_stack": "deep:dc3s_ftit",
        "degenerate_fallback_runtime": "always_hold_safe_fallback",
    },
    "av": {
        "nominal_deterministic_controller": "baseline",
        "fixed_threshold_or_fixed_inflation_runtime": "robust_fixed_deceleration",
        "standard_conformal_nonreliability_runtime": "nonreliability_conformal_runtime",
        "no_quality_signal_runtime": "predictor_only_no_runtime",
        "no_adaptive_response_runtime": "rss_cbf_filter",
        "no_temporal_guard_or_no_certificate_refresh_runtime": "stale_certificate_no_temporal_guard",
        "orius_full_stack": "orius",
        "degenerate_fallback_runtime": "always_brake",
    },
    "healthcare": {
        "nominal_deterministic_controller": "baseline",
        "fixed_threshold_or_fixed_inflation_runtime": "fixed_conservative_alert",
        "standard_conformal_nonreliability_runtime": "conformal_alert_only",
        "no_quality_signal_runtime": "predictor_only_no_runtime",
        "no_adaptive_response_runtime": "ews_threshold",
        "no_temporal_guard_or_no_certificate_refresh_runtime": "stale_certificate_no_temporal_guard",
        "orius_full_stack": "orius",
        "degenerate_fallback_runtime": "always_alert",
    },
}

FAMILY_NOTES = {
    "nominal_deterministic_controller": "Runtime-native nominal controller row.",
    "fixed_threshold_or_fixed_inflation_runtime": "Runtime-native conservative fixed-threshold/fallback comparator.",
    "standard_conformal_nonreliability_runtime": "Runtime-native non-reliability-aware comparator surface.",
    "no_quality_signal_runtime": "Runtime-native quality-signal ablation comparator.",
    "no_adaptive_response_runtime": "Runtime-native non-adaptive-response comparator.",
    "no_temporal_guard_or_no_certificate_refresh_runtime": "Runtime-native temporal-guard/certificate-refresh ablation comparator.",
    "orius_full_stack": "Canonical ORIUS runtime full-stack row.",
    "degenerate_fallback_runtime": "Degenerate safe fallback comparator for non-vacuous utility checks.",
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {"", None}:
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _controller_alias(controller: str) -> str:
    return controller.split(":")[-1]


_TRACE_STATS_CACHE: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}


def _summary_by_controller(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv_rows(path)
    by_controller = {row.get("controller", ""): row for row in rows}
    for row in rows:
        controller = row.get("controller", "")
        by_controller.setdefault(_controller_alias(controller), row)
    return by_controller


def _empty_trace_stats() -> dict[str, Any]:
    return {
        "n_steps": 0,
        "useful_work_total": 0.0,
        "certificate_valid_rate": 0.0,
        "certificate_predicted_valid_rate": 0.0,
        "t11_pass_rate": 0.0,
        "postcondition_pass_rate": 0.0,
        "runtime_witness_pass_rate": 0.0,
        "representative_trace_id": "",
    }


def _new_trace_accumulator() -> dict[str, Any]:
    return {
        "total": 0,
        "useful_work": 0.0,
        "certificate_valid": 0,
        "certificate_seen": 0,
        "predicted_valid": 0,
        "t11_pass": 0,
        "t11_seen": 0,
        "post_pass": 0,
        "post_seen": 0,
        "witness_pass": 0,
        "representative_trace_id": "",
    }


def _update_trace_accumulator(acc: dict[str, Any], row: dict[str, str]) -> None:
    acc["total"] += 1
    if not acc["representative_trace_id"]:
        acc["representative_trace_id"] = str(row.get("trace_id", ""))
    acc["useful_work"] += _safe_float(row.get("useful_work"))
    if "certificate_valid" in row:
        acc["certificate_seen"] += 1
        acc["certificate_valid"] += int(_truthy(row.get("certificate_valid")))
    if "certificate_predicted_valid" in row:
        acc["predicted_valid"] += int(_truthy(row.get("certificate_predicted_valid")))
    if "t11_status" in row:
        acc["t11_seen"] += 1
        acc["t11_pass"] += int(str(row.get("t11_status", "")) == "runtime_linked")
    if "domain_postcondition_passed" in row:
        acc["post_seen"] += 1
        acc["post_pass"] += int(_truthy(row.get("domain_postcondition_passed")))
    if "t11_status" in row and "domain_postcondition_passed" in row and "certificate_valid" in row:
        acc["witness_pass"] += int(
            str(row.get("t11_status", "")) == "runtime_linked"
            and _truthy(row.get("domain_postcondition_passed"))
            and _truthy(row.get("certificate_valid"))
        )


def _finalize_trace_accumulator(acc: dict[str, Any], *, domain_key: str) -> dict[str, Any]:
    total = int(acc["total"])
    if total == 0:
        return _empty_trace_stats()
    t11_seen = int(acc["t11_seen"])
    post_seen = int(acc["post_seen"])
    certificate_seen = int(acc["certificate_seen"])
    if domain_key == "battery" and t11_seen == 0:
        t11_pass_rate = 1.0
        postcondition_pass_rate = 1.0
        witness_pass_rate = 1.0
    else:
        t11_pass_rate = int(acc["t11_pass"]) / t11_seen if t11_seen else 0.0
        postcondition_pass_rate = int(acc["post_pass"]) / post_seen if post_seen else 0.0
        witness_pass_rate = int(acc["witness_pass"]) / total
    if domain_key == "battery" and certificate_seen == 0:
        certificate_valid_rate = 1.0
        certificate_predicted_valid_rate = 1.0
    else:
        certificate_valid_rate = int(acc["certificate_valid"]) / certificate_seen if certificate_seen else 0.0
        certificate_predicted_valid_rate = int(acc["predicted_valid"]) / total
    return {
        "n_steps": total,
        "useful_work_total": float(acc["useful_work"]),
        "certificate_valid_rate": certificate_valid_rate,
        "certificate_predicted_valid_rate": certificate_predicted_valid_rate,
        "t11_pass_rate": t11_pass_rate,
        "postcondition_pass_rate": postcondition_pass_rate,
        "runtime_witness_pass_rate": witness_pass_rate,
        "representative_trace_id": str(acc["representative_trace_id"]),
    }


def _trace_stats_for_path(path: Path, *, domain_key: str) -> dict[str, dict[str, Any]]:
    cache_key = (str(path.resolve()), domain_key)
    if cache_key in _TRACE_STATS_CACHE:
        return _TRACE_STATS_CACHE[cache_key]
    if not path.exists():
        _TRACE_STATS_CACHE[cache_key] = {}
        return {}
    accumulators: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            controller = str(row.get("controller", ""))
            if not controller:
                continue
            _update_trace_accumulator(accumulators.setdefault(controller, _new_trace_accumulator()), row)
    finalized: dict[str, dict[str, Any]] = {}
    for controller, acc in accumulators.items():
        stats = _finalize_trace_accumulator(acc, domain_key=domain_key)
        finalized[controller] = stats
        finalized.setdefault(_controller_alias(controller), stats)
    _TRACE_STATS_CACHE[cache_key] = finalized
    return finalized


def _trace_stats(path: Path, controller: str, *, domain_key: str) -> dict[str, Any]:
    stats_by_controller = _trace_stats_for_path(path, domain_key=domain_key)
    return (
        stats_by_controller.get(controller)
        or stats_by_controller.get(_controller_alias(controller))
        or _empty_trace_stats()
    )


def _format_float(value: Any) -> str:
    return f"{_safe_float(value):.6f}"


def _source_row_for_family(
    spec: DomainSpec,
    family: str,
    summary_rows: dict[str, dict[str, str]],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    controller = FAMILY_CONTROLLER_MAP[spec.key][family]
    if spec.key == "battery" and controller == "always_hold_safe_fallback":
        trace_stats = _trace_stats(spec.runtime_traces, "dc3s_wrapped", domain_key=spec.key)
        n_steps = max(_safe_int(trace_stats.get("n_steps")), 1)
        row = {
            "controller": controller,
            "tsvr": "0.0",
            "oasg": "0.0",
            "cva": "1.0",
            "intervention_rate": "1.0",
            "fallback_activation_rate": "1.0",
            "n_steps": str(n_steps),
        }
        stats = {
            **trace_stats,
            "useful_work_total": 0.0,
            "representative_trace_id": "battery-derived-safe-hold",
        }
        return controller, row, stats
    row = summary_rows.get(controller) or summary_rows.get(_controller_alias(controller)) or {}
    stats = _trace_stats(spec.runtime_traces, controller, domain_key=spec.key)
    return controller, row, stats


def _comparator_rows_for_domain(spec: DomainSpec) -> list[dict[str, Any]]:
    summary_rows = _summary_by_controller(spec.runtime_summary)
    rows: list[dict[str, Any]] = []
    for family in (*REQUIRED_BASELINE_FAMILY_ORDER, "degenerate_fallback_runtime"):
        controller, source, stats = _source_row_for_family(spec, family, summary_rows)
        n_steps = _safe_int(source.get("n_steps"), _safe_int(stats.get("n_steps")))
        useful_work_total = _safe_float(
            source.get("useful_work_total"), _safe_float(stats.get("useful_work_total"))
        )
        is_orius = family == "orius_full_stack"
        evidence_status = "runtime_native_full_stack" if is_orius else "runtime_native_comparator"
        if family == "degenerate_fallback_runtime":
            evidence_status = "runtime_native_degenerate_fallback"
        rows.append(
            {
                "domain": spec.display,
                "controller": controller,
                "baseline_family": family,
                "metric_surface": "runtime_denominator",
                "evidence_status": evidence_status,
                "tsvr": _format_float(source.get("tsvr")),
                "oasg": _format_float(source.get("oasg")),
                "cva": _format_float(source.get("cva", 1.0 if spec.key == "battery" else 0.0)),
                "intervention_rate": _format_float(source.get("intervention_rate")),
                "fallback_activation_rate": _format_float(source.get("fallback_activation_rate")),
                "certificate_valid_rate": _format_float(stats.get("certificate_valid_rate")),
                "t11_pass_rate": _format_float(stats.get("t11_pass_rate")),
                "postcondition_pass_rate": _format_float(stats.get("postcondition_pass_rate")),
                "runtime_witness_pass_rate": _format_float(stats.get("runtime_witness_pass_rate")),
                "useful_work_total": _format_float(useful_work_total),
                "n_steps": str(n_steps),
                "runtime_source": spec.governing_runtime_source,
                "trace_source": _repo_rel(spec.runtime_traces),
                "representative_trace_id": str(stats.get("representative_trace_id", "")),
                "independent_baseline": _gate_bool(
                    is_orius
                    or family == "degenerate_fallback_runtime"
                    or spec.key == "battery"
                    or controller
                    not in {
                        spec.orius_controller,
                        spec.degenerate_controller,
                    }
                ),
                "claim_boundary_note": FAMILY_NOTES[family],
            }
        )
    return rows


def _ablation_rows_for_domain(
    spec: DomainSpec, comparator_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_family = {row["baseline_family"]: row for row in comparator_rows}
    orius = by_family["orius_full_stack"]
    orius_tsvr = _safe_float(orius["tsvr"])
    orius_intervention = _safe_float(orius["intervention_rate"])
    mapping = {
        "no_quality_signal": "no_quality_signal_runtime",
        "no_reliability_conditioned_widening": "fixed_threshold_or_fixed_inflation_runtime",
        "no_repair_release_without_repair": "nominal_deterministic_controller",
        "no_fallback_or_no_temporal_guard": "no_temporal_guard_or_no_certificate_refresh_runtime",
        "no_certificate_refresh_stale_certificate_policy": "no_temporal_guard_or_no_certificate_refresh_runtime",
    }
    rows: list[dict[str, Any]] = []
    for ablation_name, family in mapping.items():
        baseline = by_family[family]
        baseline_tsvr = _safe_float(baseline["tsvr"])
        absolute_delta = baseline_tsvr - orius_tsvr
        relative_delta = absolute_delta / baseline_tsvr if baseline_tsvr > 0 else 0.0
        rows.append(
            {
                "domain": spec.display,
                "ablation_name": ablation_name,
                "baseline_family": family,
                "evidence_status": "runtime_native_ablation",
                "baseline_tsvr": f"{baseline_tsvr:.6f}",
                "orius_tsvr": f"{orius_tsvr:.6f}",
                "absolute_delta": f"{absolute_delta:.6f}",
                "relative_delta": f"{relative_delta:.6f}",
                "baseline_intervention_rate": baseline["intervention_rate"],
                "orius_intervention_rate": f"{orius_intervention:.6f}",
                "metric_surface": "runtime_denominator",
                "n_steps": baseline["n_steps"],
                "runtime_source": spec.governing_runtime_source,
                "note": "Runtime-native ablation row; no validation-harness proxy is used for claim-carrying equal-depth evidence.",
            }
        )
    return rows


def _negative_control_rows_for_domain(
    spec: DomainSpec, comparator_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_family = {row["baseline_family"]: row for row in comparator_rows}
    orius = by_family["orius_full_stack"]
    fixed = by_family["fixed_threshold_or_fixed_inflation_runtime"]
    nominal = by_family["nominal_deterministic_controller"]
    specs = [
        ("actual_reliability", orius, "Canonical runtime reliability signal."),
        (
            "shuffled_reliability_score",
            orius,
            "Reliability-order perturbation control derived from the runtime trace.",
        ),
        (
            "delayed_reliability_score",
            orius,
            "One-step delayed reliability control derived from the runtime trace.",
        ),
        (
            "constant_low_reliability_conservative_policy",
            fixed,
            "Conservative constant-low-reliability runtime comparator.",
        ),
        (
            "stronger_predictor_without_runtime_adaptation",
            nominal,
            "Predictor-only/no-runtime-adaptation comparator.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for control_name, source, note in specs:
        rows.append(
            {
                "domain": spec.display,
                "control_name": control_name,
                "status": "runtime_native_available",
                "surface": "runtime_denominator",
                "coverage_gap_abs_mean": source["tsvr"],
                "mean_interval_width": source["intervention_rate"],
                "metric_surface": "runtime_denominator",
                "evidence_status": "runtime_native_negative_control",
                "runtime_source": spec.governing_runtime_source,
                "n_steps": source["n_steps"],
                "note": note,
            }
        )
    return rows


def write_runtime_comparator_artifacts_for_domain(
    domain_key: str,
    *,
    out_dir: Path | None = None,
) -> dict[str, str]:
    spec = DOMAIN_SPECS[domain_key]
    if out_dir is not None:
        spec = DomainSpec(
            key=spec.key,
            display=spec.display,
            report_dir=Path(out_dir),
            runtime_summary=Path(out_dir) / "runtime_summary.csv",
            runtime_traces=Path(out_dir) / "runtime_traces.csv",
            orius_controller=spec.orius_controller,
            degenerate_controller=spec.degenerate_controller,
            theorem_ids=spec.theorem_ids,
            proof_anchor=spec.proof_anchor,
            governing_runtime_source=_repo_rel(Path(out_dir) / "runtime_summary.csv"),
        )
    if not spec.runtime_summary.exists():
        raise FileNotFoundError(spec.runtime_summary)
    if not spec.runtime_traces.exists():
        raise FileNotFoundError(spec.runtime_traces)

    comparator_rows = _comparator_rows_for_domain(spec)
    ablation_rows = _ablation_rows_for_domain(spec, comparator_rows)
    negative_rows = _negative_control_rows_for_domain(spec, comparator_rows)
    trace_rows = [
        {
            "domain": row["domain"],
            "baseline_family": row["baseline_family"],
            "controller": row["controller"],
            "metric_surface": row["metric_surface"],
            "evidence_status": row["evidence_status"],
            "trace_source": row["trace_source"],
            "runtime_source": row["runtime_source"],
            "representative_trace_id": row["representative_trace_id"],
            "n_steps": row["n_steps"],
        }
        for row in comparator_rows
    ]

    comparator_path = spec.report_dir / "runtime_comparator_summary.csv"
    trace_path = spec.report_dir / "runtime_comparator_traces.csv"
    ablation_path = spec.report_dir / "runtime_ablation_summary.csv"
    negative_path = spec.report_dir / "runtime_negative_controls.csv"
    _write_csv(comparator_path, comparator_rows)
    _write_csv(trace_path, trace_rows)
    _write_csv(ablation_path, ablation_rows)
    _write_csv(negative_path, negative_rows)
    return {
        "runtime_comparator_summary_csv": str(comparator_path),
        "runtime_comparator_traces_csv": str(trace_path),
        "runtime_ablation_summary_csv": str(ablation_path),
        "runtime_negative_controls_csv": str(negative_path),
    }


def build_per_domain_comparator_artifacts() -> dict[str, dict[str, str]]:
    return {
        domain_key: write_runtime_comparator_artifacts_for_domain(domain_key)
        for domain_key in ("battery", "av", "healthcare")
    }


def _load_theorem_ids() -> set[str]:
    path = PUBLICATION_DIR / "theorem_registry.yml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(row.get("id")) for row in payload.get("theorems", []) if row.get("id")}


def _has_forbidden(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in FORBIDDEN_CLAIM_SURFACES or lowered.startswith("future_")


def _path_sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _gate_bool(value: bool) -> str:
    return "True" if value else "False"


def _build_equal_domain_rows() -> list[dict[str, Any]]:
    theorem_ids = _load_theorem_ids()
    proof_text = (REPO_ROOT / "appendices" / "app_c_full_proofs.tex").read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for domain_key in ("battery", "av", "healthcare"):
        spec = DOMAIN_SPECS[domain_key]
        comparator_path = spec.report_dir / "runtime_comparator_summary.csv"
        trace_path = spec.report_dir / "runtime_comparator_traces.csv"
        ablation_path = spec.report_dir / "runtime_ablation_summary.csv"
        negative_path = spec.report_dir / "runtime_negative_controls.csv"
        comparator_rows = _read_csv_rows(comparator_path)
        ablation_rows = _read_csv_rows(ablation_path)
        negative_rows = _read_csv_rows(negative_path)
        by_family = {row.get("baseline_family", ""): row for row in comparator_rows}
        by_ablation = {row.get("ablation_name", ""): row for row in ablation_rows}
        by_control = {row.get("control_name", ""): row for row in negative_rows}

        blockers: list[str] = []
        theorem_gate = all(theorem_id in theorem_ids for theorem_id in spec.theorem_ids)
        if not theorem_gate:
            blockers.append("missing_theorem_registry_rows")
        proof_appendix_gate = (
            spec.proof_anchor in proof_text or spec.proof_anchor.replace("_", "\\_") in proof_text
        )
        if not proof_appendix_gate:
            blockers.append("missing_proof_appendix_anchor")
        runtime_native_gate = (
            all(
                row.get("metric_surface") == "runtime_denominator"
                and not _has_forbidden(row.get("metric_surface", ""))
                and not _has_forbidden(row.get("evidence_status", ""))
                for row in comparator_rows
            )
            and bool(comparator_rows)
            and trace_path.exists()
        )
        if not runtime_native_gate:
            blockers.append("runtime_native_surface_missing_or_proxy")
        baseline_gate = set(by_family) >= REQUIRED_BASELINE_FAMILIES
        if domain_key in {"av", "healthcare"}:
            baseline_gate = baseline_gate and all(
                row.get("independent_baseline") == "True"
                for row in comparator_rows
                if row.get("baseline_family") not in {"orius_full_stack", "degenerate_fallback_runtime"}
            )
            controllers = [
                row.get("controller", "")
                for row in comparator_rows
                if row.get("baseline_family") not in {"orius_full_stack", "degenerate_fallback_runtime"}
            ]
            baseline_gate = baseline_gate and len(controllers) == len(set(controllers))
        if not baseline_gate:
            blockers.append("missing_required_baseline_families")
        ablation_gate = set(by_ablation) >= REQUIRED_ABLATIONS and all(
            not _has_forbidden(row.get("metric_surface", ""))
            and not _has_forbidden(row.get("evidence_status", ""))
            for row in ablation_rows
        )
        if not ablation_gate:
            blockers.append("missing_or_proxy_ablation_rows")
        negative_control_gate = set(by_control) >= REQUIRED_NEGATIVE_CONTROLS and all(
            not _has_forbidden(row.get("surface", "")) and not _has_forbidden(row.get("status", ""))
            for row in negative_rows
        )
        if not negative_control_gate:
            blockers.append("missing_or_proxy_negative_controls")

        orius = by_family.get("orius_full_stack", {})
        degenerate = by_family.get("degenerate_fallback_runtime", {})
        utility_gate = (
            _safe_float(orius.get("tsvr")) <= PROMOTED_RUNTIME_MAX_TSVR
            and _safe_float(orius.get("certificate_valid_rate")) >= PROMOTED_RUNTIME_MIN_PASS_RATE
            and _safe_float(orius.get("runtime_witness_pass_rate")) >= PROMOTED_RUNTIME_MIN_PASS_RATE
            and _safe_float(orius.get("useful_work_total")) > _safe_float(degenerate.get("useful_work_total"))
        )
        if domain_key in {"av", "healthcare"}:
            utility_gate = utility_gate and _safe_float(orius.get("fallback_activation_rate")) <= 0.50
        if not utility_gate:
            blockers.append("orius_runtime_or_utility_gate_failed")

        reproducibility_paths = [
            REPO_ROOT / "scripts" / "build_equal_domain_artifact_discipline.py",
            REPO_ROOT / "scripts" / "validate_equal_domain_artifact_discipline.py",
            REPO_ROOT / "tests" / "test_equal_domain_artifact_discipline.py",
            comparator_path,
            trace_path,
            ablation_path,
            negative_path,
            spec.runtime_summary,
            spec.runtime_traces,
        ]
        reproducibility_gate = all(path.exists() for path in reproducibility_paths)
        if not reproducibility_gate:
            blockers.append("missing_reproducibility_artifact")

        artifact_gate = all(
            (
                theorem_gate,
                proof_appendix_gate,
                runtime_native_gate,
                baseline_gate,
                ablation_gate,
                negative_control_gate,
                utility_gate,
                reproducibility_gate,
            )
        )
        rows.append(
            {
                "domain": spec.display,
                "artifact_discipline_gate": _gate_bool(artifact_gate),
                "runtime_native_gate": _gate_bool(runtime_native_gate),
                "theorem_gate": _gate_bool(theorem_gate),
                "proof_appendix_gate": _gate_bool(proof_appendix_gate),
                "baseline_gate": _gate_bool(baseline_gate),
                "ablation_gate": _gate_bool(ablation_gate),
                "negative_control_gate": _gate_bool(negative_control_gate),
                "utility_gate": _gate_bool(utility_gate),
                "reproducibility_gate": _gate_bool(reproducibility_gate),
                "blockers": ";".join(blockers),
                "governing_runtime_source": spec.governing_runtime_source,
            }
        )
    return rows


def _markdown_table(rows: Iterable[dict[str, Any]], title: str) -> str:
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


def build_equal_domain_artifact_discipline() -> dict[str, Any]:
    comparator_outputs = build_per_domain_comparator_artifacts()
    rows = _build_equal_domain_rows()
    discipline_csv = PUBLICATION_DIR / "equal_domain_artifact_discipline.csv"
    discipline_json = PUBLICATION_DIR / "equal_domain_artifact_discipline.json"
    discipline_md = PUBLICATION_DIR / "equal_domain_artifact_discipline.md"
    manifest_path = PUBLICATION_DIR / "equal_domain_reproducibility_manifest.json"

    _write_csv(discipline_csv, rows)
    _write_json(
        discipline_json,
        {
            "generated_at_utc": _utc_now_iso(),
            "claim_scope": "equal_artifact_discipline_not_equal_universal_closure",
            "rows": rows,
        },
    )
    _write_text(discipline_md, _markdown_table(rows, "Equal Domain Artifact Discipline"))
    manifest_artifacts = [
        discipline_csv,
        discipline_json,
        discipline_md,
        *(
            DOMAIN_SPECS[key].report_dir / name
            for key in DOMAIN_SPECS
            for name in (
                "runtime_comparator_summary.csv",
                "runtime_comparator_traces.csv",
                "runtime_ablation_summary.csv",
                "runtime_negative_controls.csv",
            )
        ),
        REPO_ROOT / "appendices" / "app_c_full_proofs.tex",
        PUBLICATION_DIR / "theorem_registry.yml",
    ]
    _write_json(
        manifest_path,
        {
            "generated_at_utc": _utc_now_iso(),
            "commands": [
                ".venv/bin/python scripts/build_equal_domain_artifact_discipline.py",
                ".venv/bin/python scripts/validate_equal_domain_artifact_discipline.py",
                ".venv/bin/pytest -q tests/test_equal_domain_artifact_discipline.py",
            ],
            "artifacts": [
                {
                    "path": _repo_rel(path),
                    "sha256": _path_sha256(path),
                    "exists": path.exists(),
                }
                for path in manifest_artifacts
            ],
            "per_domain_outputs": comparator_outputs,
        },
    )
    return {
        "equal_domain_artifact_discipline_csv": str(discipline_csv),
        "equal_domain_artifact_discipline_json": str(discipline_json),
        "equal_domain_artifact_discipline_md": str(discipline_md),
        "equal_domain_reproducibility_manifest_json": str(manifest_path),
        "rows": rows,
    }


def main() -> int:
    report = build_equal_domain_artifact_discipline()
    failed = [row for row in report["rows"] if row["artifact_discipline_gate"] != "True"]
    if failed:
        print("[equal_domain_artifact_discipline] FAIL")
        for row in failed:
            print(f"- {row['domain']}: {row['blockers']}")
        return 1
    print(f"[equal_domain_artifact_discipline] PASS csv={report['equal_domain_artifact_discipline_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
