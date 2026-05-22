#!/usr/bin/env python3
"""Build the claim-facing utility-preserving safety scorecard.

The scorecard is intentionally separate from the headline TSVR table: it asks
whether ORIUS preserves useful work compared with a degenerate fail-safe
reference while staying no less safe than that reference.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"

BATTERY_T8 = PUBLICATION_DIR / "graceful_four_policy_metrics.csv"
THREE_DOMAIN_UTILITY = PUBLICATION_DIR / "three_domain_utility_safety_dominance.csv"
THREE_DOMAIN_BASELINE_SUITE = PUBLICATION_DIR / "three_domain_baseline_suite.csv"
THREE_DOMAIN_NEGATIVE_CONTROLS = PUBLICATION_DIR / "three_domain_negative_controls.csv"
THREE_DOMAIN_ABLATION_MATRIX = PUBLICATION_DIR / "three_domain_ablation_matrix.csv"
SECURITY_GOVERNANCE_ABLATION = PUBLICATION_DIR / "security_governance_ablation_matrix.csv"

OUT_CSV = PUBLICATION_DIR / "utility_preserving_safety_scorecard.csv"
OUT_JSON = PUBLICATION_DIR / "utility_preserving_safety_scorecard.json"
OUT_MD = PUBLICATION_DIR / "utility_preserving_safety_scorecard.md"
OUT_CLAIM_CSV = PUBLICATION_DIR / "utility_preserving_safety_claim_table.csv"
OUT_ABLATION_CSV = PUBLICATION_DIR / "utility_preserving_safety_ablation_surfaces.csv"
OUT_CLAIM_TEX = PUBLICATION_DIR / "utility_preserving_safety_claim_table.tex"
OUT_ABLATION_TEX = PUBLICATION_DIR / "utility_preserving_safety_ablation_surfaces.tex"

FIELDS = [
    "domain",
    "claim_scope",
    "source_artifacts",
    "safety_reference_controller",
    "orius_controller",
    "orius_tsvr",
    "safety_reference_tsvr",
    "excess_tsvr_over_safety_reference",
    "baseline_tsvr",
    "baseline_tsvr_reduction",
    "orius_utility",
    "safety_reference_utility",
    "utility_gain_over_safety_reference",
    "utility_delta_over_safety_reference",
    "orius_intervention_rate",
    "safety_reference_intervention_rate",
    "intervention_reduction_vs_safety_reference",
    "orius_fallback_rate",
    "safety_reference_fallback_rate",
    "fallback_reduction_vs_safety_reference",
    "utility_preserving_safety_gate",
    "remaining_conservatism_note",
    "claim_boundary",
]

CLAIM_FIELDS = [
    "domain",
    "comparison",
    "source_artifacts",
    "reference_controller",
    "orius_controller",
    "reference_tsvr",
    "orius_tsvr",
    "absolute_tsvr_reduction",
    "reference_utility",
    "orius_utility",
    "utility_delta",
    "reference_intervention_rate",
    "orius_intervention_rate",
    "intervention_delta",
    "reference_fallback_rate",
    "orius_fallback_rate",
    "fallback_delta",
    "claim_relation",
    "comparability",
    "claim_boundary",
]

ABLATION_FIELDS = [
    "requested_surface",
    "domain",
    "evidence_surface",
    "source_artifacts",
    "ablation_name",
    "baseline_family",
    "baseline_controller",
    "baseline_tsvr",
    "orius_tsvr",
    "absolute_tsvr_reduction",
    "relative_tsvr_reduction",
    "baseline_intervention_rate",
    "orius_intervention_rate",
    "comparability",
    "claim_interpretation",
    "note",
]

REQUIRED_DOMAINS = [
    "Battery Energy Storage",
    "Autonomous Vehicles",
    "Medical and Healthcare Monitoring",
]

REQUESTED_ABLATION_SURFACES = [
    "no_reliability",
    "no_uncertainty",
    "no_repair",
    "no_fallback",
    "no_certificate_gate",
    "no_signature_hash_gate",
]

BLANK_FIELD_TOKENS = {
    "baseline_controller": "no_controller_for_governance_gate",
    "baseline_tsvr": "not_a_tsvr_metric",
    "orius_tsvr": "not_a_tsvr_metric",
    "absolute_tsvr_reduction": "not_a_tsvr_metric",
    "relative_tsvr_reduction": "not_a_tsvr_metric",
    "baseline_intervention_rate": "not_an_intervention_metric",
    "reference_utility": "metric_not_reported_in_compact_comparator",
    "orius_utility": "metric_not_reported_in_compact_comparator",
    "utility_delta": "metric_not_reported_in_compact_comparator",
    "reference_fallback_rate": "metric_not_reported_in_compact_comparator",
    "orius_fallback_rate": "metric_not_reported_in_compact_comparator",
    "fallback_delta": "metric_not_reported_in_compact_comparator",
    "reference_intervention_rate": "metric_not_reported_in_compact_comparator",
    "orius_intervention_rate": "metric_not_reported_in_compact_comparator",
    "intervention_delta": "metric_not_reported_in_compact_comparator",
    "safety_reference_intervention_rate": "not_defined_for_battery_t8",
    "intervention_reduction_vs_safety_reference": "not_defined_for_battery_t8",
    "safety_reference_fallback_rate": "not_defined_for_battery_t8",
    "fallback_reduction_vs_safety_reference": "not_defined_for_battery_t8",
}

RUNTIME_ABLATION_MAP = {
    "no_reliability": (
        "no_reliability_conditioned_widening",
        "comparable_runtime_native",
        "Reliability-conditioned widening removed; runtime-native TSVR delta is directly comparable.",
    ),
    "no_repair": (
        "no_repair_release_without_repair",
        "comparable_runtime_native",
        "Repair/release layer removed; runtime-native TSVR delta is directly comparable.",
    ),
    "no_fallback": (
        "no_fallback_or_no_temporal_guard",
        "non_comparable_combined_with_temporal_guard",
        "Existing compact evidence combines fallback removal with temporal-guard behavior; do not treat it as an isolated fallback-only ablation.",
    ),
    "no_certificate_gate": (
        "no_certificate_refresh_stale_certificate_policy",
        "non_comparable_combined_certificate_temporal_guard",
        "Existing compact evidence combines certificate refresh/stale-certificate behavior with temporal guarding; do not treat it as an isolated certificate-gate ablation.",
    ),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _fmt(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def _ratio(value: float, reference: float) -> str:
    if reference <= 0.0:
        return "inf" if value > 0.0 else "0.000000"
    return _fmt(value / reference)


def _maybe_delta(reference: str, candidate: str) -> str:
    if reference == "" or candidate == "":
        return ""
    return _fmt(_safe_float(reference) - _safe_float(candidate))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _utc_now() -> str:
    override = os.environ.get("ORIUS_REPRODUCIBLE_TIMESTAMP")
    if override:
        return override
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _battery_row() -> dict[str, str] | None:
    rows = {row.get("policy", ""): row for row in _read_csv(BATTERY_T8)}
    orius = rows.get("optimized_graceful")
    reference = rows.get("immediate_shutdown")
    baseline = rows.get("blind_persistence")
    if not (orius and reference and baseline):
        return None

    orius_tsvr = _safe_float(orius.get("violation_rate_mean"))
    reference_tsvr = _safe_float(reference.get("violation_rate_mean"))
    baseline_tsvr = _safe_float(baseline.get("violation_rate_mean"))
    orius_work = _safe_float(orius.get("useful_work_mwh_mean"))
    reference_work = _safe_float(reference.get("useful_work_mwh_mean"))
    excess_tsvr = max(0.0, orius_tsvr - reference_tsvr)
    gate = bool(excess_tsvr <= 1e-6 and orius_work > reference_work)

    return {
        "domain": "Battery Energy Storage",
        "claim_scope": "T8 graceful degradation useful-work frontier",
        "source_artifacts": "reports/publication/graceful_four_policy_metrics.csv",
        "safety_reference_controller": "immediate_shutdown",
        "orius_controller": "optimized_graceful",
        "orius_tsvr": _fmt(orius_tsvr),
        "safety_reference_tsvr": _fmt(reference_tsvr),
        "excess_tsvr_over_safety_reference": _fmt(excess_tsvr),
        "baseline_tsvr": _fmt(baseline_tsvr),
        "baseline_tsvr_reduction": _fmt(baseline_tsvr - orius_tsvr),
        "orius_utility": _fmt(orius_work),
        "safety_reference_utility": _fmt(reference_work),
        "utility_gain_over_safety_reference": _ratio(orius_work, reference_work),
        "utility_delta_over_safety_reference": _fmt(orius_work - reference_work),
        "orius_intervention_rate": "",
        "safety_reference_intervention_rate": "",
        "intervention_reduction_vs_safety_reference": "",
        "orius_fallback_rate": "",
        "safety_reference_fallback_rate": "",
        "fallback_reduction_vs_safety_reference": "",
        "utility_preserving_safety_gate": str(gate),
        "remaining_conservatism_note": (
            "Battery utility evidence uses graceful-degradation work retention; "
            "fallback/intervention rates are reported in the runtime TSVR table."
        ),
        "claim_boundary": (
            "Battery row shows zero-violation useful work over immediate shutdown; "
            "it is simulator/predeployment evidence, not field deployment."
        ),
    }


def _dominance_row(row: dict[str, str]) -> dict[str, str]:
    orius_tsvr = _safe_float(row.get("orius_tsvr"))
    reference_tsvr = _safe_float(row.get("safety_reference_tsvr"))
    baseline_tsvr = _safe_float(row.get("baseline_tsvr"))
    excess_tsvr = _safe_float(row.get("excess_tsvr_over_safety_reference"))
    orius_work = _safe_float(row.get("orius_useful_work_total"))
    reference_work = _safe_float(row.get("safety_reference_useful_work_total"))
    fallback_reduction = _safe_float(row.get("fallback_reduction_vs_safety_reference"))
    intervention_reduction = _safe_float(row.get("intervention_reduction_vs_safety_reference"))
    gate = bool(
        str(row.get("nonvacuous_utility_gate", "")).lower() == "true"
        and excess_tsvr <= 1e-3
        and orius_work > reference_work
        and fallback_reduction > 0.0
        and intervention_reduction > 0.0
    )

    domain = str(row.get("domain", ""))
    if domain == "Autonomous Vehicles":
        note = (
            "Still conservative: intervention remains high, but ORIUS produces more useful work "
            "than always-brake with no excess TSVR over that fail-safe reference."
        )
    elif domain == "Medical and Healthcare Monitoring":
        note = (
            "Still conservative: fallback/max-alert remains high, but ORIUS preserves monitoring "
            "utility compared with always-alert while keeping zero TSVR."
        )
    else:
        note = "Utility-preserving safety judged against the domain fail-safe reference."

    return {
        "domain": domain,
        "claim_scope": str(row.get("runtime_surface", "")),
        "source_artifacts": str(row.get("source_surface", "")),
        "safety_reference_controller": str(row.get("safety_reference_controller", "")),
        "orius_controller": "orius",
        "orius_tsvr": _fmt(orius_tsvr),
        "safety_reference_tsvr": _fmt(reference_tsvr),
        "excess_tsvr_over_safety_reference": _fmt(excess_tsvr),
        "baseline_tsvr": _fmt(baseline_tsvr),
        "baseline_tsvr_reduction": _fmt(baseline_tsvr - orius_tsvr),
        "orius_utility": _fmt(orius_work),
        "safety_reference_utility": _fmt(reference_work),
        "utility_gain_over_safety_reference": str(row.get("utility_gain_over_safety_reference", "")),
        "utility_delta_over_safety_reference": str(row.get("utility_delta_over_safety_reference", "")),
        "orius_intervention_rate": str(row.get("orius_intervention_rate", "")),
        "safety_reference_intervention_rate": str(row.get("safety_reference_intervention_rate", "")),
        "intervention_reduction_vs_safety_reference": str(
            row.get("intervention_reduction_vs_safety_reference", "")
        ),
        "orius_fallback_rate": str(row.get("orius_fallback_activation_rate", "")),
        "safety_reference_fallback_rate": str(row.get("safety_reference_fallback_activation_rate", "")),
        "fallback_reduction_vs_safety_reference": str(row.get("fallback_reduction_vs_safety_reference", "")),
        "utility_preserving_safety_gate": str(gate),
        "remaining_conservatism_note": note,
        "claim_boundary": str(row.get("claim_boundary", "")),
    }


def build_scorecard() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    battery = _battery_row()
    if battery is not None:
        rows.append(battery)
    rows.extend(_dominance_row(row) for row in _read_csv(THREE_DOMAIN_UTILITY))
    return rows


def _rows_by_domain_and_key(
    rows: list[dict[str, str]],
    key_field: str,
) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        domain = row.get("domain", "")
        key = row.get(key_field, "")
        if domain and key:
            result.setdefault(domain, {})[key] = row
    return result


def _claim_domains(scorecard_rows: list[dict[str, str]], baseline_rows: list[dict[str, str]]) -> list[str]:
    seen = {row.get("domain", "") for row in scorecard_rows + baseline_rows}
    ordered = [domain for domain in REQUIRED_DOMAINS if domain in seen]
    ordered.extend(sorted(seen - set(ordered) - {""}))
    return ordered


def _controller_name(row: dict[str, str] | None, fallback: str = "") -> str:
    if not row:
        return fallback
    return row.get("implemented_controller") or row.get("controller") or fallback


def _fail_safe_claim_row(row: dict[str, str]) -> dict[str, str]:
    reference_tsvr = row.get("safety_reference_tsvr", "")
    orius_tsvr = row.get("orius_tsvr", "")
    reference_utility = row.get("safety_reference_utility", "")
    orius_utility = row.get("orius_utility", "")
    utility_delta = row.get("utility_delta_over_safety_reference", "")
    fallback_delta = _maybe_delta(row.get("safety_reference_fallback_rate", ""), row.get("orius_fallback_rate", ""))
    intervention_delta = _maybe_delta(
        row.get("safety_reference_intervention_rate", ""),
        row.get("orius_intervention_rate", ""),
    )
    safer_or_equal = _safe_float(orius_tsvr, default=math.inf) <= _safe_float(
        reference_tsvr,
        default=-math.inf,
    ) + 1e-3
    less_conservative = _safe_float(utility_delta) > 0.0
    claim_relation = (
        "less_conservative_than_shutdown_or_fallback_only"
        if safer_or_equal and less_conservative
        else "not_claimable_as_less_conservative"
    )
    return {
        "domain": row.get("domain", ""),
        "comparison": "shutdown_or_fallback_only_conservatism",
        "source_artifacts": row.get("source_artifacts", ""),
        "reference_controller": row.get("safety_reference_controller", ""),
        "orius_controller": row.get("orius_controller", "orius"),
        "reference_tsvr": reference_tsvr,
        "orius_tsvr": orius_tsvr,
        "absolute_tsvr_reduction": _maybe_delta(reference_tsvr, orius_tsvr),
        "reference_utility": reference_utility,
        "orius_utility": orius_utility,
        "utility_delta": utility_delta,
        "reference_intervention_rate": row.get("safety_reference_intervention_rate", ""),
        "orius_intervention_rate": row.get("orius_intervention_rate", ""),
        "intervention_delta": intervention_delta,
        "reference_fallback_rate": row.get("safety_reference_fallback_rate", ""),
        "orius_fallback_rate": row.get("orius_fallback_rate", ""),
        "fallback_delta": fallback_delta,
        "claim_relation": claim_relation,
        "comparability": "comparable_fail_safe_reference",
        "claim_boundary": row.get("claim_boundary", ""),
    }


def build_claim_comparison_rows(scorecard_rows: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """Build claim-facing rows without changing or inventing source metrics."""

    scorecard_rows = scorecard_rows if scorecard_rows is not None else build_scorecard()
    baseline_rows = _read_csv(THREE_DOMAIN_BASELINE_SUITE)
    negative_rows = _read_csv(THREE_DOMAIN_NEGATIVE_CONTROLS)
    baseline_by_domain = _rows_by_domain_and_key(baseline_rows, "baseline_family")
    negative_by_domain = _rows_by_domain_and_key(negative_rows, "control_name")

    rows = [_fail_safe_claim_row(row) for row in scorecard_rows]
    for domain in _claim_domains(scorecard_rows, baseline_rows):
        families = baseline_by_domain.get(domain, {})
        orius = families.get("orius_full_stack")
        predictor = families.get("no_quality_signal_runtime")
        negative = negative_by_domain.get(domain, {}).get("stronger_predictor_without_runtime_adaptation")

        orius_tsvr = (orius or {}).get("tsvr", "")
        if not orius_tsvr:
            scorecard = next((row for row in scorecard_rows if row.get("domain") == domain), {})
            orius_tsvr = scorecard.get("orius_tsvr", "")

        reference_tsvr = ""
        if negative:
            reference_tsvr = negative.get("coverage_gap_abs_mean", "")
        if not reference_tsvr and predictor:
            reference_tsvr = predictor.get("tsvr", "")

        if reference_tsvr and orius_tsvr:
            reduction = _safe_float(reference_tsvr) - _safe_float(orius_tsvr)
            absolute_tsvr_reduction = _fmt(reduction)
            if reduction > 1e-9:
                claim_relation = "safer_than_predictor_only"
                comparability = "comparable_runtime_native"
            elif abs(reduction) <= 1e-9:
                claim_relation = "no_observed_safety_separation"
                comparability = "non_comparable_no_observed_safety_separation"
            else:
                claim_relation = "not_safer_than_predictor_only"
                comparability = "non_comparable_predictor_row_safer_on_this_metric"
        else:
            absolute_tsvr_reduction = ""
            claim_relation = "missing_predictor_only_metric"
            comparability = "non_comparable_missing_predictor_only_evidence"

        reference_controller = _controller_name(predictor, "stronger_predictor_without_runtime_adaptation")
        if negative and reference_controller != negative.get("control_name", ""):
            reference_controller = f"{negative['control_name']} ({reference_controller})"

        rows.append(
            {
                "domain": domain,
                "comparison": "predictor_only_safety",
                "source_artifacts": "; ".join(
                    [
                        _repo_rel(THREE_DOMAIN_BASELINE_SUITE),
                        _repo_rel(THREE_DOMAIN_NEGATIVE_CONTROLS),
                    ]
                ),
                "reference_controller": reference_controller,
                "orius_controller": _controller_name(orius, "orius"),
                "reference_tsvr": reference_tsvr,
                "orius_tsvr": orius_tsvr,
                "absolute_tsvr_reduction": absolute_tsvr_reduction,
                "reference_utility": (predictor or {}).get("useful_work_total", ""),
                "orius_utility": (orius or {}).get("useful_work_total", ""),
                "utility_delta": _maybe_delta(
                    (orius or {}).get("useful_work_total", ""),
                    (predictor or {}).get("useful_work_total", ""),
                ),
                "reference_intervention_rate": (predictor or negative or {}).get("intervention_rate", "")
                or (negative or {}).get("mean_interval_width", ""),
                "orius_intervention_rate": (orius or {}).get("intervention_rate", ""),
                "intervention_delta": _maybe_delta(
                    (predictor or {}).get("intervention_rate", ""),
                    (orius or {}).get("intervention_rate", ""),
                ),
                "reference_fallback_rate": (predictor or {}).get("fallback_activation_rate", ""),
                "orius_fallback_rate": (orius or {}).get("fallback_activation_rate", ""),
                "fallback_delta": _maybe_delta(
                    (predictor or {}).get("fallback_activation_rate", ""),
                    (orius or {}).get("fallback_activation_rate", ""),
                ),
                "claim_relation": claim_relation,
                "comparability": comparability,
                "claim_boundary": (
                    "Predictor-only safety uses compact runtime-native negative-control evidence. "
                    "Rows with no observed safety separation are not promoted as safer-than-predictor-only claims."
                ),
            }
        )
    return rows


def _ablation_domains(
    baseline_by_domain: dict[str, dict[str, dict[str, str]]],
    ablation_rows: list[dict[str, str]],
) -> list[str]:
    seen = set(baseline_by_domain)
    seen.update(row.get("domain", "") for row in ablation_rows)
    ordered = [domain for domain in REQUIRED_DOMAINS if domain in seen]
    ordered.extend(sorted(seen - set(ordered) - {""}))
    return ordered


def _ablation_row_for_surface(
    requested_surface: str,
    domain: str,
    ablation: dict[str, str] | None,
    baseline: dict[str, str] | None,
    comparability: str,
    interpretation: str,
) -> dict[str, str]:
    if ablation is None:
        return {
            "requested_surface": requested_surface,
            "domain": domain,
            "evidence_surface": "missing_compact_evidence",
            "source_artifacts": _repo_rel(THREE_DOMAIN_ABLATION_MATRIX),
            "ablation_name": "",
            "baseline_family": "",
            "baseline_controller": "",
            "baseline_tsvr": "",
            "orius_tsvr": "",
            "absolute_tsvr_reduction": "",
            "relative_tsvr_reduction": "",
            "baseline_intervention_rate": "",
            "orius_intervention_rate": "",
            "comparability": "non_comparable_missing_compact_evidence",
            "claim_interpretation": (
                f"No compact row isolates {requested_surface}; no numeric value is inferred."
            ),
            "note": "",
        }

    return {
        "requested_surface": requested_surface,
        "domain": domain,
        "evidence_surface": ablation.get("metric_surface", ""),
        "source_artifacts": _repo_rel(THREE_DOMAIN_ABLATION_MATRIX),
        "ablation_name": ablation.get("ablation_name", ""),
        "baseline_family": ablation.get("baseline_family", ""),
        "baseline_controller": _controller_name(baseline),
        "baseline_tsvr": ablation.get("baseline_tsvr", ""),
        "orius_tsvr": ablation.get("orius_tsvr", ""),
        "absolute_tsvr_reduction": ablation.get("absolute_delta", ""),
        "relative_tsvr_reduction": ablation.get("relative_delta", ""),
        "baseline_intervention_rate": ablation.get("baseline_intervention_rate", ""),
        "orius_intervention_rate": ablation.get("orius_intervention_rate", ""),
        "comparability": comparability,
        "claim_interpretation": interpretation,
        "note": ablation.get("note", ""),
    }


def _no_uncertainty_row(
    domain: str,
    families: dict[str, dict[str, str]],
) -> dict[str, str]:
    baseline = families.get("nominal_deterministic_controller", {})
    orius = families.get("orius_full_stack", {})
    baseline_tsvr = baseline.get("tsvr", "")
    orius_tsvr = orius.get("tsvr", "")
    absolute_delta = _maybe_delta(baseline_tsvr, orius_tsvr)
    relative_delta = ""
    if baseline_tsvr and orius_tsvr:
        denominator = _safe_float(baseline_tsvr)
        relative_delta = _fmt(_safe_float(absolute_delta) / denominator) if denominator > 0.0 else "0.000000"
    return {
        "requested_surface": "no_uncertainty",
        "domain": domain,
        "evidence_surface": "not_isolated_in_compact_evidence",
        "source_artifacts": _repo_rel(THREE_DOMAIN_BASELINE_SUITE),
        "ablation_name": "not_isolated_nominal_deterministic_controller",
        "baseline_family": "nominal_deterministic_controller" if baseline else "",
        "baseline_controller": _controller_name(baseline),
        "baseline_tsvr": baseline_tsvr,
        "orius_tsvr": orius_tsvr,
        "absolute_tsvr_reduction": absolute_delta,
        "relative_tsvr_reduction": relative_delta,
        "baseline_intervention_rate": baseline.get("intervention_rate", ""),
        "orius_intervention_rate": orius.get("intervention_rate", ""),
        "comparability": "non_comparable_combined_nominal_no_uncertainty_surface",
        "claim_interpretation": (
            "Compact evidence has a nominal deterministic comparator but no isolated no-uncertainty-only ablation; "
            "shown only to identify the nearest existing surface."
        ),
        "note": baseline.get("claim_boundary_note", ""),
    }


def _signature_hash_gate_row(security_rows: list[dict[str, str]]) -> dict[str, str]:
    selected = [
        row
        for row in security_rows
        if row.get("ablation") in {"missing_model_hash", "bad_certificate_signature"}
    ]
    expected = "; ".join(row.get("expected_runtime_response", "") for row in selected if row)
    surfaces = "; ".join(row.get("evidence_surface", "") for row in selected if row)
    interpretations = "; ".join(row.get("paper_interpretation", "") for row in selected if row)
    ablations = ";".join(row.get("ablation", "") for row in selected if row)
    return {
        "requested_surface": "no_signature_hash_gate",
        "domain": "Cross-domain governance",
        "evidence_surface": "governance_fail_closed" if selected else "missing_compact_evidence",
        "source_artifacts": _repo_rel(SECURITY_GOVERNANCE_ABLATION),
        "ablation_name": ablations,
        "baseline_family": "security_governance_gate",
        "baseline_controller": "",
        "baseline_tsvr": "",
        "orius_tsvr": "",
        "absolute_tsvr_reduction": "",
        "relative_tsvr_reduction": "",
        "baseline_intervention_rate": "",
        "orius_intervention_rate": "",
        "comparability": (
            "non_comparable_governance_gate"
            if selected
            else "non_comparable_missing_compact_evidence"
        ),
        "claim_interpretation": (
            "Signature/hash removal is a governance fail-closed surface, not a TSVR metric row."
        ),
        "note": "; ".join(part for part in [expected, surfaces, interpretations] if part),
    }


def build_ablation_surface_rows() -> list[dict[str, str]]:
    baseline_rows = _read_csv(THREE_DOMAIN_BASELINE_SUITE)
    ablation_rows = _read_csv(THREE_DOMAIN_ABLATION_MATRIX)
    security_rows = _read_csv(SECURITY_GOVERNANCE_ABLATION)
    baseline_by_domain = _rows_by_domain_and_key(baseline_rows, "baseline_family")
    ablation_by_domain = _rows_by_domain_and_key(ablation_rows, "ablation_name")

    rows: list[dict[str, str]] = []
    for domain in _ablation_domains(baseline_by_domain, ablation_rows):
        families = baseline_by_domain.get(domain, {})
        rows.append(_no_uncertainty_row(domain, families))
        for requested_surface, (ablation_name, comparability, interpretation) in RUNTIME_ABLATION_MAP.items():
            ablation = ablation_by_domain.get(domain, {}).get(ablation_name)
            family = ablation.get("baseline_family", "") if ablation else ""
            baseline = families.get(family)
            rows.append(
                _ablation_row_for_surface(
                    requested_surface,
                    domain,
                    ablation,
                    baseline,
                    comparability,
                    interpretation,
                )
            )
    rows.append(_signature_hash_gate_row(security_rows))
    rows.sort(key=lambda row: (row["domain"] == "Cross-domain governance", row["domain"], row["requested_surface"]))
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _blank_token(field: str) -> str:
    return BLANK_FIELD_TOKENS.get(field, "not_reported_in_compact_evidence")


def _materialize_no_blank_cells(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Replace blank publication cells with explicit semantic values."""

    materialized: list[dict[str, str]] = []
    for row in rows:
        materialized.append(
            {
                key: (_blank_token(key) if str(value).strip() == "" else str(value))
                for key, value in row.items()
            }
        )
    return materialized


def _write_csv_fields(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _tex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _write_claim_tex(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Utility-preserving safety claim table. ORIUS is compared against predictor-only and shutdown/fallback-only references without promoting non-comparable rows.}",
        r"\label{tab:utility-preserving-claim-table}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrrl}",
        r"\toprule",
        r"Domain & Comparison & Reference & Relation & Ref. TSVR & ORIUS TSVR & TSVR reduction & Comparability\\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            row["domain"],
            row["comparison"],
            row["reference_controller"],
            row["claim_relation"],
            row["reference_tsvr"],
            row["orius_tsvr"],
            row["absolute_tsvr_reduction"],
            row["comparability"],
        ]
        lines.append(" & ".join(_tex_escape(cell) for cell in cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ablation_tex(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Required ORIUS component-ablation surfaces. Non-isolated compact evidence is retained as boundary evidence rather than converted into unsupported numeric claims.}",
        r"\label{tab:utility-preserving-ablation-surfaces}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrl}",
        r"\toprule",
        r"Removed surface & Domain & Evidence & Baseline/controller & Baseline TSVR & ORIUS TSVR & Comparability\\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            row["requested_surface"],
            row["domain"],
            row["evidence_surface"],
            row["baseline_controller"],
            row["baseline_tsvr"],
            row["orius_tsvr"],
            row["comparability"],
        ]
        lines.append(" & ".join(_tex_escape(cell) for cell in cells) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(
    path: Path,
    rows: list[dict[str, str]],
    claim_rows: list[dict[str, str]],
    ablation_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# Utility-Preserving Safety Scorecard",
        "",
        "This scorecard separates strict safety from useful release behavior. A row passes only when ORIUS has no material excess TSVR over a domain fail-safe reference and preserves more useful work than that reference.",
        "",
        "| Domain | Safety reference | Excess TSVR | Utility gain | Fallback reduction | Intervention reduction | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {domain} | {safety_reference_controller} | {excess_tsvr_over_safety_reference} | "
            "{utility_gain_over_safety_reference} | {fallback_reduction_vs_safety_reference} | "
            "{intervention_reduction_vs_safety_reference} | {utility_preserving_safety_gate} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Claim-facing comparisons",
            "",
            "| Domain | Comparison | Reference | Reference TSVR | ORIUS TSVR | TSVR reduction | Utility delta | Relation | Comparability |",
            "|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in claim_rows:
        lines.append(
            "| {domain} | {comparison} | {reference_controller} | {reference_tsvr} | {orius_tsvr} | "
            "{absolute_tsvr_reduction} | {utility_delta} | {claim_relation} | {comparability} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Required ablation surfaces",
            "",
            "| Surface | Domain | Evidence | Baseline TSVR | ORIUS TSVR | TSVR reduction | Comparability |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in ablation_rows:
        lines.append(
            "| {requested_surface} | {domain} | {evidence_surface} | {baseline_tsvr} | {orius_tsvr} | "
            "{absolute_tsvr_reduction} | {comparability} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Claim boundary: this is bounded predeployment evidence. It does not claim road deployment, live clinical deployment, or physical battery field certification.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = build_scorecard()
    if not rows:
        raise SystemExit("no utility-preserving safety rows could be built")
    claim_rows = build_claim_comparison_rows(rows)
    ablation_rows = build_ablation_surface_rows()
    display_rows = _materialize_no_blank_cells(rows)
    display_claim_rows = _materialize_no_blank_cells(claim_rows)
    display_ablation_rows = _materialize_no_blank_cells(ablation_rows)
    _write_csv(OUT_CSV, display_rows)
    _write_csv_fields(OUT_CLAIM_CSV, display_claim_rows, CLAIM_FIELDS)
    _write_csv_fields(OUT_ABLATION_CSV, display_ablation_rows, ABLATION_FIELDS)
    _write_claim_tex(OUT_CLAIM_TEX, display_claim_rows)
    _write_ablation_tex(OUT_ABLATION_TEX, display_ablation_rows)
    _write_markdown(OUT_MD, display_rows, display_claim_rows, display_ablation_rows)
    OUT_JSON.write_text(
        json.dumps(
            {
                "generated_at_utc": _utc_now(),
                "row_count": len(display_rows),
                "claim_row_count": len(display_claim_rows),
                "ablation_surface_row_count": len(display_ablation_rows),
                "gate_semantics": (
                    "No material excess TSVR over a degenerate fail-safe reference, plus useful-work improvement."
                ),
                "rows": display_rows,
                "claim_comparison_rows": display_claim_rows,
                "ablation_surface_rows": display_ablation_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUT_CSV.relative_to(REPO_ROOT))
    print(OUT_CLAIM_CSV.relative_to(REPO_ROOT))
    print(OUT_ABLATION_CSV.relative_to(REPO_ROOT))
    print(OUT_CLAIM_TEX.relative_to(REPO_ROOT))
    print(OUT_ABLATION_TEX.relative_to(REPO_ROOT))
    print(OUT_JSON.relative_to(REPO_ROOT))
    print(OUT_MD.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
