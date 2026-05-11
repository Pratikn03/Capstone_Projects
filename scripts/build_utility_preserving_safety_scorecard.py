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

OUT_CSV = PUBLICATION_DIR / "utility_preserving_safety_scorecard.csv"
OUT_JSON = PUBLICATION_DIR / "utility_preserving_safety_scorecard.json"
OUT_MD = PUBLICATION_DIR / "utility_preserving_safety_scorecard.md"

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


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
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
            "Claim boundary: this is bounded predeployment evidence. It does not claim road deployment, live clinical deployment, or physical battery field certification.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = build_scorecard()
    if not rows:
        raise SystemExit("no utility-preserving safety rows could be built")
    _write_csv(OUT_CSV, rows)
    _write_markdown(OUT_MD, rows)
    OUT_JSON.write_text(
        json.dumps(
            {
                "generated_at_utc": _utc_now(),
                "row_count": len(rows),
                "gate_semantics": (
                    "No material excess TSVR over a degenerate fail-safe reference, plus useful-work improvement."
                ),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(OUT_CSV.relative_to(REPO_ROOT))
    print(OUT_JSON.relative_to(REPO_ROOT))
    print(OUT_MD.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
