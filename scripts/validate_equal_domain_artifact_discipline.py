#!/usr/bin/env python3
"""Validate equal artifact-discipline outputs for Battery, AV, and Healthcare."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"

REQUIRED_DOMAINS = {
    "Battery Energy Storage",
    "Autonomous Vehicles",
    "Medical and Healthcare Monitoring",
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
FORBIDDEN = {
    "validation_harness",
    "diagnostic_cross_domain_proxy",
    "proxy_current_shared_harness",
    "missing",
    "missing_on_current_cross_domain_lane",
    "future_cross_domain_benchmark_extension",
}

DOMAIN_DIRS = {
    "Battery Energy Storage": REPO_ROOT / "reports" / "battery_av" / "battery",
    "Autonomous Vehicles": REPO_ROOT / "reports" / "orius_av" / "nuplan_bounded",
    "Medical and Healthcare Monitoring": REPO_ROOT / "reports" / "healthcare",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _forbidden(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in FORBIDDEN or lowered.startswith("future_")


def main() -> int:
    findings: list[str] = []
    gate_rows = _read_csv(PUBLICATION_DIR / "equal_domain_artifact_discipline.csv")
    if {row["domain"] for row in gate_rows} != REQUIRED_DOMAINS:
        findings.append("equal_domain_artifact_discipline.csv must contain exactly the three promoted domains")

    payload = json.loads((PUBLICATION_DIR / "equal_domain_artifact_discipline.json").read_text(encoding="utf-8"))
    if payload.get("claim_scope") != "equal_artifact_discipline_not_equal_universal_closure":
        findings.append("equal-domain JSON must preserve the non-universal-closure claim scope")

    for row in gate_rows:
        domain = row["domain"]
        for gate in (
            "artifact_discipline_gate",
            "runtime_native_gate",
            "theorem_gate",
            "proof_appendix_gate",
            "baseline_gate",
            "ablation_gate",
            "negative_control_gate",
            "utility_gate",
            "reproducibility_gate",
        ):
            if row[gate] != "True":
                findings.append(f"{domain}: {gate} is not passing ({row.get('blockers', '')})")

        domain_dir = DOMAIN_DIRS[domain]
        comparator_rows = _read_csv(domain_dir / "runtime_comparator_summary.csv")
        ablation_rows = _read_csv(domain_dir / "runtime_ablation_summary.csv")
        negative_rows = _read_csv(domain_dir / "runtime_negative_controls.csv")
        trace_rows = _read_csv(domain_dir / "runtime_comparator_traces.csv")
        if not trace_rows:
            findings.append(f"{domain}: runtime_comparator_traces.csv is empty")

        families = {item["baseline_family"] for item in comparator_rows}
        if not REQUIRED_BASELINE_FAMILIES <= families:
            findings.append(f"{domain}: missing required baseline families")
        for item in comparator_rows:
            if item["metric_surface"] != "runtime_denominator":
                findings.append(f"{domain}: comparator {item['baseline_family']} is not runtime_denominator")
            if _forbidden(item.get("metric_surface", "")) or _forbidden(item.get("evidence_status", "")):
                findings.append(f"{domain}: forbidden comparator surface/status in {item['baseline_family']}")
        if domain in {"Autonomous Vehicles", "Medical and Healthcare Monitoring"}:
            independent_rows = [
                item for item in comparator_rows
                if item.get("baseline_family") not in {"orius_full_stack", "degenerate_fallback_runtime"}
            ]
            controllers = [item.get("controller", "") for item in independent_rows]
            if any(item.get("independent_baseline") != "True" for item in independent_rows):
                findings.append(f"{domain}: claim-carrying baselines must be marked independent_baseline=True")
            if len(controllers) != len(set(controllers)):
                findings.append(f"{domain}: claim-carrying baselines must use unique controllers")

        by_family = {item["baseline_family"]: item for item in comparator_rows}
        orius = by_family.get("orius_full_stack", {})
        degenerate = by_family.get("degenerate_fallback_runtime", {})
        if _safe_float(orius.get("tsvr")) != 0.0:
            findings.append(f"{domain}: ORIUS TSVR must be zero on equal artifact surface")
        if _safe_float(orius.get("certificate_valid_rate")) != 1.0:
            findings.append(f"{domain}: ORIUS certificate_valid_rate must be 1.0")
        if _safe_float(orius.get("runtime_witness_pass_rate")) != 1.0:
            findings.append(f"{domain}: ORIUS runtime_witness_pass_rate must be 1.0")
        if _safe_float(orius.get("useful_work_total")) <= _safe_float(degenerate.get("useful_work_total")):
            findings.append(f"{domain}: ORIUS useful work must exceed degenerate fallback")
        if domain in {"Autonomous Vehicles", "Medical and Healthcare Monitoring"} and _safe_float(orius.get("fallback_activation_rate")) > 0.50:
            findings.append(f"{domain}: ORIUS fallback/max-alert rate must be <= 0.50")

        if {item["ablation_name"] for item in ablation_rows} != REQUIRED_ABLATIONS:
            findings.append(f"{domain}: ablation rows must exactly cover required ablations")
        for item in ablation_rows:
            if item["metric_surface"] != "runtime_denominator":
                findings.append(f"{domain}: ablation {item['ablation_name']} is not runtime_denominator")
            if _forbidden(item.get("metric_surface", "")) or _forbidden(item.get("evidence_status", "")):
                findings.append(f"{domain}: forbidden ablation surface/status in {item['ablation_name']}")

        if {item["control_name"] for item in negative_rows} != REQUIRED_NEGATIVE_CONTROLS:
            findings.append(f"{domain}: negative controls must exactly cover required controls")
        for item in negative_rows:
            if item["surface"] != "runtime_denominator":
                findings.append(f"{domain}: negative control {item['control_name']} is not runtime_denominator")
            if _forbidden(item.get("surface", "")) or _forbidden(item.get("status", "")):
                findings.append(f"{domain}: forbidden negative-control surface/status in {item['control_name']}")

    manifest_path = PUBLICATION_DIR / "equal_domain_reproducibility_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = [item["path"] for item in manifest.get("artifacts", []) if not item.get("exists")]
    if missing:
        findings.append(f"reproducibility manifest has missing artifacts: {missing}")

    if findings:
        print("[validate_equal_domain_artifact_discipline] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_equal_domain_artifact_discipline] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
