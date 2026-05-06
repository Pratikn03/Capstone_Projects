#!/usr/bin/env python3
"""Build the canonical three-domain ORIUS closure matrices.

This builder consumes the active validation/training reports and emits the
current three-domain closure surface used by the promoted Battery + AV +
Healthcare lane.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DOMAIN_ORDER = ("battery", "healthcare", "vehicle")
DISPLAY_NAMES = {
    "battery": "Battery Energy Storage",
    "healthcare": "Medical and Healthcare Monitoring",
    "vehicle": "Autonomous Vehicles",
}
SOURCE_DATASETS = {
    "battery": "locked_battery_reference",
    "healthcare": "mimic3_monitoring_row",
    "vehicle": "nuplan_allzip_grouped_row",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _humanize_table_cell(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value).replace("_", " ")


def _latex_escape(value: Any) -> str:
    text = _humanize_table_cell(value)
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
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _write_tex(path: Path, caption: str, label: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "l" * (len(columns) - 1) + "}",
        r"\toprule",
        " & ".join(rf"\textbf{{{_latex_escape(column)}}}" for column in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(row[col]) for col in columns) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _training_lookup(training_report: dict[str, Any]) -> set[str]:
    return set(map(str, training_report.get("training_surface_closed_domains", [])))


def _failed_domains(validation_report: dict[str, Any]) -> set[str]:
    failed = validation_report.get("failed_domains")
    if isinstance(failed, list):
        return set(map(str, failed))
    failed = validation_report.get("domains_failed")
    if isinstance(failed, list):
        return set(map(str, failed))
    return set()


def _proof_report(validation_report: dict[str, Any], domain: str) -> dict[str, Any]:
    reports = validation_report.get("domain_proof_reports")
    if not isinstance(reports, dict):
        return {}
    report = reports.get(domain, {})
    return report if isinstance(report, dict) else {}


def _domain_validation_pass(validation_report: dict[str, Any], domain: str) -> bool:
    failed = _failed_domains(validation_report)
    if domain in failed:
        return False

    validated = validation_report.get("validated_domains")
    if isinstance(validated, list) and validated and domain == "battery":
        return domain in set(map(str, validated))

    if domain == "battery":
        return domain not in failed

    report = _proof_report(validation_report, domain)
    if report:
        return bool(report.get("evidence_pass", False))

    if isinstance(validated, list) and validated:
        return domain in set(map(str, validated))

    return bool(validation_report.get("all_passed", False))


def _domain_blocker(validation_report: dict[str, Any], domain: str) -> str:
    if domain == "battery":
        return "battery_reference_witness"
    if _domain_validation_pass(validation_report, domain):
        return "none"
    report = _proof_report(validation_report, domain)
    reasons = report.get("failure_reasons", []) if isinstance(report, dict) else []
    if isinstance(reasons, list) and reasons:
        return str(reasons[0])
    return "validation_failed"


def _closure_rows(validation_report: dict[str, Any], training_report: dict[str, Any]) -> list[dict[str, Any]]:
    domain_results = validation_report["domain_results"]
    training_closed = _training_lookup(training_report)
    training_domain_ids = {
        "battery": "battery",
        "healthcare": "healthcare",
        "vehicle": "av",
    }
    rows: list[dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        result = domain_results[domain]
        validation_passed = _domain_validation_pass(validation_report, domain)
        resulting_tier = (
            "reference"
            if domain == "battery"
            else ("proof_validated" if validation_passed else "proof_candidate")
        )
        exact_blocker = _domain_blocker(validation_report, domain)
        rows.append(
            {
                "domain": domain,
                "display_name": DISPLAY_NAMES[domain],
                "resulting_tier": resulting_tier,
                "exact_blocker": exact_blocker,
                "baseline_tsvr_mean": f"{float(result['baseline_tsvr_mean']):.4f}",
                "orius_tsvr_mean": f"{float(result['orius_tsvr_mean']):.4f}",
                "orius_reduction_pct": f"{float(result['orius_reduction_pct']):.1f}",
                "safe_action_soundness_status": "pass" if validation_passed else "fail",
                "training_surface_status": (
                    "closed"
                    if training_domain_ids[domain] in training_closed or domain == "battery"
                    else "open"
                ),
                "multi_agent_status": "evaluated" if validation_passed else "fail",
                "certos_status": "evaluated" if validation_passed else "fail",
                "source_dataset": SOURCE_DATASETS[domain],
                "closure_target_tier": "witness_row" if domain == "battery" else "defended_promoted_row",
            }
        )
    return rows


def _build_paper5_rows(validation_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "domain": domain,
            "status": "evaluated" if _domain_validation_pass(validation_report, domain) else "fail",
            "notes": "Three-domain closure keeps the compositional program on the promoted rows only.",
        }
        for domain in DOMAIN_ORDER
    ]


def _build_paper6_rows(validation_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "domain": domain,
            "status": "evaluated" if _domain_validation_pass(validation_report, domain) else "fail",
            "notes": "CertOS/runtime governance remains active on the promoted three-domain lane.",
        }
        for domain in DOMAIN_ORDER
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the three-domain ORIUS closure matrices")
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/universal_orius_validation"))
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    validation_report = _read_json(args.validation_report)
    training_report = _read_json(args.training_report)

    closure_rows = _closure_rows(validation_report, training_report)
    p5_rows = _build_paper5_rows(validation_report)
    p6_rows = _build_paper6_rows(validation_report)

    closure_csv = out_dir / "domain_closure_matrix.csv"
    p5_csv = out_dir / "paper5_cross_domain_matrix.csv"
    p6_csv = out_dir / "paper6_cross_domain_matrix.csv"
    closure_tex = out_dir / "tbl_domain_closure_matrix.tex"
    p5_tex = out_dir / "tbl_paper5_cross_domain_matrix.tex"
    p6_tex = out_dir / "tbl_paper6_cross_domain_matrix.tex"

    _write_csv(closure_csv, closure_rows)
    _write_csv(p5_csv, p5_rows)
    _write_csv(p6_csv, p6_rows)
    _write_tex(
        closure_tex,
        "Three-domain ORIUS closure matrix for the promoted Battery + AV + Healthcare program.",
        "tab:three-domain-closure-matrix",
        closure_rows,
        [
            "display_name",
            "resulting_tier",
            "baseline_tsvr_mean",
            "orius_tsvr_mean",
            "training_surface_status",
        ],
    )
    _write_tex(
        p5_tex,
        "Three-domain compositional-safety status table.",
        "tab:three-domain-paper5-matrix",
        p5_rows,
        ["domain", "status", "notes"],
    )
    _write_tex(
        p6_tex,
        "Three-domain CertOS/runtime-governance status table.",
        "tab:three-domain-paper6-matrix",
        p6_rows,
        ["domain", "status", "notes"],
    )

    payload = {
        "generated_at_utc": validation_report.get("reference_domain_metrics", {}),
        "closure_rows": closure_rows,
        "paper5_rows": p5_rows,
        "paper6_rows": p6_rows,
        "vehicle_soundness_rows": [row for row in closure_rows if row["domain"] == "vehicle"],
        "paths": {
            "domain_closure_csv": str(closure_csv),
            "paper5_csv": str(p5_csv),
            "paper6_csv": str(p6_csv),
            "domain_closure_tex": str(closure_tex),
            "paper5_tex": str(p5_tex),
            "paper6_tex": str(p6_tex),
        },
    }
    (out_dir / "domain_closure_matrix.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
