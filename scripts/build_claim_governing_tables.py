#!/usr/bin/env python3
"""Regenerate claim-governing runtime tables from compact publication evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"

DOMAIN_ORDER = (
    "Battery Energy Storage",
    "Autonomous Vehicles",
    "Medical and Healthcare Monitoring",
)

BENCHMARK_FIELDS = (
    "domain",
    "tier",
    "baseline_tsvr_mean",
    "orius_tsvr_mean",
    "relative_delta",
    "intervention_rate",
    "fallback_activation_rate",
    "certificate_valid_release_rate",
    "runtime_witness_pass_rate",
    "strict_runtime_gate",
    "runtime_latency_p95_ms",
)
EVIDENCE_FIELDS = ("domain", "active_dataset", "claim_boundary")

FINAL_RUNTIME_FIELDS = (
    "domain",
    "baseline_tsvr",
    "orius_tsvr",
    "relative_delta",
    "intervention",
    "fallback",
    "certificate_valid",
    "runtime_witness",
    "latency_p95_ms",
)

DISPLAY_DOMAIN = {
    "Battery Energy Storage": "Battery",
    "Autonomous Vehicles": "AV",
    "Medical and Healthcare Monitoring": "Healthcare",
}

DISPLAY_TIER = {
    "reference": "witness",
    "runtime_contract_closed": "bounded runtime",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"compact evidence input missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_domain_rows(path: Path, required_fields: tuple[str, ...]) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"compact evidence input has no rows: {path}")
    missing_fields = set(required_fields) - set(rows[0])
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"compact evidence input {path} is missing fields: {missing}")

    by_domain = {row["domain"]: row for row in rows}
    missing_domains = [domain for domain in DOMAIN_ORDER if domain not in by_domain]
    extra_domains = sorted(set(by_domain) - set(DOMAIN_ORDER))
    if missing_domains or extra_domains:
        raise ValueError(
            f"compact evidence input {path} must contain exactly promoted domains; "
            f"missing={missing_domains}, extra={extra_domains}"
        )
    return by_domain


def _tex_escape(value: Any) -> str:
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
    text = str(value)
    return "".join(replacements.get(char, char) for char in text)


def _as_float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except KeyError as exc:
        raise ValueError(f"missing numeric field {field!r} for {row.get('domain', '<unknown>')}") from exc
    except ValueError as exc:
        raise ValueError(
            f"invalid numeric field {field!r} for {row.get('domain', '<unknown>')}: {row.get(field)!r}"
        ) from exc


def _format_float(row: dict[str, str], field: str, places: int = 6) -> str:
    return f"{_as_float(row, field):.{places}f}"


def _format_percent(value: str | float, places: int) -> str:
    return f"{100.0 * float(value):.{places}f}%"


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _build_runtime_rows(benchmark_rows: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for domain in DOMAIN_ORDER:
        source = benchmark_rows[domain]
        rows.append(
            {
                "domain": domain,
                "baseline_tsvr": _format_float(source, "baseline_tsvr_mean"),
                "orius_tsvr": _format_float(source, "orius_tsvr_mean"),
                "relative_delta": _format_float(source, "relative_delta"),
                "intervention": _format_float(source, "intervention_rate"),
                "fallback": _format_float(source, "fallback_activation_rate"),
                "certificate_valid": _format_float(source, "certificate_valid_release_rate"),
                "runtime_witness": _format_float(source, "runtime_witness_pass_rate"),
                "latency_p95_ms": _format_float(source, "runtime_latency_p95_ms", 3),
            }
        )
    return rows


def _build_claim_governing_tex(
    benchmark_rows: dict[str, dict[str, str]],
    evidence_rows: dict[str, dict[str, str]],
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Claim-governing three-domain runtime evidence. Values are regenerated from compact publication summaries; long artifact paths remain in the appendix registers.}",
        r"\label{tab:claim-governing-runtime-evidence}",
        r"\begin{tabular}{@{}llrrrrrl@{}}",
        r"\toprule",
        r"Domain & Tier & Base TSVR & ORIUS TSVR & Interv. & Fallback & Cert. & Witness\\",
        r"\midrule",
    ]
    for domain in DOMAIN_ORDER:
        benchmark = benchmark_rows[domain]
        evidence = evidence_rows[domain]
        row = [
            DISPLAY_DOMAIN[domain],
            DISPLAY_TIER.get(benchmark["tier"], benchmark["tier"].replace("_", " ")),
            _format_float(benchmark, "baseline_tsvr_mean"),
            _format_float(benchmark, "orius_tsvr_mean"),
            _format_float(benchmark, "intervention_rate"),
            _format_float(benchmark, "fallback_activation_rate"),
            _format_float(benchmark, "certificate_valid_release_rate"),
            evidence["active_dataset"]
            .replace("German OPSD battery witness", "OPSD")
            .replace("nuPlan all-zip grouped replay", "nuPlan")
            .replace("MIMIC retrospective monitoring", "MIMIC"),
        ]
        lines.append(" & ".join(_tex_escape(cell) for cell in row) + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\footnotesize Boundaries: Battery is a bounded predeployment witness; AV is bounded nuPlan replay, not road deployment; Healthcare is retrospective monitoring, not live clinical validation. Sources are the compact publication evidence CSVs.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _build_executive_runtime_tex(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Executive three-domain ORIUS result summary. This table is claim-governing for the headline runtime denominator only; deployment boundaries are stated in the text.}",
        r"\label{tab:executive-runtime-evidence}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Domain & Baseline TSVR & ORIUS TSVR & Reduction & Fallback\\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            DISPLAY_DOMAIN[row["domain"]],
            f"{float(row['baseline_tsvr']):.6f}",
            f"{float(row['orius_tsvr']):.6f}",
            _format_percent(row["relative_delta"], 2),
            _format_percent(row["fallback"], 1),
        ]
        lines.append(" & ".join(_tex_escape(cell) for cell in cells) + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def _build_final_runtime_tex(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Final claim-governing runtime safety evidence across the three promoted ORIUS domains.}",
        r"\label{tbl:final-runtime-safety}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Domain & Baseline TSVR & ORIUS TSVR & Reduction & Fallback & Cert-valid & p95 ms\\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            row["domain"],
            f"{float(row['baseline_tsvr']):.6f}",
            f"{float(row['orius_tsvr']):.6f}",
            _format_percent(row["relative_delta"], 2),
            _format_percent(row["fallback"], 1),
            _format_percent(row["certificate_valid"], 1),
            f"{float(row['latency_p95_ms']):.3f}",
        ]
        lines.append(" & ".join(_tex_escape(cell) for cell in cells) + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_claim_governing_tables(
    *,
    publication_dir: Path = PUBLICATION_DIR,
    benchmark_path: Path | None = None,
    evidence_path: Path | None = None,
) -> list[Path]:
    """Build manuscript-facing claim tables from tracked compact evidence."""

    benchmark_path = benchmark_path or publication_dir / "three_domain_ml_benchmark.csv"
    evidence_path = evidence_path or publication_dir / "three_domain_forecast_calibration_runtime_evidence.csv"
    benchmark_rows = _read_domain_rows(benchmark_path, BENCHMARK_FIELDS)
    evidence_rows = _read_domain_rows(evidence_path, EVIDENCE_FIELDS)

    runtime_rows = _build_runtime_rows(benchmark_rows)
    outputs = [
        publication_dir / "claim_governing_three_domain_runtime_evidence.tex",
        publication_dir / "final_runtime_safety_for_paper.csv",
        publication_dir / "tbl_final_runtime_safety.tex",
        publication_dir / "tbl_executive_runtime_evidence.tex",
    ]

    _write_text(outputs[0], _build_claim_governing_tex(benchmark_rows, evidence_rows))
    _write_csv(outputs[1], runtime_rows, FINAL_RUNTIME_FIELDS)
    _write_text(outputs[2], _build_final_runtime_tex(runtime_rows))
    _write_text(outputs[3], _build_executive_runtime_tex(runtime_rows))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publication-dir",
        type=Path,
        default=PUBLICATION_DIR,
        help="Directory containing compact publication evidence CSV inputs.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Compact three-domain ML benchmark CSV. Defaults under --publication-dir.",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=None,
        help="Compact forecast/calibration/runtime evidence CSV. Defaults under --publication-dir.",
    )
    args = parser.parse_args()

    written = build_claim_governing_tables(
        publication_dir=args.publication_dir,
        benchmark_path=args.benchmark,
        evidence_path=args.evidence,
    )
    for path in written:
        print(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
