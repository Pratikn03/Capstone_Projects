#!/usr/bin/env python3
"""Audit table/result artifacts for publishable placeholder hygiene.

The default audit covers claim-governing publication and promoted domain
surfaces, not every local run cache.  Numeric zero values are reported as
warnings only because zero is a valid safety result for ORIUS violation/failure
metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOTS = (
    "paper/assets/tables",
    "reports/publication",
    "reports/battery_av/battery",
    "reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest",
    "reports/healthcare/heldout_95",
    "reports/hil/software_hil_95",
)

# Strict blocking is intentionally limited to the current claim-governing
# publication surfaces.  Broader generated, archived, and historical report
# surfaces are still scanned by default, but their findings are downgraded to
# non-blocking warnings so the release gate reflects the active claim surface
# instead of old run caches.
CURRENT_CLAIM_GOVERNING_SURFACES = {
    "paper/assets/tables/tbl01_main_results.csv",
    "paper/assets/tables/tbl02_ablations.csv",
    "paper/assets/tables/tbl03_cqr_group_coverage.csv",
    "paper/assets/tables/tbl04_transfer_stress.csv",
    "paper/assets/tables/tbl05_dataset_summary.csv",
    "paper/assets/tables/tbl06_hyperparams.csv",
    "paper/assets/tables/tbl07_dataset_cards.csv",
    "paper/assets/tables/tbl08_forecast_baselines.csv",
    "reports/publication/claim_governing_three_domain_runtime_evidence.tex",
    "reports/publication/final_freeze_validation_for_paper.csv",
    "reports/publication/final_paper_results_summary.json",
    "reports/publication/final_runtime_safety_for_paper.csv",
    "reports/publication/final_training_quality_for_paper.csv",
    "reports/publication/novelty_separation_matrix.csv",
    "reports/publication/novelty_separation_matrix.json",
    "reports/publication/orius_cross_domain_design_principles.csv",
    "reports/publication/orius_domain_closure_matrix.csv",
    "reports/publication/orius_failure_modes_falsification_table.csv",
    "reports/publication/orius_framework_gap_matrix.csv",
    "reports/publication/orius_literature_matrix.csv",
    "reports/publication/orius_maturity_matrix.csv",
    "reports/publication/orius_module_claim_crosswalk.csv",
    "reports/publication/orius_monograph_chapter_map.csv",
    "reports/publication/orius_publication_artifact_index.csv",
    "reports/publication/orius_universal_claim_matrix.csv",
    "reports/publication/runtime_release_contract_witnesses.csv",
    "reports/publication/runtime_release_contract_witnesses.json",
    "reports/publication/runtime_release_contract_witnesses.tex",
    "reports/publication/security_governance_ablation_matrix.csv",
    "reports/publication/theorem_defensibility_10.json",
    "reports/publication/theorem_promotion_matrix.json",
    "reports/publication/three_domain_ablation_matrix.csv",
    "reports/publication/three_domain_baseline_suite.csv",
    "reports/publication/three_domain_forecast_calibration_runtime_evidence.csv",
    "reports/publication/three_domain_forecast_calibration_runtime_evidence.json",
    "reports/publication/three_domain_grouped_coverage.csv",
    "reports/publication/three_domain_grouped_width.csv",
    "reports/publication/three_domain_ml_benchmark.csv",
    "reports/publication/three_domain_ml_benchmark_summary.json",
    "reports/publication/three_domain_negative_controls.csv",
    "reports/publication/three_domain_nonvacuity_checks.json",
    "reports/publication/three_domain_reliability_calibration.csv",
    "reports/publication/three_domain_utility_safety_dominance.csv",
    "reports/publication/three_domain_utility_safety_dominance.json",
    "reports/publication/tbl_final_freeze_validation.tex",
    "reports/publication/tbl_final_runtime_safety.tex",
    "reports/publication/tbl_final_training_quality.tex",
    "reports/publication/tbl_final_utility_preserving_safety.tex",
    "reports/publication/utility_preserving_safety_ablation_surfaces.csv",
    "reports/publication/utility_preserving_safety_ablation_surfaces.tex",
    "reports/publication/utility_preserving_safety_claim_table.csv",
    "reports/publication/utility_preserving_safety_claim_table.tex",
    "reports/publication/utility_preserving_safety_scorecard.csv",
    "reports/publication/utility_preserving_safety_scorecard.json",
    "reports/publication/what_orius_is_not_matrix.csv",
    "reports/publication/what_orius_is_not_matrix.json",
}
CURRENT_CLAIM_GOVERNING_PREFIXES = (
    "reports/publication/theorem_result_cards/",
)
NONBLOCKING_SURFACE_DETAIL = (
    "Non-current historical/generated surface; audited for visibility but excluded "
    "from the strict current-publication gate."
)
CSV_SEMANTIC_BLANK_COLUMNS_BY_SURFACE_AND_DOMAIN = {
    "reports/publication/utility_preserving_safety_scorecard.csv": {
        "Battery Energy Storage": {
            "orius_intervention_rate",
            "safety_reference_intervention_rate",
            "intervention_reduction_vs_safety_reference",
            "orius_fallback_rate",
            "safety_reference_fallback_rate",
            "fallback_reduction_vs_safety_reference",
        },
    },
    "reports/publication/three_domain_utility_safety_dominance.csv": {
        "Medical and Healthcare Monitoring": {
            "orius_progress_total",
            "orius_near_miss_rate",
            "orius_collision_proxy_rate",
            "orius_mean_abs_jerk",
        },
    },
}
JSON_SEMANTIC_BLANK_COLUMNS_BY_SURFACE_AND_DOMAIN = {
    source.replace(".csv", ".json"): mapping
    for source, mapping in CSV_SEMANTIC_BLANK_COLUMNS_BY_SURFACE_AND_DOMAIN.items()
}
UTILITY_SCORECARD_JSON_OPTIONAL_COLUMNS = {
    ("claim_comparison_rows", "shutdown_or_fallback_only_conservatism"): {
        "reference_intervention_rate",
        "orius_intervention_rate",
        "intervention_delta",
        "reference_fallback_rate",
        "orius_fallback_rate",
        "fallback_delta",
    },
    ("claim_comparison_rows", "predictor_only_safety"): {
        "reference_utility",
        "orius_utility",
        "utility_delta",
        "reference_fallback_rate",
        "orius_fallback_rate",
        "fallback_delta",
    },
    ("ablation_surface_rows", "no_signature_hash_gate"): {
        "baseline_controller",
        "baseline_tsvr",
        "orius_tsvr",
        "absolute_tsvr_reduction",
        "relative_tsvr_reduction",
        "baseline_intervention_rate",
        "orius_intervention_rate",
    },
}
SKIP_PARTS = {
    ".git",
    ".venv",
    "node_modules",
    ".next",
    "__pycache__",
    ".pytest_cache",
}

BLOCKING_TOKENS = {
    "",
    "_",
    "---",
    "--",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "tbd",
    "todo",
    "placeholder",
}
SEMANTIC_TOKENS = {
    "not_applicable",
    "not_reported",
    "not_run",
    "not_canonical",
    "verified_existing_artifacts",
    "no_failure",
    "no_failed_obligations",
    "no_intervention",
    "none_required",
    "no_open_blocker",
    "no_open_gap",
    "source_location_not_required",
    "external_review_owner_pending",
    "pending_external_review_artifact",
    "pending_artifact",
    "no_release_id",
    "no_supplemental_hf_artifact_required",
}
NONCANONICAL_ROW_TOKENS = {
    "not_canonical",
    "pending_artifact",
    "pending_external_review_artifact",
    "not_run",
}
MISSING_CONTEXT_MARKERS = (
    "artifact",
    "blocker",
    "closure",
    "command",
    "evidence",
    "fail",
    "finding",
    "location",
    "obligation",
    "owner",
    "provenance",
    "reason",
    "source",
    "status",
)
SYNTHETIC_NUMERIC_COLUMNS = {
    "adaptive_quantile",
    "conditional_coverage_gap",
    "e_t_mwh",
    "expires_at_step",
    "gamma_mw",
    "half_life_steps",
    "soc_tube_lower_mwh",
    "soc_tube_upper_mwh",
    "validity_horizon_h_t",
    "validity_score",
}
SYNTHETIC_TIMESTAMP_COLUMNS = {"expires_at", "timeout_at"}
ZERO_REVIEW_MARKERS = (
    "auc",
    "coverage",
    "cost",
    "error",
    "latency",
    "loss",
    "mae",
    "picp",
    "rate",
    "rmse",
    "score",
    "severity",
    "violation",
    "width",
)
MAX_SCAN_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    severity: str
    blocking: bool
    source_type: str
    path: str
    issue: str
    column: str = ""
    table: str = ""
    row_index: str = ""
    value: str = ""
    detail: str = ""
    recommendation: str = ""

    def as_row(self) -> dict[str, Any]:
        def filled(value: Any, default: str = "not_applicable") -> Any:
            if isinstance(value, str) and value == "":
                return default
            return value

        return {
            "severity": self.severity,
            "blocking": self.blocking,
            "source_type": self.source_type,
            "path": self.path,
            "table": filled(self.table),
            "column": filled(self.column),
            "row_index": filled(self.row_index),
            "issue": self.issue,
            "value": filled(self.value),
            "detail": filled(self.detail),
            "recommendation": filled(self.recommendation, "none_required"),
        }


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _should_skip(path: Path) -> bool:
    if path.name.startswith("._") or any(part in SKIP_PARTS for part in path.parts):
        return True
    if path.is_symlink():
        return True
    try:
        return path.stat().st_size > MAX_SCAN_BYTES
    except OSError:
        return True


def _iter_files(roots: Iterable[str]) -> Iterable[Path]:
    for root in roots:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        if base.is_file():
            if not _should_skip(base):
                yield base
            continue
        for path in base.rglob("*"):
            if path.is_file() and not _should_skip(path):
                yield path


def _looks_missing(value: Any, column: str = "") -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    text = str(value).strip()
    lower = text.lower()
    if lower in SEMANTIC_TOKENS:
        return False
    if lower == "none":
        if "blocker" in column.lower():
            return False
        return any(marker in column.lower() for marker in MISSING_CONTEXT_MARKERS)
    return lower in BLOCKING_TOKENS


def _is_current_claim_governing_surface(path: Path) -> bool:
    source = rel(path)
    return source in CURRENT_CLAIM_GOVERNING_SURFACES or any(
        source.startswith(prefix) for prefix in CURRENT_CLAIM_GOVERNING_PREFIXES
    )


def _as_nonblocking_context(finding: Finding) -> Finding:
    detail = finding.detail
    if not detail:
        detail = NONBLOCKING_SURFACE_DETAIL
    elif NONBLOCKING_SURFACE_DETAIL not in detail:
        detail = f"{detail} {NONBLOCKING_SURFACE_DETAIL}"
    return replace(finding, severity="warning", blocking=False, detail=detail)


def _apply_strict_scope(
    path: Path, findings: list[Finding], *, strict_current_only: bool
) -> list[Finding]:
    if not strict_current_only or _is_current_claim_governing_surface(path):
        return findings
    return [_as_nonblocking_context(finding) for finding in findings]


def _csv_semantic_blank_allowed(source: str, column: str, row: pd.Series) -> bool:
    by_domain = CSV_SEMANTIC_BLANK_COLUMNS_BY_SURFACE_AND_DOMAIN.get(source)
    if not by_domain:
        return False
    domain = str(row.get("domain", "")).strip()
    return column in by_domain.get(domain, set())


def _json_semantic_blank_allowed(source: str, root: Any, parts: list[str]) -> bool:
    if (
        source == "reports/publication/theorem_defensibility_10.json"
        and _theorem_defensibility_blank_allowed(root, parts)
    ):
        return True
    by_domain = JSON_SEMANTIC_BLANK_COLUMNS_BY_SURFACE_AND_DOMAIN.get(source)
    if (
        source == "reports/publication/utility_preserving_safety_scorecard.json"
        and _utility_scorecard_json_blank_allowed(root, parts)
    ):
        return True
    if not by_domain or len(parts) < 3 or parts[0] != "rows" or not parts[1].isdigit():
        return False
    rows = root.get("rows") if isinstance(root, dict) else None
    if not isinstance(rows, list):
        return False
    row_index = int(parts[1])
    if row_index >= len(rows) or not isinstance(rows[row_index], dict):
        return False
    domain = str(rows[row_index].get("domain", "")).strip()
    return parts[-1] in by_domain.get(domain, set())


def _utility_scorecard_json_blank_allowed(root: Any, parts: list[str]) -> bool:
    if len(parts) < 3 or not isinstance(root, dict) or not parts[1].isdigit():
        return False
    section = parts[0]
    rows = root.get(section)
    if not isinstance(rows, list):
        return False
    row_index = int(parts[1])
    if row_index >= len(rows) or not isinstance(rows[row_index], dict):
        return False
    row = rows[row_index]
    discriminator = str(row.get("comparison") or row.get("requested_surface") or "").strip()
    return parts[-1] in UTILITY_SCORECARD_JSON_OPTIONAL_COLUMNS.get((section, discriminator), set())


def _theorem_defensibility_blank_allowed(root: Any, parts: list[str]) -> bool:
    if parts != ["formal", "lake_output"] or not isinstance(root, dict):
        return False
    formal = root.get("formal")
    if not isinstance(formal, dict):
        return False
    checks = formal.get("checks")
    return (
        isinstance(checks, dict)
        and formal.get("pass") is True
        and checks.get("formal_core_lake_build") is True
    )


def _semantic_recommendation(column: str) -> str:
    col = column.lower()
    if "failed_obligation" in col:
        return "Use no_failed_obligations for passing rows."
    if "failure" in col or "reason" in col:
        return "Use no_failure or a concrete failure reason."
    if "train_command" in col:
        return "Use verified_existing_artifacts or the exact training command."
    if "owner" in col:
        return "Use external_review_owner_pending or a concrete owner."
    if "closure" in col:
        return "Use pending_external_review_artifact or a closure artifact path."
    if "source" in col or "artifact" in col:
        return "Use a concrete artifact path or not_canonical."
    return "Replace with an explicit semantic value such as not_applicable."


def _scan_csv(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    source = rel(path)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    except Exception as exc:
        return [
            Finding(
                "error",
                True,
                "csv",
                source,
                "csv_read_error",
                detail=str(exc),
                recommendation="Repair CSV syntax or regenerate the artifact.",
            )
        ]

    if df.empty:
        return [
            Finding(
                "error",
                True,
                "csv",
                source,
                "empty_csv_rows",
                detail="CSV has a header but no semantic data row.",
                recommendation="Add an explicit status row or remove it from active surfaces.",
            )
        ]

    for column in df.columns:
        series = df[column].map(lambda value: str(value).strip())
        missing_mask = pd.Series(
            [
                _looks_missing(value, str(column))
                and not _csv_semantic_blank_allowed(source, str(column), df.loc[index])
                for index, value in series.items()
            ],
            index=df.index,
        )
        count = int(missing_mask.sum())
        if count:
            sample_index = str(int(missing_mask[missing_mask].index[0]) + 2)
            sample = str(series[missing_mask].iloc[0])
            findings.append(
                Finding(
                    "error",
                    True,
                    "csv",
                    source,
                    "placeholder_or_blank_cell",
                    column=str(column),
                    row_index=sample_index,
                    value=sample,
                    detail=f"{count}/{len(series)} rows contain missing/placeholder values.",
                    recommendation=_semantic_recommendation(str(column)),
                )
            )

        numeric = pd.to_numeric(df[column], errors="coerce")
        numeric_count = int(numeric.notna().sum())
        if numeric_count >= 3 and bool((numeric.fillna(0) == 0).all()):
            col_lower = str(column).lower()
            if any(marker in col_lower for marker in ZERO_REVIEW_MARKERS):
                findings.append(
                    Finding(
                        "warning",
                        False,
                        "csv",
                        source,
                        "all_zero_metric_review",
                        column=str(column),
                        detail=f"{numeric_count} numeric values are all zero.",
                        recommendation="Review for semantic zero vs. placeholder zero; not blocking.",
                    )
                )
    return findings


def _scan_json(path: Path) -> list[Finding]:
    source = rel(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            Finding(
                "error",
                True,
                "json",
                source,
                "json_read_error",
                detail=str(exc),
                recommendation="Repair JSON syntax or regenerate the artifact.",
            )
        ]

    findings: list[Finding] = []

    def walk(value: Any, parts: list[str]) -> None:
        key = parts[-1] if parts else ""
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, [*parts, str(child_key)])
        elif isinstance(value, list):
            for index, child_value in enumerate(value):
                walk(child_value, [*parts, str(index)])
        elif _looks_missing(value, key) and not _json_semantic_blank_allowed(source, payload, parts):
            findings.append(
                Finding(
                    "error",
                    True,
                    "json",
                    source,
                    "placeholder_or_null_value",
                    column=".".join(parts),
                    value="null" if value is None else str(value),
                    recommendation=_semantic_recommendation(key),
                )
            )

    walk(payload, [])
    return findings


TEX_PLACEHOLDER_RE = re.compile(
    r"(?i)(\bNaN\b|\bNone\b|(?<![A-Za-z-])null\b|TODO|TBD|PLACEHOLDER|&\s*(---|--|N/A|n/a|_)\s*(?=&|\\\\)|&\s*(?=&|\\\\))"
)


def _scan_tex(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    source = rel(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if TEX_PLACEHOLDER_RE.search(line):
            findings.append(
                Finding(
                    "error",
                    True,
                    "tex",
                    source,
                    "rendered_placeholder_cell",
                    row_index=str(line_no),
                    value=line.strip()[:220],
                    recommendation="Render an explicit semantic value in the table cell.",
                )
            )
            break
    return findings


def _scan_duckdb(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    source = rel(path)
    try:
        con = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        return [
            Finding(
                "error",
                True,
                "duckdb",
                source,
                "duckdb_open_error",
                detail=str(exc),
                recommendation="Repair or regenerate the audit database.",
            )
        ]
    try:
        tables = [str(row[0]) for row in con.execute("SHOW TABLES").fetchall()]
        for table in tables:
            try:
                df = con.execute(f'SELECT * FROM "{table}"').df()
            except Exception as exc:
                findings.append(
                    Finding(
                        "error",
                        True,
                        "duckdb",
                        source,
                        "duckdb_table_read_error",
                        table=table,
                        detail=str(exc),
                    )
                )
                continue
            if df.empty:
                findings.append(
                    Finding(
                        "error",
                        True,
                        "duckdb",
                        source,
                        "empty_table",
                        table=table,
                        recommendation="Regenerate or drop stale empty audit table.",
                    )
                )
                continue
            noncanonical_mask = _duckdb_noncanonical_mask(df)
            for column in df.columns:
                allowed_null_mask = _duckdb_allowed_null_mask(df, str(column))
                unresolved_null_mask = df[column].isna() & ~noncanonical_mask & ~allowed_null_mask
                na_count = int(unresolved_null_mask.sum())
                if na_count:
                    findings.append(
                        Finding(
                            "error",
                            True,
                            "duckdb",
                            source,
                            "null_table_value",
                            table=table,
                            column=str(column),
                            detail=f"{na_count}/{len(df)} canonical rows are null.",
                            recommendation=_semantic_recommendation(str(column)),
                        )
                    )
                if pd.api.types.is_numeric_dtype(df[column]):
                    numeric = pd.to_numeric(df.loc[~noncanonical_mask, column], errors="coerce")
                    if str(column).lower() in SYNTHETIC_NUMERIC_COLUMNS:
                        sentinel_count = int((numeric == -1).sum())
                        if sentinel_count:
                            findings.append(
                                Finding(
                                    "error",
                                    True,
                                    "duckdb",
                                    source,
                                    "synthetic_fixed_result_value",
                                    table=table,
                                    column=str(column),
                                    value="-1",
                                    detail=f"{sentinel_count} canonical rows use a fixed sentinel instead of derived evidence.",
                                    recommendation="Regenerate from runtime/training artifacts or mark the rows noncanonical.",
                                )
                            )
                    if int(numeric.notna().sum()) >= 3 and bool((numeric.fillna(0) == 0).all()):
                        col_lower = str(column).lower()
                        if any(marker in col_lower for marker in ZERO_REVIEW_MARKERS):
                            findings.append(
                                Finding(
                                    "warning",
                                    False,
                                    "duckdb",
                                    source,
                                    "all_zero_metric_review",
                                    table=table,
                                    column=str(column),
                                    detail=f"{int(numeric.notna().sum())} numeric values are all zero.",
                                )
                            )
                elif str(column).lower() in SYNTHETIC_TIMESTAMP_COLUMNS:
                    timestamp_text = df.loc[~noncanonical_mask, column].astype(str)
                    sentinel_count = int(timestamp_text.str.startswith("1970-01-01").sum())
                    if sentinel_count:
                        findings.append(
                            Finding(
                                "error",
                                True,
                                "duckdb",
                                source,
                                "synthetic_fixed_result_value",
                                table=table,
                                column=str(column),
                                value="1970-01-01",
                                detail=f"{sentinel_count} canonical rows use an epoch sentinel instead of a real timestamp.",
                                recommendation="Regenerate the timestamp or leave it null where the event is not applicable.",
                            )
                        )
    finally:
        con.close()
    return findings


def _duckdb_noncanonical_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    status_like = [
        column
        for column in df.columns
        if any(
            marker in str(column).lower() for marker in ("status", "surface", "closure", "tier", "canonical")
        )
    ]
    for column in status_like:
        values = df[column].astype(str).str.strip().str.lower()
        mask |= values.isin(NONCANONICAL_ROW_TOKENS)
    return mask


def _duckdb_allowed_null_mask(df: pd.DataFrame, column: str) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    col = column.lower()
    if col in {"expires_at", "timeout_at"} and "status" in df.columns:
        status = df["status"].astype(str).str.strip().str.lower()
        mask |= status.isin({"acked", "complete", "completed"})
        if "timeout_reason" in df.columns:
            reason = df["timeout_reason"].astype(str).str.strip().str.lower()
            mask |= reason.isin({"no_failure", "not_applicable", "none_required"})
    return mask


def run_audit(
    roots: Iterable[str], *, strict_current_only: bool = True
) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    scanned = {"csv": 0, "json": 0, "tex": 0, "duckdb": 0}
    for path in _iter_files(roots):
        suffix = path.suffix.lower()
        path_findings: list[Finding] = []
        if suffix == ".csv":
            scanned["csv"] += 1
            path_findings = _scan_csv(path)
        elif suffix == ".json":
            scanned["json"] += 1
            path_findings = _scan_json(path)
        elif suffix == ".tex":
            scanned["tex"] += 1
            path_findings = _scan_tex(path)
        elif suffix == ".duckdb" and path.parent.name == "audit":
            scanned["duckdb"] += 1
            path_findings = _scan_duckdb(path)
        findings.extend(_apply_strict_scope(path, path_findings, strict_current_only=strict_current_only))

    blocking = [finding for finding in findings if finding.blocking]
    warnings = [finding for finding in findings if not finding.blocking]
    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "roots": list(roots),
        "strict_scope": "current_claim_governing_surfaces" if strict_current_only else "all_scanned_surfaces",
        "scanned": scanned,
        "finding_count": len(findings),
        "blocking_count": len(blocking),
        "warning_count": len(warnings),
        "passes": len(blocking) == 0,
        "top_blocking": [finding.as_row() for finding in blocking[:50]],
        "top_warnings": [finding.as_row() for finding in warnings[:50]],
    }
    return findings, summary


def _write_outputs(findings: list[Finding], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [finding.as_row() for finding in findings]
    if not rows:
        rows = [
            {
                "severity": "info",
                "blocking": False,
                "source_type": "audit",
                "path": "reports/audit/table_result_integrity.csv",
                "table": "not_applicable",
                "column": "not_applicable",
                "row_index": "not_applicable",
                "issue": "no_findings",
                "value": "not_applicable",
                "detail": "No table/result integrity findings were detected.",
                "recommendation": "none_required",
            }
        ]
    csv_path = out_dir / "table_result_integrity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(Finding("", False, "", "", "").as_row().keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "table_result_integrity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Table/Result Integrity Audit",
        "",
        f"- Generated: `{summary['generated_at_utc']}`",
        f"- Passes: `{summary['passes']}`",
        f"- Blocking findings: `{summary['blocking_count']}`",
        f"- Warning findings: `{summary['warning_count']}`",
        f"- Scanned: `{summary['scanned']}`",
        "",
    ]
    if summary["top_blocking"]:
        md_lines.append("## Top Blocking Findings")
        for finding in summary["top_blocking"][:25]:
            md_lines.append(
                f"- `{finding['path']}` `{finding['column']}`: {finding['issue']} ({finding['detail'] or finding['value']})"
            )
        md_lines.append("")
    if summary["top_warnings"]:
        md_lines.append("## Top Warnings")
        for finding in summary["top_warnings"][:25]:
            md_lines.append(
                f"- `{finding['path']}` `{finding['column']}`: {finding['issue']} ({finding['detail']})"
            )
    (out_dir / "table_result_integrity.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ORIUS table/result artifact integrity.")
    parser.add_argument(
        "--root", action="append", default=None, help="Repo-relative root to scan; may be repeated."
    )
    parser.add_argument("--out-dir", "--output-dir", default="reports/audit")
    parser.add_argument(
        "--no-fail", action="store_true", help="Write report but do not exit nonzero on blocking findings."
    )
    parser.add_argument(
        "--all-blocking",
        action="store_true",
        help="Treat findings from historical/generated surfaces as blocking too.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = args.root or list(DEFAULT_ROOTS)
    findings, summary = run_audit(roots, strict_current_only=not args.all_blocking)
    _write_outputs(findings, summary, REPO_ROOT / args.out_dir)
    print(json.dumps(summary, indent=2))
    if summary["blocking_count"] and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
